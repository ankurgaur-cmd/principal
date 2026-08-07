"""End-to-end tests through the real FastAPI app, with a stub provider.

The stub simulates the one behaviour that matters and cannot be faked by
inspection: a provider-side prompt cache. It returns cache-read usage when it
has seen a prefix before on the same model, and cache-write usage the first
time. That is enough to exercise routing, stickiness, the pilot, pricing, and
the ledger without spending a cent.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from aigateway.cache.hints import prefix_fingerprint
from aigateway.main import create_app
from aigateway.schemas import ProviderResponse, Usage
from aigateway.tokens import estimate_request_tokens

BIG_CONTEXT = "You are a senior code reviewer for a payments platform. " * 400


class StubProvider:
    """Stands in for a vendor, with a working model-scoped prompt cache."""

    name = "stub"

    def __init__(self):
        self.seen: set[str] = set()
        self.calls: list[tuple[str, str]] = []

    async def invoke(self, canonical, model_key, effort, cache_plan):
        self.calls.append((model_key, effort))
        prefix_tokens, volatile_tokens = estimate_request_tokens(canonical)
        key = prefix_fingerprint(canonical, model_key)

        read = write = 0
        if cache_plan.cacheable:
            # The await between the lookup and the write is the point of this
            # stub. A real cache entry is not readable until the first response
            # lands, so concurrent requests on the same prefix genuinely all
            # miss. Deciding read-vs-write atomically would hide exactly the
            # behaviour the pilot exists to fix.
            already_cached = key in self.seen
            await asyncio.sleep(0.02)
            if already_cached:
                read = prefix_tokens
            else:
                write = prefix_tokens
                self.seen.add(key)

        fresh = prefix_tokens - read - write + volatile_tokens
        return ProviderResponse(
            text=f"[stub answer from {model_key} at effort={effort}]",
            finish_reason="stop",
            model=model_key,
            usage=Usage(
                prompt_tokens=max(0, fresh) + read + write,
                completion_tokens=120,
                cache_read_tokens=read,
                cache_write_tokens=write,
            ),
        )

    async def stream(self, *a, **k):  # pragma: no cover - not exercised here
        yield {"delta": {"content": "stub"}, "finish_reason": None}

    async def classify(self, *a, **k):
        return None

    async def count_tokens(self, *a, **k):
        return None

    async def validate(self):
        # Mimics a vendor rejecting a malformed key without a network call.
        return (True, "authenticated (stub)") if self.key_ok else (False, "key rejected (401)")

    key_ok = True


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stub")
    monkeypatch.setenv("OPENAI_API_KEY", "stub")
    monkeypatch.setenv("GATEWAY_RECORD_PATH", str(tmp_path / "records.db"))
    monkeypatch.setenv("GATEWAY_LLM_CLASSIFIER_ENABLED", "false")

    from aigateway.config import get_settings

    get_settings.cache_clear()

    stub = StubProvider()

    def fake_init(self, settings):
        self._providers = {"anthropic": stub, "openai": stub}
        self._masked = {"anthropic": "sk-…stub", "openai": "sk-…stub"}
        self._source = {"anthropic": "env", "openai": "env"}

    monkeypatch.setattr(
        "aigateway.providers.registry.ProviderRegistry.__init__", fake_init
    )
    # Runtime credential updates should build the stub, not a real SDK client.
    monkeypatch.setitem(
        __import__("aigateway.providers.registry", fromlist=["BUILDERS"]).BUILDERS,
        "anthropic",
        lambda key, kind="api_key": stub,
    )

    app = create_app()
    with TestClient(app) as c:
        c.stub = stub
        yield c
    get_settings.cache_clear()


def _chat(client, prompt, session="s1", context=BIG_CONTEXT, **ext):
    messages = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": prompt})
    body = {"model": "auto", "messages": messages, "max_tokens": 512}
    body["x_gateway"] = {"session_id": session, **ext}
    res = client.post(
        "/v1/chat/completions",
        json=body,
        headers={"x-tenant-id": "t-e2e", "x-agent-id": "pytest"},
    )
    return res


def test_console_page_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "AI Gateway Console" in res.text
    # Self-contained: no CDN, no build step.
    assert "http://" not in res.text.replace("http://www.w3.org", "")


def test_end_to_end_returns_routing_metadata(client):
    res = _chat(client, "Classify the sentiment of this ticket.")
    assert res.status_code == 200, res.text

    data = res.json()
    meta = data["x_gateway"]
    assert data["choices"][0]["message"]["content"].startswith("[stub answer")
    assert meta["chosen_model"]
    assert meta["resolved_intent"]
    assert meta["routing_reason"]
    assert meta["considered"], "the alternatives should be visible, not just the winner"
    assert meta["actual_cost_usd"] > 0


def test_second_call_on_same_session_reads_the_cache(client):
    """The headline behaviour: cold write, then warm read on the same session."""
    first = _chat(client, "Review this diff for bugs.", session="warm").json()["x_gateway"]
    second = _chat(client, "Now check the error handling.", session="warm").json()["x_gateway"]

    assert first["cache_write_tokens"] > 0
    assert first["cache_read_tokens"] == 0

    assert second["cache_read_tokens"] > 0
    assert second["chosen_model"] == first["chosen_model"], "stickiness should hold the model"
    assert second["actual_cost_usd"] < first["actual_cost_usd"]


def test_reset_forces_a_cold_route(client):
    _chat(client, "Review this diff.", session="reset-me")
    res = client.post("/demo/reset/reset-me", headers={"x-tenant-id": "t-e2e"})
    assert res.status_code == 200
    assert res.json()["reset"] is True


def _stages(client, **kw):
    """Run one request through the SSE trace and return {stage: event}."""
    import json as _json

    body = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Review this diff."}],
        **kw,
    }
    res = client.post("/demo/trace", json=body, headers={"x-tenant-id": "t-e2e"})
    assert res.status_code == 200, res.text
    events = {}
    for frame in res.text.split("\n\n"):
        for line in frame.split("\n"):
            if line.startswith("data: ") and line[6:] != "[DONE]":
                ev = _json.loads(line[6:])
                events[ev["stage"]] = ev
    return events


def test_every_timed_stage_carries_a_baseline_verdict(client):
    """The console colours from these. A stage without one is drawn neutral,
    which silently hides exactly the slow step you were looking for."""
    events = _stages(client)

    for name in ("canonicalised", "classified", "routed", "served"):
        ev = events[name]
        assert "stage_ms" in ev, f"{name} has no per-stage duration"
        assert "baseline" in ev, f"{name} has no baseline verdict"
        assert ev["baseline"]["band"] in {
            "fast", "normal", "warn", "critical", "learning",
        }
        assert ev["baseline"]["note"]


def test_a_cold_run_starts_out_learning_rather_than_guessing(client):
    """With no history there is nothing to judge against, and inventing a colour
    would be worse than admitting it."""
    assert _stages(client)["classified"]["baseline"]["band"] == "learning"


def test_the_upstream_call_is_judged_on_its_own_latency(client):
    """Wall-clock between stages includes whatever the cache pilot spent
    waiting. Blaming the vendor for our own wait would be unfalsifiable."""
    served = _stages(client)["served"]
    assert served["baseline"]["measured_ms"] == served["latency_ms"]
    assert served["baseline"]["segment"].startswith("served:")
    assert served["model"] in served["baseline"]["segment"]


def test_baselines_become_confident_and_are_published(client):
    """Every colour has to be checkable against the numbers behind it."""
    from aigateway.observability.baselines import MIN_SAMPLES

    for _ in range(MIN_SAMPLES + 2):
        _stages(client)

    snap = client.get("/admin/baselines", headers={"x-tenant-id": "t-e2e"}).json()
    by_key = {s["key"]: s for s in snap["segments"]}

    assert snap["min_samples"] == MIN_SAMPLES
    classified = by_key["classified"]
    assert classified["confident"] is True
    assert classified["samples"] >= MIN_SAMPLES
    # The thresholds a band was measured against are published, not implied.
    if classified["stddev_ms"] > 0:
        assert classified["critical_above_ms"] > classified["warn_above_ms"]
    else:
        # A stub provider makes every local stage identically fast. There is no
        # sigma to multiply, so publishing a number would misrepresent the rule.
        assert classified["degenerate"] is True
        assert classified["warn_above_ms"] is None

    # And once confident, a stage stops reporting `learning`.
    assert _stages(client)["classified"]["baseline"]["band"] != "learning"


def test_upstream_segments_separate_cache_states(client):
    """A cold write and a warm read are not the same operation, and pooling
    them paints every cold request red."""
    for _ in range(3):
        _stages(client, x_gateway={"session_id": "base-warm"})

    snap = client.get("/admin/baselines", headers={"x-tenant-id": "t-e2e"}).json()
    served = [s["key"] for s in snap["segments"] if s["key"].startswith("served:")]
    states = {k.rsplit(":", 1)[-1] for k in served}
    assert states, "the upstream call must be segmented"
    assert all(":" in k for k in served)


def test_an_absent_budget_is_sized_to_the_intent(client):
    """Output budget dominates wall-clock — the same prompt measured 18s at
    1,200 tokens and 58s at 8,000, while every gateway stage together took
    1.3ms. One global default is either slow for classification or starving for
    review, and it cannot be both right."""
    from aigateway.routing.policy import policy_for

    events = _stages(client)  # no max_tokens sent
    budgeted = events.get("budgeted")

    assert budgeted, "an absent budget must be sized, not defaulted globally"
    assert budgeted["max_tokens"] == policy_for(events["classified"]["intent"]).max_tokens
    assert "max_tokens" in budgeted["note"], "and it must say what it did"


def test_a_caller_supplied_budget_always_wins(client):
    """Their budget is their decision. The gateway sizes an absent one; it does
    not overrule one that was given."""
    events = _stages(client, max_tokens=333)
    assert "budgeted" not in events


def test_light_work_gets_a_smaller_budget_than_heavy_work(client):
    from aigateway.routing.policy import policy_for

    assert policy_for("classify").max_tokens < policy_for("code_review").max_tokens
    assert policy_for("code_review").max_tokens < policy_for("architecture").max_tokens


def test_every_intent_leaves_room_for_an_answer(client):
    """This test used to assert `>= floor` and passed while shipping a default
    that could not work: the floor is what *reasoning* spends, so a budget equal
    to it leaves exactly nothing to reply with. code_review shipped at 8,000
    against a `high` floor of 8,000 and returned empty on a large query, with
    every check reporting the budget as adequate. The gap is the assertion."""
    from aigateway.quality import budget_for_effort, effort_that_fits
    from aigateway.routing.policy import INTENT_POLICY

    for policy in INTENT_POLICY.values():
        if policy.min_tier.name == "LIGHT":
            # Deliberate exception: the floors were measured on demanding
            # prompts, and a classify genuinely finishes in ~150 tokens.
            continue
        need = budget_for_effort(policy.effort)
        assert policy.max_tokens >= need, (
            f"{policy.intent}: budget {policy.max_tokens} leaves no room for an "
            f"answer at effort '{policy.effort}' (needs {need})"
        )
        # And the budget must actually sustain the effort the policy asked for,
        # or the policy is quietly asking for depth it cannot pay for.
        assert effort_that_fits(policy.effort, policy.max_tokens) == policy.effort, (
            f"{policy.intent}: budget {policy.max_tokens} silently downgrades "
            f"effort '{policy.effort}'"
        )


def test_health_publishes_the_switches(client):
    switches = client.get("/health").json()["switches"]
    assert switches["auto_size_max_tokens"] is True
    assert switches["quality_judge"] is False, "a billable extra call stays opt-in"
    for name in ("latency_baselines", "hop_trace", "quality_checks", "effort_tracking"):
        assert name in switches


def test_fanout_pilot_means_one_writer(client):
    """Eight sub-agents, one shared prefix: one write, the rest read."""
    res = client.post(
        "/demo/fanout",
        json={
            "prompt": "Audit one module.",
            "shared_context": BIG_CONTEXT,
            "agents": 8,
            "session_id": "fan-on",
            "pilot_enabled": True,
        },
        headers={"x-tenant-id": "t-e2e", "x-agent-id": "fanout"},
    )
    assert res.status_code == 200, res.text
    data = res.json()

    pilots = [a for a in data["agents"] if a.get("pilot_role") == "pilot"]
    readers = [a for a in data["agents"] if a.get("cache_read_tokens", 0) > 0]

    assert len(pilots) == 1, "exactly one agent should write the cache"
    assert len(readers) >= 6, f"the rest should read it, got {len(readers)}"


def test_fanout_returns_each_agents_answer_and_the_task_it_answered(client):
    """The panel was a table of metrics with no output in it: you could see what
    each agent cost but not whether it produced anything worth the money. An
    answer is also unreviewable without the question it answered."""
    res = client.post(
        "/demo/fanout",
        json={
            "prompt": "Audit one module.",
            "shared_context": BIG_CONTEXT,
            "agents": 3,
            "session_id": "fan-answers",
        },
        headers={"x-tenant-id": "t-e2e"},
    )
    assert res.status_code == 200, res.text

    for agent in res.json()["agents"]:
        assert agent["task"], "every agent reports what it was asked"
        assert agent["answer"], "and what it answered"
        assert agent["quality"] in ("pass", "warn", "fail")


def test_fanout_gives_each_agent_its_own_task_when_asked(client):
    """Sending all N the identical prompt produces N near-identical answers,
    which makes per-agent output impossible to judge. The shared prefix is what
    the cache needs to be identical — the tasks are not."""
    subtasks = ["Check the retries.", "Check the ledger.", "Check the tests."]
    res = client.post(
        "/demo/fanout",
        json={
            "prompt": "ignored when subtasks are given",
            "shared_context": BIG_CONTEXT,
            "agents": 3,
            "subtasks": subtasks,
            "session_id": "fan-subtasks",
        },
        headers={"x-tenant-id": "t-e2e"},
    )
    assert res.status_code == 200, res.text
    data = res.json()

    assert [a["task"] for a in data["agents"]] == subtasks
    # Distinct tasks must not cost the shared prefix its cacheability.
    assert len([a for a in data["agents"] if a.get("pilot_role") == "pilot"]) == 1


def test_fanout_cycles_subtasks_when_there_are_more_agents_than_tasks(client):
    res = client.post(
        "/demo/fanout",
        json={
            "prompt": "unused",
            "shared_context": BIG_CONTEXT,
            "agents": 4,
            "subtasks": ["A", "B"],
            "session_id": "fan-cycle",
        },
        headers={"x-tenant-id": "t-e2e"},
    )
    assert [a["task"] for a in res.json()["agents"]] == ["A", "B", "A", "B"]


def test_fanout_without_the_pilot_pays_repeatedly(client):
    """Turning the pilot off is the status quo every other gateway ships."""
    with_pilot = client.post(
        "/demo/fanout",
        json={
            "prompt": "Audit one module.",
            "shared_context": BIG_CONTEXT,
            "agents": 6,
            "session_id": "cmp-on",
            "pilot_enabled": True,
        },
        headers={"x-tenant-id": "t-e2e"},
    ).json()

    client.stub.seen.clear()  # fresh provider-side cache for a fair comparison

    without = client.post(
        "/demo/fanout",
        json={
            "prompt": "Audit one module.",
            "shared_context": BIG_CONTEXT,
            "agents": 6,
            "session_id": "cmp-off",
            "pilot_enabled": False,
        },
        headers={"x-tenant-id": "t-e2e"},
    ).json()

    assert without["total_cache_write_tokens"] > with_pilot["total_cache_write_tokens"]
    assert without["total_cost_usd"] > with_pilot["total_cost_usd"]


def test_declared_intent_is_honoured(client):
    meta = _chat(client, "anything", session="i1", intent="architecture").json()["x_gateway"]
    assert meta["resolved_intent"] == "architecture"
    assert meta["intent_source"] == "declared"
    assert meta["tier"] == "heavy"


def test_max_tier_caps_the_router(client):
    meta = _chat(
        client, "Design the sharding strategy.", session="i2",
        intent="architecture", max_tier="light",
    ).json()["x_gateway"]
    assert meta["tier"] == "light"
    assert "capped by caller" in meta["routing_reason"]


def test_usage_is_recorded_against_the_tenant(client):
    _chat(client, "Summarise this.", session="ledger")
    usage = client.get("/admin/usage/t-e2e").json()
    assert usage["spend_usd_today"] > 0
    assert usage["limit_usd_daily"] > 0


def _trace_events(client, prompt, session="tr1"):
    res = client.post(
        "/demo/trace",
        json={
            "model": "auto",
            "messages": [
                {"role": "system", "content": BIG_CONTEXT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 256,
            "x_gateway": {"session_id": session},
        },
        headers={"x-tenant-id": "t-e2e", "x-agent-id": "console"},
    )
    assert res.status_code == 200, res.text
    events = []
    for line in res.text.splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            events.append(json.loads(line[6:]))
    return events


def test_trace_streams_every_pipeline_stage_in_order(client):
    """The console renders from these events, so the sequence is a contract."""
    events = _trace_events(client, "Review this diff for bugs.")
    stages = [e["stage"] for e in events]

    required = (
        "accepted", "canonicalised", "classified", "routed", "cache", "served", "done",
    )
    for expected in required:
        assert expected in stages, f"missing stage {expected}: {stages}"

    assert stages.index("classified") < stages.index("routed")
    assert stages.index("routed") < stages.index("served")
    assert stages[-1] == "done"


def test_trace_stages_carry_real_timings(client):
    events = _trace_events(client, "Summarise this.", session="tr2")
    elapsed = [e["elapsed_ms"] for e in events if "elapsed_ms" in e]
    assert elapsed == sorted(elapsed), "stage timings must be monotonic"


def test_trace_routed_stage_lists_candidates(client):
    events = _trace_events(client, "Design the sharding strategy.", session="tr3")
    routed = next(e for e in events if e["stage"] == "routed")

    assert routed["considered"], "the alternatives must reach the visual"
    assert sum(1 for c in routed["considered"] if c["chosen"]) == 1


def test_trace_reports_errors_as_a_stage(client):
    """A failure must arrive as an error event, not a dead stream."""
    res = client.post(
        "/demo/trace",
        json={"model": "auto", "messages": [{"role": "user", "content": "hi"}],
              "x_gateway": {"pin_model": "no-such-model"}},
        headers={"x-tenant-id": "t-e2e"},
    )
    assert res.status_code == 200
    stages = [
        json.loads(line[6:])
        for line in res.text.splitlines()
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]
    assert any(e["stage"] == "error" for e in stages)


def test_pool_endpoint_reports_every_model(client):
    d = client.get("/admin/pool").json()
    assert len(d["models"]) == len(__import__("aigateway.catalog", fromlist=["CATALOG"]).CATALOG)
    assert {"model", "status", "breaker", "p50_latency_ms"} <= set(d["models"][0])
    assert d["breaker_failure_threshold"] >= 1


def test_pool_records_latency_from_real_traffic(client):
    _chat(client, "Classify this.", session="pool-1")
    d = client.get("/admin/pool").json()
    exercised = [m for m in d["models"] if m["total_ok"] > 0]
    assert exercised, "a served request should show up in the pool"
    assert exercised[0]["p50_latency_ms"] is not None
    assert exercised[0]["status"] == "healthy"


SECRET = "sk-ant-secret-value-do-not-leak-1234"


def test_credential_status_never_returns_a_key(client):
    d = client.get("/admin/credentials").json()
    assert d["editable"] is True
    body = str(d)
    assert "sk-ant-secret" not in body
    for p in d["providers"]:
        assert "api_key" not in p
        assert p["masked_key"].count("…") <= 1


def test_setting_a_valid_key_configures_the_provider(client):
    client.stub.key_ok = True
    res = client.post(
        "/admin/credentials",
        json={"provider": "anthropic", "api_key": SECRET, "persist": False},
    )
    assert res.status_code == 200, res.text
    d = res.json()

    assert d["configured"] is True
    assert d["persisted_to_env"] is False
    # The response must carry a mask, never the key.
    assert SECRET not in str(d)
    assert d["masked_key"].startswith("sk-") and d["masked_key"].endswith("1234")


def test_a_rejected_key_is_not_kept(client):
    """A bad key must not silently replace a working one."""
    client.stub.key_ok = False
    res = client.post(
        "/admin/credentials",
        json={"provider": "anthropic", "api_key": "sk-bogus-key-value", "persist": False},
    )
    assert res.status_code == 400
    assert "401" in res.json()["error"]["message"]

    status = client.get("/admin/credentials").json()
    anthropic = next(p for p in status["providers"] if p["provider"] == "anthropic")
    assert anthropic["configured"] is False, "a rejected key must be discarded"

    client.stub.key_ok = True


def test_unknown_provider_is_rejected(client):
    res = client.post(
        "/admin/credentials", json={"provider": "notavendor", "api_key": "sk-whatever"}
    )
    assert res.status_code == 422


def test_removing_a_key_deconfigures_the_provider(client):
    assert client.delete("/admin/credentials/openai").status_code == 200
    status = client.get("/admin/credentials").json()
    openai = next(p for p in status["providers"] if p["provider"] == "openai")
    assert openai["configured"] is False


def test_preview_works_before_any_credentials_exist(client):
    """The routing logic must be inspectable before a key is entered — but the
    response has to say plainly that nothing can actually serve it."""
    client.delete("/admin/credentials/anthropic")
    client.delete("/admin/credentials/openai")

    res = client.post(
        "/admin/route/preview",
        json={"model": "auto", "messages": [{"role": "user", "content": "Review this diff."}]},
    )
    assert res.status_code == 200, res.text
    d = res.json()
    assert d["decision"]["model"]
    assert d["servable_now"] is False
    assert all(c["available"] is False for c in d["considered"])


def test_preview_can_restrict_to_servable_models(client):
    client.delete("/admin/credentials/anthropic")
    client.delete("/admin/credentials/openai")
    res = client.post(
        "/admin/route/preview?include_unavailable=false",
        json={"model": "auto", "messages": [{"role": "user", "content": "Review this diff."}]},
    )
    # 503, not 422: nothing is wrong with the request — we have nothing to
    # serve it with. The status code is how a client knows retrying is sane.
    assert res.status_code == 503
    body = res.json()["error"]
    assert "provider not configured" in body["message"]
    assert body["cause"] == "no_credentials"
    assert body["remedy"]
    assert res.headers["retry-after"]


def test_route_preview_spends_nothing(client):
    before = len(client.stub.calls)
    res = client.post(
        "/admin/route/preview",
        json={"model": "auto", "messages": [{"role": "user", "content": "Classify this."}]},
    )
    assert res.status_code == 200
    assert res.json()["decision"]["model"]
    assert len(client.stub.calls) == before, "preview must not call a provider"


def test_trace_records_every_hop_including_origination(client):
    events = _trace_events(client, "Review this diff.", session="hops-1")
    hops_ev = next((e for e in events if e["stage"] == "hops"), None)
    assert hops_ev, "the hop trace must reach the console"

    hops = hops_ev["hops"]
    assert hops[0]["kind"] == "origin", "hop 0 is who asked; without it the trace has no subject"
    assert "tenant=t-e2e" in hops[0]["detail"]

    model_hops = [h for h in hops if h["kind"] == "model"]
    assert model_hops, "the model call must appear as a hop"
    assert model_hops[0]["host"], "a hop has to name the server it contacted"
    assert model_hops[0]["tokens_in"] > 0


def test_trace_summary_separates_gateway_overhead_from_upstream(client):
    events = _trace_events(client, "Summarise this.", session="hops-2")
    t = next(e for e in events if e["stage"] == "hops")

    assert t["total_ms"] >= t["upstream_ms"]
    assert t["gateway_overhead_ms"] == max(0, t["total_ms"] - t["upstream_ms"])
    assert t["hosts_contacted"], "at least one upstream host should be named"
    assert t["failed_hops"] == 0


def test_hop_trace_is_returned_on_the_normal_api_too(client):
    """Not a demo-only feature — an agent can read its own trace."""
    meta = _chat(client, "Classify this.", session="hops-3").json()["x_gateway"]
    assert meta["trace"]["hop_count"] >= 2
    assert meta["trace"]["trace_id"] == meta["trace_id"]


def test_fleet_aggregates_across_transactions(client):
    """A hop trace covers one request; the fleet view covers all of them."""
    _chat(client, "Classify this ticket.", session="fleet-a")
    _chat(client, "Review this diff for bugs.", session="fleet-b")
    _chat(client, "Summarise this.", session="fleet-c")

    f = client.get("/admin/fleet").json()
    assert f["total_requests"] >= 3
    assert f["by_host"], "traffic has to be attributed to a server"
    assert f["by_model"] and f["by_intent"] and f["by_tenant"]
    assert f["total_cost_usd"] > 0

    top = f["by_host"][0]
    assert top["requests"] >= 1
    assert 0 < top["share"] <= 1.0
    assert top["p50_ms"] is not None


def test_fleet_flows_show_distinct_paths(client):
    """The flow rows are the story: who sends what, and where it lands."""
    _chat(client, "Classify this ticket.", session="flow-1", intent="classify")
    _chat(client, "Design the sharding strategy.", session="flow-2", intent="architecture")

    flows = client.get("/admin/fleet").json()["flows"]
    intents = {f["intent"] for f in flows}
    assert {"classify", "architecture"} <= intents

    for row in flows:
        assert {"tenant", "intent", "tier", "model", "host", "requests"} <= set(row)
    # Distinct intents must not collapse into one path.
    assert len({(f["intent"], f["model"]) for f in flows}) >= 2


def test_fleet_counts_failures_not_just_successes(client):
    """An availability view that only sees successes is worse than none."""
    before = client.get("/admin/fleet").json()["total_requests"]
    client.post(
        "/v1/chat/completions",
        json={"model": "no-such-model", "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-tenant-id": "t-e2e"},
    )
    after = client.get("/admin/fleet").json()
    assert after["total_requests"] == before + 1


def test_console_is_served_uncacheable(client):
    """A cached console silently hides new panels, which looks like a bug."""
    res = client.get("/")
    assert "no-store" in res.headers.get("cache-control", "")


def test_fanin_routes_each_leg_independently(client):
    """Scatter/gather is where a router earns its keep — narrow worker jobs and
    a synthesis step that must hold all of their output should not land on the
    same model just because they arrived together."""
    res = client.post(
        "/demo/fanin",
        json={
            "task": "Should we adopt this?",
            "subtasks": [
                "Classify the risk as low, medium or high",
                "Review this design for race conditions",
            ],
            "shared_context": BIG_CONTEXT,
            "session_id": "fi-test",
            "max_tokens": 400,
        },
        headers={"x-tenant-id": "t-e2e", "x-agent-id": "fanin"},
    )
    assert res.status_code == 200, res.text
    d = res.json()

    assert len(d["workers"]) == 2
    assert d["synthesis"]["model"]
    # Each leg is classified on its own merits, not inherited from the parent.
    intents = {w["intent"] for w in d["workers"] if "intent" in w}
    assert len(intents) > 1, f"subtasks of different difficulty collapsed: {intents}"

    t = d["totals"]
    assert t["total_cost_usd"] == pytest.approx(
        t["worker_cost_usd"] + t["synthesis_cost_usd"], rel=1e-6
    )
    assert t["scatter_ms"] >= 0 and t["gather_ms"] >= 0


def test_fanin_scatter_really_is_parallel(client):
    """Wall clock should track the slowest worker, not their sum."""
    res = client.post(
        "/demo/fanin",
        json={
            "task": "Summarise",
            "subtasks": ["Classify this", "Summarize this", "Translate this"],
            "shared_context": BIG_CONTEXT,
            "session_id": "fi-par",
            "max_tokens": 300,
        },
        headers={"x-tenant-id": "t-e2e"},
    ).json()

    workers = [w for w in res["workers"] if "latency_ms" in w]
    assert len(workers) == 3
    serial = sum(w["latency_ms"] for w in workers)
    assert res["totals"]["scatter_ms"] < serial, "workers did not run concurrently"


def test_fanin_fails_loudly_if_every_worker_fails(client):
    """Synthesising over nothing would produce a confident, empty answer."""
    res = client.post(
        "/demo/fanin",
        json={"task": "x", "subtasks": [], "session_id": "fi-empty"},
        headers={"x-tenant-id": "t-e2e"},
    )
    assert res.status_code == 502
    assert "nothing to synthesise" in res.json()["error"]["message"]


def test_console_exposes_all_four_tabs(client):
    page = client.get("/").text
    for tab in ("route", "agents", "fleet", "models"):
        assert f'data-tab="{tab}"' in page
    assert "Fan-in" in page and "Fan-out" in page


# -- analytics over the record database --------------------------------------
def test_analytics_reflects_served_traffic(client):
    """The durable analytics view must agree with what was just served."""
    assert _chat(client, "Classify the sentiment of this ticket.").status_code == 200
    assert _chat(client, "Classify this one too.").status_code == 200

    res = client.get("/admin/analytics?hours=1")
    assert res.status_code == 200
    data = res.json()
    assert data["overview"]["requests"] == 2
    assert data["overview"]["spend_usd"] > 0
    assert data["by_model"], "served traffic must appear in the per-model rollup"
    assert data["by_model"][0]["requests"] >= 1
    assert data["timeseries"], "hourly buckets must cover the traffic just sent"


def test_analytics_dashboard_page_is_served(client):
    page = client.get("/analytics")
    assert page.status_code == 200
    assert "Gateway analytics" in page.text
    # Same rule as the console: self-contained, no CDN.
    assert "http://" not in page.text.replace("http://www.w3.org", "")


# -- streaming parity ---------------------------------------------------------
def _stream_chat(client, prompt, session="stream-1", **ext):
    messages = [
        {"role": "system", "content": BIG_CONTEXT},
        {"role": "user", "content": prompt},
    ]
    body = {
        "model": "auto",
        "messages": messages,
        "max_tokens": 512,
        "stream": True,
        "x_gateway": {"session_id": session, **ext},
    }
    res = client.post(
        "/v1/chat/completions",
        json=body,
        headers={"x-tenant-id": "t-e2e", "x-agent-id": "pytest"},
    )
    assert res.status_code == 200, res.text
    frames = [
        json.loads(line[6:])
        for line in res.text.split("\n")
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]
    assert res.text.rstrip().endswith("data: [DONE]"), "stream must close properly"
    return frames


def test_streaming_announces_its_routing_in_the_first_chunk(client):
    frames = _stream_chat(client, "Review this diff for bugs.")
    meta = frames[0]["x_gateway"]
    assert meta["chosen_model"]
    assert meta["resolved_intent"]
    assert meta["pilot_role"], "streaming must go through the cache pilot too"
    assert meta["trace_id"]


def test_streaming_requests_are_recorded_and_billed(client):
    """The old streaming path was invisible: no record, and no ledger write
    when usage never arrived. Both are load-bearing."""
    _stream_chat(client, "Review this diff for race conditions.", session="stream-bill")

    data = client.get("/admin/analytics?hours=1").json()
    assert data["overview"]["requests"] == 1
    assert data["overview"]["spend_usd"] > 0, "served tokens must reach the ledger"

    spend = client.get(
        "/admin/usage/t-e2e", headers={"x-tenant-id": "t-e2e"}
    ).json()
    assert spend["spend_usd_today"] > 0


def test_streaming_keeps_the_session_sticky_for_unary_followups(client):
    """One pipeline: a streamed turn must warm the same session state the
    unary path reads."""
    frames = _stream_chat(client, "Review this diff.", session="stream-warm")
    streamed_model = frames[0]["x_gateway"]["chosen_model"]

    follow = _chat(client, "Now check error handling.", session="stream-warm").json()
    assert follow["x_gateway"]["chosen_model"] == streamed_model


# -- live fan-out / fan-in ----------------------------------------------------
def _live_events(client, path, body):
    res = client.post(path, json=body, headers={"x-tenant-id": "t-e2e"})
    assert res.status_code == 200, res.text
    events = [
        json.loads(line[6:])
        for line in res.text.split("\n")
        if line.startswith("data: ") and line[6:] != "[DONE]"
    ]
    assert res.text.rstrip().endswith("data: [DONE]")
    return events


def test_fanout_live_streams_every_workers_stages(client):
    events = _live_events(client, "/demo/fanout/live", {
        "prompt": "Review this.", "shared_context": BIG_CONTEXT,
        "agents": 3, "session_id": "live-fo", "max_tokens": 512,
    })
    stages = [e for e in events if e["event"] == "stage"]
    workers = {e["worker"] for e in stages}
    assert workers == {1, 2, 3}, "every worker's pipeline must be visible"
    for w in workers:
        seen = [e["stage"] for e in stages if e["worker"] == w]
        assert "routed" in seen and "served" in seen, f"worker {w} lane is incomplete"
    # The unified feed's clock: run-relative and non-decreasing per worker.
    for w in workers:
        ats = [e["at_ms"] for e in stages if e["worker"] == w]
        assert ats == sorted(ats)

    done = [e for e in events if e["event"] == "worker_done"]
    assert len(done) == 3
    summary = [e for e in events if e["event"] == "summary"]
    assert len(summary) == 1
    assert len(summary[0]["agents"]) == 3
    assert summary[0]["total_cost_usd"] > 0

    # The gateway-vs-LLM split: each lane closes with a hops event carrying
    # it, and the summary accumulates it — the number that recurs on every
    # agent call, which is the whole case for keeping the gateway light.
    hops = [e for e in stages if e["stage"] == "hops"]
    assert len(hops) == 3
    for h in hops:
        assert "gateway_overhead_ms" in h and "upstream_ms" in h
        assert "pilot_wait_ms" in h

    split = summary[0]["time_split"]
    assert split["calls"] == 3
    assert split["llm_ms_total"] > 0
    assert split["gateway_ms_total"] >= 0
    for agent in summary[0]["agents"]:
        assert "gateway_ms" in agent and "upstream_ms" in agent


def test_fanin_live_streams_workers_then_synthesis(client):
    events = _live_events(client, "/demo/fanin/live", {
        "task": "Decide.", "subtasks": ["Classify the risk", "Summarize the cost"],
        "shared_context": BIG_CONTEXT, "session_id": "live-fi", "max_tokens": 512,
    })
    stages = [e for e in events if e["event"] == "stage"]
    assert {e["worker"] for e in stages} == {1, 2, "synth"}

    # The synthesiser must not start until every worker has finished.
    first_synth = next(i for i, e in enumerate(stages) if e["worker"] == "synth")
    served = [i for i, e in enumerate(stages)
              if e["worker"] != "synth" and e["stage"] == "served"]
    assert len(served) == 2 and max(served) < first_synth

    summary = [e for e in events if e["event"] == "summary"]
    assert len(summary) == 1
    assert summary[0]["synthesis"]["model"]
    assert summary[0]["totals"]["workers"] == 2
    # Cumulative gateway time covers every leg: both workers AND the synth.
    assert summary[0]["totals"]["time_split"]["calls"] == 3
    assert summary[0]["synthesis"]["upstream_ms"] >= 0


# -- account / subscription credentials ---------------------------------------
def test_a_subscription_token_is_detected_and_accepted(client):
    """`claude setup-token` tokens carry the sk-ant-oat prefix; the gateway
    should recognise the shape without the user knowing the taxonomy."""
    client.stub.key_ok = True
    res = client.post(
        "/admin/credentials",
        json={"provider": "anthropic", "api_key": "sk-ant-oat01-subscription-token-xyz"},
    )
    assert res.status_code == 200, res.text
    d = res.json()
    assert d["kind"] == "oauth_token"
    assert "sk-ant-oat01-subscription-token-xyz" not in str(d)

    status = client.get("/admin/credentials").json()
    anthropic = next(p for p in status["providers"] if p["provider"] == "anthropic")
    assert anthropic["kind"] == "oauth_token"
    assert "oauth_token" in anthropic["accepts"]


def test_openai_refuses_subscription_tokens_with_a_reason(client):
    """A ChatGPT plan is not API access; failing clearly beats failing 401."""
    res = client.post(
        "/admin/credentials",
        json={"provider": "openai", "api_key": "some-subscription-token",
              "kind": "oauth_token"},
    )
    assert res.status_code == 422
    msg = res.json()["error"]["message"]
    assert "subscription" in msg and "platform.openai.com" in msg

    openai = next(
        p for p in client.get("/admin/credentials").json()["providers"]
        if p["provider"] == "openai"
    )
    assert openai["accepts"] == ["api_key"]


# -- live pricing -------------------------------------------------------------
def test_price_refresh_applies_the_feed_and_reroutes(client, monkeypatch):
    """The button's whole promise: pull, apply with provenance, and the
    router selects on the new rates immediately."""
    from aigateway.catalog import CATALOG, get_model
    from aigateway.prices import PriceFeed

    snapshot = dict(CATALOG)
    try:
        before = _chat(client, "Classify this ticket.", session="price-a").json()
        cheap = get_model(before["x_gateway"]["chosen_model"])

        fake_feed = {
            cheap.vendor_model_id: {
                "input_cost_per_token": cheap.price_in_per_mtok * 5 / 1e6,
                "output_cost_per_token": cheap.price_out_per_mtok * 5 / 1e6,
            }
        }

        async def fake_fetch(self):
            return fake_feed

        monkeypatch.setattr(PriceFeed, "fetch", fake_fetch)
        res = client.post("/admin/prices/refresh")
        assert res.status_code == 200, res.text
        report = res.json()
        assert [u["model"] for u in report["updated"]] == [cheap.key]

        status = client.get("/admin/prices").json()
        row = next(m for m in status["models"] if m["model"] == cheap.key)
        assert row["price_in_per_mtok"] == pytest.approx(cheap.price_in_per_mtok * 5)
        assert row["checked"]

        after = _chat(client, "Classify this one too.", session="price-b").json()
        assert after["x_gateway"]["chosen_model"] != cheap.key
    finally:
        CATALOG.clear()
        CATALOG.update(snapshot)


def test_unreachable_price_feed_fails_loudly_and_changes_nothing(client, monkeypatch):
    from aigateway.catalog import CATALOG
    from aigateway.prices import PriceFeed

    before = {k: m.price_in_per_mtok for k, m in CATALOG.items()}

    async def broken_fetch(self):
        raise RuntimeError("feed unreachable")

    monkeypatch.setattr(PriceFeed, "fetch", broken_fetch)
    res = client.post("/admin/prices/refresh")
    assert res.status_code == 502
    assert "feed" in res.json()["error"]["message"]
    assert {k: m.price_in_per_mtok for k, m in CATALOG.items()} == before


# -- transaction log ----------------------------------------------------------
def test_transaction_log_carries_the_full_decision_record(client):
    _chat(client, "Review this diff for races.", session="txn-1")
    _chat(client, "Classify this ticket.", session="txn-2")

    res = client.get("/admin/transactions?hours=1")
    assert res.status_code == 200
    d = res.json()
    assert len(d["rows"]) == 2
    newest = d["rows"][0]
    # The drill-down promise: not just what happened, but why.
    assert newest["routing_reason"]
    assert isinstance(newest["considered"], list) and newest["considered"]
    assert newest["chosen_model"] and newest["outcome"] == "ok"
    assert newest["actual_cost_usd"] > 0
    assert d["facets"]["models"], "filter dropdowns need facet values"

    only = client.get(
        "/admin/transactions?hours=1&session=txn-1"
    ).json()["rows"]
    assert len(only) == 1 and only[0]["session_id"] == "txn-1"


def test_transactions_page_is_served(client):
    page = client.get("/transactions")
    assert page.status_code == 200
    assert "Transactions" in page.text
    assert "http://" not in page.text.replace("http://www.w3.org", "")


# -- database explorer --------------------------------------------------------
def test_db_explorer_endpoint_and_page(client):
    _chat(client, "Classify this.", session="db-1")

    schema = client.get("/admin/db/schema").json()
    assert schema["tables"]["records"]["rows"] == 1

    q = client.post(
        "/admin/db/query",
        json={"sql": "SELECT chosen_model, actual_cost_usd FROM records"},
    ).json()
    assert q["row_count"] == 1
    assert q["columns"] == ["chosen_model", "actual_cost_usd"]

    refused = client.post(
        "/admin/db/query", json={"sql": "DROP TABLE records"}
    ).json()
    assert "read-only" in refused["error"]

    page = client.get("/db")
    assert page.status_code == 200
    assert "Database" in page.text
    assert "http://" not in page.text.replace("http://www.w3.org", "")
