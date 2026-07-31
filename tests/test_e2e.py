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
    monkeypatch.setenv("GATEWAY_RECORD_PATH", str(tmp_path / "records.jsonl"))
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
        lambda key: stub,
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
    assert len(d["models"]) == 6
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
    assert res.status_code == 422
    assert "provider not configured" in res.json()["error"]["message"]


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
