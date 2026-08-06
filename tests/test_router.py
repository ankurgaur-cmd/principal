"""Router behaviour: the cache/cost trade-off is the thing under test."""

from __future__ import annotations

import pytest
from conftest import make_request

from aigateway.catalog import Tier, get_model
from aigateway.errors import NoCapableModel, NoModelsAvailable


async def test_light_intent_picks_a_light_model(router):
    decision = await router.route(make_request(user_text="label this review"), "classify")
    assert decision.model.tier == Tier.LIGHT


async def test_heavy_intent_never_drops_below_its_floor(router):
    decision = await router.route(make_request(), "architecture")
    assert decision.tier >= Tier.HEAVY, decision.reason


async def test_session_stickiness_keeps_the_warm_model(router, store):
    req = make_request(session_id="s1", system_tokens=8000)

    first = await router.route(req, "code_write")
    await router.remember("s1", first.model)

    # A cheaper intent arrives on the same session. Escalation-only means we
    # stay on the warm model rather than throwing away a cached prefix.
    second = await router.route(make_request(session_id="s1", system_tokens=8000), "summarize")
    assert second.model.key == first.model.key
    assert "sticky" in second.reason


async def test_stickiness_allows_escalation(router):
    await router.remember("s2", get_model("claude-haiku-4-5"))
    decision = await router.route(
        make_request(session_id="s2", system_tokens=8000), "architecture"
    )
    assert decision.tier >= Tier.HEAVY
    assert decision.model.key != "claude-haiku-4-5"


async def test_warm_model_is_scored_cheaper_than_an_identical_cold_one(router):
    """The cache transition has to show up in the cost, not just the reason."""
    req = make_request(session_id="s3", system_tokens=20_000)
    cold = await router.route(req, "code_write")

    await router.remember("s3", cold.model)
    warm = await router.route(req, "code_write")

    assert warm.model.key == cold.model.key
    assert warm.cache_state == "warm_read"
    assert warm.estimated_cost_usd < cold.estimated_cost_usd


async def test_budget_ceiling_degrades_the_tier(router):
    normal = await router.route(make_request(), "architecture")
    degraded = await router.route(make_request(), "architecture", cost_ceiling_tier=Tier.LIGHT)

    assert degraded.tier < normal.tier
    assert degraded.degraded is True
    assert "DEGRADED" in degraded.reason


async def test_context_window_excludes_undersized_models(router):
    # ~600k tokens of prefix: only the million-token models can serve it.
    decision = await router.route(make_request(system_tokens=600_000), "summarize")
    assert decision.model.context_window >= 1_000_000


async def test_pin_bypasses_routing_but_is_recorded(router):
    req = make_request(pin_model="claude-opus-5")
    decision = await router.route(req, "classify")
    assert decision.model.key == "claude-opus-5"
    assert "pinned" in decision.reason


async def test_unknown_pin_is_rejected(router):
    with pytest.raises(NoCapableModel):
        await router.route(make_request(pin_model="not-a-model"), "classify")


async def test_nothing_available_is_a_503_not_a_422(settings, store):
    """Different problems, different status codes. A 422 says "your request is
    wrong" and a client retrying one is wasting its time; a 503 says "we cannot
    serve right now" and retrying is exactly right. An empty candidate set is
    always the second kind."""
    from aigateway.routing import Router

    isolated = Router(settings, store, set())
    with pytest.raises(NoModelsAvailable) as exc:
        await isolated.route(make_request(), "classify")

    assert exc.value.status_code == 503
    assert exc.value.cause == "no_credentials"
    assert exc.value.remedy, "an alert nobody can act on is just noise"
    assert exc.value.headers["retry-after"]


async def test_a_pinned_request_still_notices_a_warm_session(router):
    """The pin path hardcoded is_warm=False, so a pinned model reported
    `cold_write` on every turn however many times the session had used it —
    wrong cache state, a cost estimate inflated by a write premium already paid,
    and no evidence at all reaching the cache-effectiveness learner, which only
    counts requests where a hit was expected."""
    req = make_request(session_id="pin-warm", system_tokens=20_000, pin_model="claude-opus-5")

    first = await router.route(req, "code_write")
    assert first.cache_state == "cold_write"

    await router.remember("pin-warm", first.model)
    second = await router.route(req, "code_write")

    assert second.pinned is True
    assert second.cache_state == "warm_read"
    assert second.estimated_cost_usd < first.estimated_cost_usd
