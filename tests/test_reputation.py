"""Quality-adjusted routing.

Feedback loops from quality into routing fail in characteristic ways — they
overreact to noise, they punish models permanently, and they average away the
per-task detail that made the signal useful. Most of these tests are about those
failure modes rather than the happy path.
"""

from __future__ import annotations

import random

import pytest
from conftest import make_request

from aigateway.routing import Reputation, Router, explain


@pytest.fixture
def rep() -> Reputation:
    # Seeded RNG so exploration is deterministic under test.
    return Reputation(min_samples=5, max_penalty=4.0, exploration_rate=0.0,
                      rng=random.Random(0))


def _feed(rep, model, intent, ok_count, fail_count):
    for _ in range(ok_count):
        rep.record(model, intent, True)
    for _ in range(fail_count):
        rep.record(model, intent, False)


# -- the signal itself -----------------------------------------------------
def test_no_evidence_means_no_adjustment(rep):
    """Penalising on one bad response would make routing depend on the order
    requests happened to arrive in."""
    assert rep.multiplier("gpt-5-nano", "classify") == 1.0
    rep.record("gpt-5-nano", "classify", False)
    assert rep.multiplier("gpt-5-nano", "classify") == 1.0, "one sample proves nothing"
    assert rep.success_rate("gpt-5-nano", "classify") is None


def test_penalty_is_expected_cost_not_an_arbitrary_weight(rep):
    """Succeeding half the time means you need ~2 attempts, so it costs ~2x."""
    _feed(rep, "gpt-5-nano", "code_review", ok_count=5, fail_count=5)
    assert rep.success_rate("gpt-5-nano", "code_review") == 0.5
    assert rep.multiplier("gpt-5-nano", "code_review") == pytest.approx(2.0)


def test_a_clean_record_earns_no_penalty(rep):
    _feed(rep, "gpt-5-nano", "classify", ok_count=10, fail_count=0)
    assert rep.multiplier("gpt-5-nano", "classify") == 1.0


def test_penalty_is_capped(rep):
    """A bad patch must not exile a model with an unbounded multiplier."""
    _feed(rep, "gpt-5-nano", "code_review", ok_count=0, fail_count=20)
    assert rep.multiplier("gpt-5-nano", "code_review") == 4.0


def test_reputation_is_per_intent_not_per_model(rep):
    """A model can be excellent at one task and hopeless at another. A single
    global score is wrong for both."""
    _feed(rep, "gpt-5-nano", "classify", ok_count=20, fail_count=0)
    _feed(rep, "gpt-5-nano", "code_review", ok_count=2, fail_count=8)

    assert rep.multiplier("gpt-5-nano", "classify") == 1.0
    assert rep.multiplier("gpt-5-nano", "code_review") > 1.0


def test_old_failures_age_out_of_the_window():
    """A model that was fixed should stop being punished for its past."""
    rep = Reputation(window=10, min_samples=5, exploration_rate=0.0)
    _feed(rep, "gpt-5-nano", "classify", ok_count=0, fail_count=10)
    assert rep.multiplier("gpt-5-nano", "classify") == pytest.approx(4.0)

    _feed(rep, "gpt-5-nano", "classify", ok_count=10, fail_count=0)
    assert rep.multiplier("gpt-5-nano", "classify") == 1.0


def test_exploration_ignores_the_penalty_sometimes():
    """Without this a penalised model gets no traffic, so it gets no new
    observations, so it can never recover. The loop becomes a ratchet."""
    never = Reputation(exploration_rate=0.0, rng=random.Random(1))
    assert not any(never.should_explore() for _ in range(200))

    always = Reputation(exploration_rate=1.0, rng=random.Random(1))
    assert all(always.should_explore() for _ in range(50))

    sometimes = Reputation(exploration_rate=0.2, rng=random.Random(7))
    hits = sum(sometimes.should_explore() for _ in range(2000))
    assert 300 < hits < 500, f"expected roughly 20%, got {hits / 2000:.1%}"


def test_snapshot_explains_why_a_model_is_unadjusted(rep):
    _feed(rep, "gpt-5-nano", "classify", ok_count=2, fail_count=0)
    row = next(r for r in rep.snapshot() if r["model"] == "gpt-5-nano")
    assert row["status"] == "insufficient evidence"
    assert row["needs"] == 3

    _feed(rep, "gpt-5-nano", "classify", ok_count=0, fail_count=5)
    row = next(r for r in rep.snapshot() if r["model"] == "gpt-5-nano")
    assert row["status"] == "penalised"
    assert row["needs"] == 0


# -- routing integration ---------------------------------------------------
@pytest.fixture
def router(settings, store, rep) -> Router:
    settings.quality_routing_enabled = True
    return Router(settings, store, {"anthropic", "openai"}, reputation=rep)


async def test_a_failing_model_loses_to_a_pricier_one(router, rep):
    """The whole point: the cheapest model stops winning once it is shown to
    be bad at this particular task."""
    first = await router.route(make_request(), "code_review")
    cheapest = first.model.key

    _feed(rep, cheapest, "code_review", ok_count=0, fail_count=10)
    after = await router.route(make_request(), "code_review")

    assert after.model.key != cheapest


async def test_quality_does_not_override_the_tier_floor(router, rep):
    """Reputation adjusts cost. It must not let a light model serve heavy work."""
    from aigateway.catalog import Tier

    for spec in ("gpt-5", "gpt-5.4", "claude-opus-5"):
        _feed(rep, spec, "architecture", ok_count=0, fail_count=10)

    decision = await router.route(make_request(), "architecture")
    assert decision.tier >= Tier.HEAVY


async def test_candidates_report_sticker_cost_separately_from_the_score(router, rep):
    """One number is what you pay, the other is how we rank. Conflating them
    would make the cost estimate a lie."""
    first = await router.route(make_request(), "code_review")
    penalised = first.model.key
    _feed(rep, penalised, "code_review", ok_count=2, fail_count=8)

    after = await router.route(make_request(), "code_review")
    entry = next(c for c in after.considered if c.model.key == penalised)

    assert entry.quality_multiplier > 1.0
    assert entry.cost_usd > entry.raw_cost_usd
    assert entry.quality_success_rate == pytest.approx(0.2)


async def test_disabling_quality_routing_restores_pure_price(settings, store, rep):
    settings.quality_routing_enabled = False
    router = Router(settings, store, {"anthropic", "openai"}, reputation=rep)

    first = await router.route(make_request(), "code_review")
    _feed(rep, first.model.key, "code_review", ok_count=0, fail_count=20)
    after = await router.route(make_request(), "code_review")

    assert after.model.key == first.model.key


async def test_explanation_says_quality_changed_the_answer(router, rep):
    first = await router.route(make_request(), "code_review")
    _feed(rep, first.model.key, "code_review", ok_count=1, fail_count=9)

    after = await router.route(make_request(), "code_review")
    ex = explain(after, 0.8, "rules")
    verdict = ex["steps"][3]

    assert "quality" in verdict["value"].lower()
    assert first.model.key in verdict["detail"]
    assert "%" in verdict["detail"], "should quote the observed success rate"


async def test_a_pinned_model_is_never_quality_adjusted(router, rep):
    """A pin bypasses the router; second-guessing it would defeat the point."""
    _feed(rep, "claude-opus-5", "classify", ok_count=0, fail_count=20)
    decision = await router.route(make_request(pin_model="claude-opus-5"), "classify")
    assert decision.model.key == "claude-opus-5"
