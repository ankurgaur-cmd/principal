"""Does the cache the router priced actually turn up?

The router discounts input cost by ~90% on the strength of an assumed warm read.
Measured on this gateway at an identical 1,095-token prompt, two of four models
never returned a single cached token — and because the router picks the cheapest
candidate, a model that fails to cache is favoured by the very discount it does
not earn. The error compounds in the worst direction, which is why it is worth
checking the assumption rather than trusting the catalog.
"""

from __future__ import annotations

import pytest
from conftest import make_request

from aigateway.cache.effectiveness import (
    DEAD_CACHE_RATE,
    MIN_SAMPLES,
    CacheEffectiveness,
)
from aigateway.catalog import CATALOG
from aigateway.routing import Router


@pytest.fixture
def eff() -> CacheEffectiveness:
    return CacheEffectiveness()


def feed(eff, model, *, delivered: bool, n: int) -> None:
    for _ in range(n):
        eff.record(model, expected_hit=True, cached_tokens=1024 if delivered else 0)


# -- what counts as evidence ----------------------------------------------
def test_a_cold_write_returning_nothing_is_not_a_miss(eff):
    """Correct behaviour, not a failure. Counting it would condemn every model
    on its first request, before it had any chance to cache."""
    for _ in range(20):
        eff.record("gpt-5", expected_hit=False, cached_tokens=0)
    assert eff.hit_rate("gpt-5") is None
    assert eff.delivers("gpt-5") is True


def test_only_expected_hits_are_recorded(eff):
    eff.record("gpt-5", expected_hit=False, cached_tokens=0)
    eff.record("gpt-5", expected_hit=True, cached_tokens=0)
    assert eff.snapshot()[0]["samples"] == 1


# -- no evidence, no adjustment -------------------------------------------
def test_an_unproven_model_gets_the_benefit_of_the_doubt(eff):
    """The alternative is routing on the order requests happened to arrive."""
    feed(eff, "gpt-5-nano", delivered=False, n=MIN_SAMPLES - 1)
    assert eff.hit_rate("gpt-5-nano") is None
    assert eff.delivers("gpt-5-nano") is True


def test_an_unknown_model_is_trusted(eff):
    assert eff.delivers("never-seen") is True


# -- the finding this exists for ------------------------------------------
def test_a_model_that_never_delivers_stops_being_credited(eff):
    """gpt-5-nano and gpt-5.4-mini, measured: 0 cached tokens at a prompt size
    where gpt-5-mini cached 1,024."""
    feed(eff, "gpt-5-nano", delivered=False, n=MIN_SAMPLES)
    assert eff.hit_rate("gpt-5-nano") == 0.0
    assert eff.delivers("gpt-5-nano") is False


def test_a_model_that_delivers_keeps_its_discount(eff):
    feed(eff, "gpt-5-mini", delivered=True, n=MIN_SAMPLES)
    assert eff.delivers("gpt-5-mini") is True


def test_an_occasional_miss_is_a_cache_working_normally(eff):
    """Vendors evict under load. One miss in ten is not a broken promise, and
    treating it as one would throw away a cache that mostly works."""
    feed(eff, "gpt-5-mini", delivered=True, n=9)
    feed(eff, "gpt-5-mini", delivered=False, n=1)
    assert eff.hit_rate("gpt-5-mini") > DEAD_CACHE_RATE
    assert eff.delivers("gpt-5-mini") is True


def test_a_model_can_recover(eff):
    """A vendor that ships prefix caching next month must climb back out on its
    own — the window rolls, so nothing is condemned permanently."""
    eff = CacheEffectiveness(window=6, min_samples=MIN_SAMPLES)
    feed(eff, "gpt-5-nano", delivered=False, n=6)
    assert eff.delivers("gpt-5-nano") is False

    feed(eff, "gpt-5-nano", delivered=True, n=6)
    assert eff.delivers("gpt-5-nano") is True


# -- the effect on routing ------------------------------------------------
async def test_the_router_stops_pricing_a_cache_that_never_arrives(settings, store, eff):
    """The whole point: a model that does not cache must not keep winning on a
    discount it does not earn.

    The comparison has to be made on a *warm* session. On a cold one, losing
    cacheability actually makes a model slightly cheaper — it no longer pays the
    1.25x write premium — which is correct, and not where the error was. The
    damage is on the warm path, where the router hands out a 0.1x read rate for
    a hit that never lands.
    """
    victim = "gpt-5-mini"
    big = make_request(system_tokens=20_000, session_id="ce-warm")

    trusting = Router(settings, store, {"anthropic", "openai"})
    await trusting.remember("ce-warm", CATALOG[victim])
    before = await trusting.route(big, "code_write")
    warm_row = next(c for c in before.considered if c.model.key == victim)
    assert warm_row.cache_state == "warm_read", "precondition: priced as a warm read"

    feed(eff, victim, delivered=False, n=MIN_SAMPLES)

    informed = Router(settings, store, {"anthropic", "openai"}, cache_effectiveness=eff)
    after = await informed.route(big, "code_write")
    row = next(c for c in after.considered if c.model.key == victim)

    assert row.plan.cacheable is False
    assert "does not return cached tokens" in row.plan.reason
    assert row.cost_usd > warm_row.cost_usd, (
        "forfeiting a read rate it never earned must make it more expensive"
    )


async def test_a_cold_request_is_not_charged_a_write_premium_it_will_not_pay(
    settings, store, eff
):
    """The other half, and the reason the warm case had to be isolated: if a
    model will not cache, it also will not pay the 1.25x cold-write premium.
    Pricing that premium would be wrong in the opposite direction."""
    victim = "gpt-5-mini"
    cold = make_request(system_tokens=20_000, session_id="ce-cold")

    before = await Router(settings, store, {"anthropic", "openai"}).route(cold, "code_write")
    cold_row = next(c for c in before.considered if c.model.key == victim)
    assert cold_row.cache_state == "cold_write"

    feed(eff, victim, delivered=False, n=MIN_SAMPLES)
    after = await Router(
        settings, store, {"anthropic", "openai"}, cache_effectiveness=eff
    ).route(cold, "code_write")
    row = next(c for c in after.considered if c.model.key == victim)

    assert row.cache_state == "uncached"
    assert row.cost_usd < cold_row.cost_usd


async def test_models_that_deliver_are_untouched(settings, store, eff):
    feed(eff, "claude-opus-5", delivered=True, n=MIN_SAMPLES)
    big = make_request(system_tokens=20_000, session_id="ce-2")

    a = await Router(settings, store, {"anthropic", "openai"}).route(big, "code_write")
    b = await Router(
        settings, store, {"anthropic", "openai"}, cache_effectiveness=eff
    ).route(big, "code_write")
    assert a.model.key == b.model.key


# -- reporting -------------------------------------------------------------
def test_the_snapshot_says_what_it_knows_and_what_it_needs(eff):
    feed(eff, "gpt-5-nano", delivered=False, n=MIN_SAMPLES)
    feed(eff, "gpt-5-mini", delivered=True, n=1)
    rows = {r["model"]: r for r in eff.snapshot()}

    assert rows["gpt-5-nano"]["trusted"] is False
    assert "never returns cached tokens" in rows["gpt-5-nano"]["status"]
    assert rows["gpt-5-mini"]["status"] == "learning"
    assert rows["gpt-5-mini"]["needs"] == MIN_SAMPLES - 1


def test_snapshot_is_serialisable(eff):
    import json

    feed(eff, "gpt-5", delivered=True, n=3)
    json.dumps(eff.snapshot())
