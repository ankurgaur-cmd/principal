"""Budget, ledger, and rate limiting."""

from __future__ import annotations

import pytest

from aigateway.catalog import Tier, get_model
from aigateway.config import Settings
from aigateway.errors import BudgetExceeded, RateLimited
from aigateway.governance import BudgetGuard, CostLedger, RateLimiter, price_usage
from aigateway.schemas import Usage
from aigateway.state import MemoryStore


def test_cached_input_is_priced_differently_from_fresh_input():
    """A ledger that prices all prompt tokens at list rate overstates a
    cache-heavy workload and understates model thrashing."""
    model = get_model("claude-opus-5")
    fresh = price_usage(Usage(prompt_tokens=10_000, completion_tokens=500), model)
    cached = price_usage(
        Usage(prompt_tokens=10_000, completion_tokens=500, cache_read_tokens=9_000), model
    )
    assert cached.total_usd < fresh.total_usd
    assert cached.cache_savings_usd > 0


def test_cache_write_costs_more_than_fresh_input():
    """Writes are 1.25x at 5m and 2x at 1h — a write is an investment, and the
    router has to know it does not pay off on a single request."""
    model = get_model("claude-opus-5")
    usage = Usage(prompt_tokens=10_000, completion_tokens=0, cache_write_tokens=10_000)
    five_min = price_usage(usage, model, "5m")
    one_hour = price_usage(usage, model, "1h")
    fresh = price_usage(Usage(prompt_tokens=10_000, completion_tokens=0), model)

    assert five_min.total_usd > fresh.total_usd
    assert one_hour.total_usd > five_min.total_usd


async def test_budget_allows_under_limit():
    store = MemoryStore()
    settings = Settings(redis_url=None, default_tenant_daily_usd=10.0)
    ledger = CostLedger(store)
    guard = BudgetGuard(settings, store, ledger)

    verdict = await guard.check("t1", 0.01)
    assert verdict.allowed and verdict.tier_ceiling is None


async def test_soft_budget_degrades_before_it_fails():
    store = MemoryStore()
    settings = Settings(redis_url=None, default_tenant_daily_usd=1.0, budget_mode="soft")
    ledger = CostLedger(store)
    guard = BudgetGuard(settings, store, ledger)

    await store.incr_float(ledger._tenant_key("t2"), 0.95)
    verdict = await guard.check("t2", 0.20)  # would cross the limit

    assert verdict.allowed
    assert verdict.tier_ceiling == Tier.LIGHT
    assert "degraded" in verdict.message


async def test_soft_budget_eventually_fails():
    store = MemoryStore()
    settings = Settings(redis_url=None, default_tenant_daily_usd=1.0, budget_mode="soft")
    ledger = CostLedger(store)
    guard = BudgetGuard(settings, store, ledger)

    await store.incr_float(ledger._tenant_key("t3"), 1.5)  # already over
    with pytest.raises(BudgetExceeded):
        await guard.check("t3", 0.01)


async def test_hard_budget_rejects_immediately():
    store = MemoryStore()
    settings = Settings(redis_url=None, default_tenant_daily_usd=1.0, budget_mode="hard")
    ledger = CostLedger(store)
    guard = BudgetGuard(settings, store, ledger)

    await store.incr_float(ledger._tenant_key("t4"), 0.99)
    with pytest.raises(BudgetExceeded):
        await guard.check("t4", 0.5)


async def test_exact_preflight_only_near_the_cap():
    """count_tokens is a real round trip; spend it only when it can change
    the outcome."""
    store = MemoryStore()
    settings = Settings(
        redis_url=None, default_tenant_daily_usd=10.0, preflight_exact_threshold=0.85
    )
    ledger = CostLedger(store)
    guard = BudgetGuard(settings, store, ledger)

    assert (await guard.check("t5", 0.01)).needs_exact_preflight is False

    await store.incr_float(ledger._tenant_key("t5"), 9.0)
    assert (await guard.check("t5", 0.01)).needs_exact_preflight is True


async def test_rate_limiter_enforces_rpm():
    store = MemoryStore()
    settings = Settings(redis_url=None, default_tenant_rpm=3)
    limiter = RateLimiter(settings, store)

    for _ in range(3):
        await limiter.check_tenant("t6")
    with pytest.raises(RateLimited):
        await limiter.check_tenant("t6")


async def test_upstream_pools_are_tracked_separately():
    """Model tiers sit in separate upstream pools, so shedding load between
    them is a capacity lever and not only a cost lever."""
    store = MemoryStore()
    limiter = RateLimiter(Settings(redis_url=None), store)

    await limiter.note_upstream("anthropic-opus-5")
    await limiter.note_upstream("anthropic-opus-5")
    await limiter.note_upstream("anthropic-haiku")

    assert await limiter.pool_pressure("anthropic-opus-5") == 2
    assert await limiter.pool_pressure("anthropic-haiku") == 1
