"""Regression tests for the design-review fixes.

Each class here pins one fix: the pilot's timing rules, hard-mode budget
reservations, catalog-honest cache savings, config coupling and guards,
prefix-validated session warmth, pool pressure as a routing input, the learned
output forecast, and the store-shared switchboard.
"""

from __future__ import annotations

import asyncio

import pytest
from conftest import make_request

from aigateway.cache.pilot import CachePilot, PilotRole
from aigateway.config import Settings
from aigateway.errors import BudgetExceeded
from aigateway.governance import BudgetGuard, CostLedger, price_usage
from aigateway.providers.switchboard import Switchboard
from aigateway.routing import Router
from aigateway.routing.outputs import OutputEstimator
from aigateway.schemas import Usage


# -- cache pilot timing ------------------------------------------------------
class TestPilotTiming:
    async def test_follower_wait_honours_the_per_call_override(self, store):
        pilot = CachePilot(store, wait_ms=60_000)
        assert await pilot.acquire("fp", 300) is PilotRole.PILOT

        loop = asyncio.get_running_loop()
        started = loop.time()
        role = await pilot.acquire("fp", 300, wait_ms=120)
        assert role is PilotRole.TIMEOUT
        assert loop.time() - started < 2, "override must beat the configured wait"

    async def test_a_dead_pilots_seat_is_taken_over(self, store):
        """The lock going free without warmth means the pilot died; a waiting
        follower should take the seat instead of timing out pointlessly."""
        pilot = CachePilot(store, wait_ms=5_000)
        assert await pilot.acquire("fp", 300) is PilotRole.PILOT

        async def die_soon():
            await asyncio.sleep(0.05)
            await pilot.release_failed("fp")

        takeover, _ = await asyncio.gather(pilot.acquire("fp", 300), die_soon())
        assert takeover is PilotRole.PILOT

    async def test_heartbeat_keeps_a_live_pilot_in_the_seat(self, store, monkeypatch):
        """A fixed short lock TTL re-elected a second pilot mid-flight — the
        duplicate cold write the module exists to prevent."""
        from aigateway.cache import pilot as pm

        monkeypatch.setattr(pm, "LOCK_TTL_SECONDS", 0.2)
        pilot = CachePilot(store, wait_ms=1)
        role = await pilot.acquire("fp", 300)
        assert role is PilotRole.PILOT

        async with pilot.holding("fp", role):
            await asyncio.sleep(0.5)  # well past the unrefreshed TTL
            assert await pilot.acquire("fp", 300, wait_ms=1) is PilotRole.TIMEOUT

    async def test_first_token_warmth_releases_followers(self, store):
        pilot = CachePilot(store)
        assert await pilot.acquire("fp", 300) is PilotRole.PILOT
        await pilot.mark_warm("fp", 300)  # the streaming path fires this at chunk 1
        assert await pilot.acquire("fp", 300) is PilotRole.WARM


# -- hard-mode budget reservation --------------------------------------------
class TestBudgetReservation:
    @pytest.fixture
    def guard(self, store):
        settings = Settings(redis_url=None, budget_mode="hard", default_tenant_daily_usd=50.0)
        return BudgetGuard(settings, store, CostLedger(store)), CostLedger(store)

    async def test_concurrent_checks_cannot_both_pass_on_one_snapshot(self, guard):
        g, ledger = guard
        first = await g.check("t", 30.0)
        assert first.reserved_usd == 30.0
        with pytest.raises(BudgetExceeded):
            await g.check("t", 30.0)  # 30 reserved + 30 projected > 50
        # The rejected request's estimate was refunded, not kept.
        assert await ledger.spend_today("t") == pytest.approx(30.0)

    async def test_settlement_replaces_the_estimate_with_actuals(self, guard):
        g, ledger = guard
        verdict = await g.check("t", 30.0)

        from aigateway.catalog import get_model

        priced = price_usage(
            Usage(prompt_tokens=100_000, completion_tokens=0), get_model("gpt-5")
        )
        await ledger.record("t", "a", "gpt-5", priced, reserved_usd=verdict.reserved_usd)
        # Actual ($0.125) replaces the $30 estimate rather than stacking on it.
        assert await ledger.spend_today("t") == pytest.approx(0.125)

    async def test_release_refunds_an_unserved_request(self, guard):
        g, ledger = guard
        verdict = await g.check("t", 30.0)
        await g.release("t", verdict)
        assert await ledger.spend_today("t") == pytest.approx(0.0)


# -- savings priced from the catalog -----------------------------------------
def test_cache_savings_use_the_models_own_read_multiplier():
    from aigateway.catalog import get_model

    model = get_model("claude-opus-5")
    priced = price_usage(
        Usage(prompt_tokens=1_000_000, cache_read_tokens=1_000_000), model
    )
    price_in = model.price_in_per_mtok
    expected = price_in * (1.0 - model.cache_read_multiplier)
    assert priced.cache_savings_usd == pytest.approx(expected)
    # The old formula hardcoded read=0.1x as `read_cost * 9`.
    assert priced.cache_savings_usd <= price_in


# -- config coupling and guards ----------------------------------------------
def test_session_ttl_follows_the_cache_ttl_unless_chosen():
    assert Settings(redis_url=None, cache_ttl="1h").session_ttl_seconds == 3600
    assert Settings(redis_url=None, cache_ttl="5m").session_ttl_seconds == 300
    explicit = Settings(redis_url=None, cache_ttl="1h", session_ttl_seconds=120)
    assert explicit.session_ttl_seconds == 120, "an operator's choice stands"


def test_jwt_mode_refuses_the_default_secret():
    with pytest.raises(Exception, match="JWT_SECRET"):
        Settings(redis_url=None, auth_mode="jwt")
    ok = Settings(redis_url=None, auth_mode="jwt", jwt_secret="s3cret-enough")
    assert ok.auth_mode == "jwt"


# -- prefix-validated session warmth -----------------------------------------
class TestPrefixWarmth:
    async def test_growing_history_keeps_the_session_warm(self, router):
        req = make_request(system_tokens=6000, session_id="w1")
        first = await router.route(req, "code_review")
        await router.remember("w1", first.model, req)

        # Next turn: same system/tools, more history — the normal case.
        again = make_request(system_tokens=6000, session_id="w1")
        second = await router.route(again, "code_review")
        assert second.cache_state == "warm_read"

    async def test_a_changed_system_prompt_invalidates_warmth(self, router):
        req = make_request(system_tokens=6000, session_id="w2")
        first = await router.route(req, "code_review")
        await router.remember("w2", first.model, req)

        changed = make_request(system_tokens=6000, session_id="w2")
        changed.system = ["y" * len(changed.system[0])]  # same size, new bytes
        second = await router.route(changed, "code_review")
        # The provider's prefix match dies at position zero; pricing a warm
        # read here flattered the sticky model on exactly the requests where
        # switching would have been free.
        assert second.cache_state != "warm_read"
        assert second.sticky is False

    async def test_legacy_records_without_a_fingerprint_stay_warm(self, router):
        req = make_request(system_tokens=6000, session_id="w3")
        first = await router.route(req, "code_review")
        await router.remember("w3", first.model)  # no canonical → no fingerprint
        second = await router.route(
            make_request(system_tokens=6000, session_id="w3"), "code_review"
        )
        assert second.cache_state == "warm_read"


# -- pool pressure as a routing input ----------------------------------------
class TestPoolPressure:
    @pytest.fixture
    def pressured(self, settings, store):
        from aigateway.governance import RateLimiter

        limiter = RateLimiter(settings, store)
        router = Router(
            settings, store, {"anthropic", "openai"}, limiter=limiter
        )
        return router, limiter

    async def test_a_saturated_pool_is_excluded(self, pressured, settings):
        router, limiter = pressured
        baseline = await router.route(make_request(), "classify")
        pool = baseline.model.rate_limit_pool

        settings.pool_rpm_limits = {pool: 3}
        for _ in range(3):
            await limiter.note_upstream(pool)

        decision = await router.route(make_request(), "classify")
        assert decision.model.rate_limit_pool != pool
        assert any(e["kind"] == "pool_saturated" for e in decision.excluded)

    async def test_a_nearly_full_pool_is_penalised_not_excluded(self, pressured, settings):
        router, limiter = pressured
        baseline = await router.route(make_request(), "classify")
        pool = baseline.model.rate_limit_pool

        settings.pool_rpm_limits = {pool: 10}
        for _ in range(9):  # 90% of the window
            await limiter.note_upstream(pool)

        decision = await router.route(make_request(), "classify")
        chosen = next(c for c in decision.considered if c.model.key == decision.model.key)
        affected = [
            c for c in decision.considered if c.model.rate_limit_pool == pool
        ]
        assert affected and all("pool at 90%" in c.note for c in affected)
        # Still allowed to win if it is cheap enough — a nudge, not a ban.
        assert chosen is not None


# -- learned output forecast --------------------------------------------------
class TestOutputEstimator:
    def test_abstains_below_the_evidence_floor(self):
        est = OutputEstimator(min_samples=8)
        for _ in range(7):
            est.record("classify", 50)
        assert est.expected("classify") is None

    def test_median_resists_truncation_outliers(self):
        est = OutputEstimator(min_samples=8)
        for _ in range(20):
            est.record("classify", 60)
        est.record("classify", 16_000)  # one runaway
        assert est.expected("classify") == 60

    async def test_router_prices_from_the_learned_volume(self, settings, store):
        heavy = OutputEstimator(min_samples=1)
        for _ in range(5):
            heavy.record("classify", 8_000)
        providers = {"anthropic", "openai"}

        light = await Router(settings, store, providers).route(
            make_request(max_tokens=16_000), "classify"
        )
        weighted = await Router(settings, store, providers, outputs=heavy).route(
            make_request(max_tokens=16_000), "classify"
        )
        assert weighted.estimated_cost_usd > light.estimated_cost_usd


# -- switchboard shared through the store ------------------------------------
class TestSharedSwitchboard:
    async def test_a_switch_in_one_worker_is_seen_by_another(self, store):
        worker_a, worker_b = Switchboard(store), Switchboard(store)
        worker_a.set_provider("openai", False)
        await worker_a.save()

        assert worker_b.is_enabled("gpt-5", "openai") is True  # stale mirror
        await worker_b.refresh()
        assert worker_b.is_enabled("gpt-5", "openai") is False

    async def test_routing_follows_switches_made_elsewhere(self, settings, store):
        operator_console = Switchboard(store)
        serving_worker = Switchboard(store)
        router = Router(
            settings, store, {"anthropic", "openai"}, switchboard=serving_worker
        )

        before = await router.route(make_request(), "classify")
        assert before.model.provider == "openai"

        operator_console.set_provider("openai", False)
        await operator_console.save()

        after = await router.route(make_request(), "classify")
        assert after.model.provider == "anthropic"

    async def test_without_a_store_it_degrades_to_per_process(self):
        board = Switchboard()  # dev shape: no store, everything local
        board.set_model("gpt-5", False)
        await board.refresh()  # must not raise or reset local state
        assert board.is_enabled("gpt-5", "openai") is False
