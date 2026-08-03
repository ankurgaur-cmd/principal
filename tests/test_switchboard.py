"""Operator on/off switches, and their interaction with health and pricing."""

from __future__ import annotations

import pytest
from conftest import make_request

from aigateway.catalog import Tier
from aigateway.errors import NoCapableModel
from aigateway.providers.switchboard import Switchboard
from aigateway.routing import Router, explain


@pytest.fixture
def board() -> Switchboard:
    return Switchboard()


@pytest.fixture
def router(settings, store, board) -> Router:
    return Router(settings, store, {"anthropic", "openai"}, switchboard=board)


def test_everything_is_on_by_default(board):
    """A restart returns to 'everything the credentials allow' — the safe default."""
    assert board.is_enabled("gpt-5-nano", "openai") is True
    assert board.reason("gpt-5-nano", "openai") is None


def test_disabling_a_model_gives_a_reason(board):
    board.set_model("gpt-5-nano", False)
    assert board.is_enabled("gpt-5-nano", "openai") is False
    assert "operator" in board.reason("gpt-5-nano", "openai")
    # Siblings are untouched.
    assert board.is_enabled("gpt-5-mini", "openai") is True


def test_disabling_a_vendor_covers_all_of_its_models(board):
    board.set_provider("openai", False)
    assert board.is_enabled("gpt-5-nano", "openai") is False
    assert board.is_enabled("gpt-5", "openai") is False
    assert board.is_enabled("claude-opus-5", "anthropic") is True
    assert "vendor 'openai'" in board.reason("gpt-5", "openai")


def test_state_distinguishes_model_off_from_vendor_off(board):
    """Otherwise the two look identical in the UI and the toggle is confusing."""
    board.set_model("gpt-5-nano", False)
    board.set_provider("anthropic", False)
    by_key = {m["model"]: m for m in board.state()["models"]}

    assert by_key["gpt-5-nano"]["enabled"] is False
    assert by_key["gpt-5-nano"]["disabled_by_provider"] is False

    assert by_key["claude-opus-5"]["enabled"] is False
    assert by_key["claude-opus-5"]["disabled_by_provider"] is True


def test_unknown_names_are_rejected(board):
    with pytest.raises(NoCapableModel):
        board.set_model("not-a-model", False)
    with pytest.raises(NoCapableModel):
        board.set_provider("not-a-vendor", False)


def test_reset_switches_everything_back_on(board):
    board.set_model("gpt-5", False)
    board.set_provider("anthropic", False)
    board.reset()
    assert board.state()["disabled_models"] == []
    assert board.state()["disabled_providers"] == []


# -- routing integration ---------------------------------------------------
async def test_switching_a_vendor_off_reroutes_to_the_other(router, board):
    """The point of the feature: watch where traffic goes instead."""
    before = await router.route(make_request(), "code_review")
    assert before.model.provider == "openai"

    board.set_provider("openai", False)
    after = await router.route(make_request(), "code_review")

    assert after.model.provider == "anthropic"
    assert after.tier >= Tier.HEAVY, "failover must not quietly drop capability"


async def test_switching_the_cheapest_model_off_promotes_the_next(router, board):
    first = await router.route(make_request(), "classify")
    board.set_model(first.model.key, False)
    second = await router.route(make_request(), "classify")

    assert second.model.key != first.model.key
    assert second.estimated_cost_usd >= first.estimated_cost_usd


async def test_switched_off_models_appear_as_ruled_out_with_a_plain_reason(router, board):
    board.set_provider("openai", False)
    decision = await router.route(make_request(), "code_review")
    ex = explain(decision, 0.8, "rules")

    excluded = {e["model"]: e["plain"] for e in ex["excluded"]}
    assert any("switched off by you" in v for v in excluded.values())


async def test_switching_everything_off_fails_loudly(router, board):
    board.set_provider("openai", False)
    board.set_provider("anthropic", False)
    with pytest.raises(NoCapableModel) as exc:
        await router.route(make_request(), "classify")
    assert "switched off" in str(exc.value.detail)


async def test_a_dry_run_preview_also_respects_the_switches(router, board):
    """A preview that ignored your switches would not be previewing your setup."""
    board.set_provider("openai", False)
    decision = await router.route(make_request(), "classify", require_available=False)
    assert decision.model.provider == "anthropic"


async def test_an_operator_switch_is_not_healed_by_the_circuit_breaker(settings, store, board):
    """Health recovers on its own; an operator decision must not."""
    from aigateway.providers.health import HealthMonitor

    class _Reg:
        enabled = {"anthropic", "openai"}

    health = HealthMonitor(_Reg())
    r = Router(settings, store, {"anthropic", "openai"}, health=health, switchboard=board)

    board.set_model("gpt-5-nano", False)
    health.record_success("gpt-5-nano", 50)  # would close any breaker

    decision = await r.route(make_request(), "classify")
    assert decision.model.key != "gpt-5-nano"


# -- vendor weighting ------------------------------------------------------
async def test_vendor_weight_can_override_a_pure_price_decision(settings, store):
    """Price-only routing always concentrates on the cheapest vendor. The
    weight is how you express a preference that price cannot."""
    neutral = await Router(settings, store, {"anthropic", "openai"}).route(
        make_request(), "code_review"
    )
    assert neutral.model.provider == "openai", "on published rates OpenAI is cheaper"

    settings.vendor_weights = {"anthropic": 0.2}
    weighted = await Router(settings, store, {"anthropic", "openai"}).route(
        make_request(), "code_review"
    )
    assert weighted.model.provider == "anthropic"


async def test_vendor_weight_never_changes_what_you_are_billed(settings, store):
    """The thumb on the scale belongs to the score, never to the ledger."""
    from aigateway.governance import price_usage
    from aigateway.schemas import Usage

    settings.vendor_weights = {"anthropic": 0.2}
    decision = await Router(settings, store, {"anthropic", "openai"}).route(
        make_request(), "code_review"
    )
    priced = price_usage(Usage(prompt_tokens=1_000_000, completion_tokens=0), decision.model)
    assert priced.total_usd == pytest.approx(decision.model.price_in_per_mtok)


# -- pricing provenance ----------------------------------------------------
def test_every_price_is_traceable_to_a_source_and_date():
    from aigateway.catalog import CATALOG

    for spec in CATALOG.values():
        assert spec.price_verified, f"{spec.key} still has placeholder pricing"
        assert spec.price_source.startswith("https://"), f"{spec.key} has no source"
        assert spec.price_checked, f"{spec.key} has no verification date"


def test_lapsed_promotional_pricing_is_detected():
    """A promo rate that quietly expires re-routes traffic with no code change."""
    from aigateway.catalog import stale_prices

    assert stale_prices("2026-08-15") == [], "intro pricing is valid through August"

    lapsed = {s["model"]: s for s in stale_prices("2026-09-02")}
    assert "claude-sonnet-5" in lapsed
    assert lapsed["claude-sonnet-5"]["catalog_price"] == (2.00, 10.00)
    assert lapsed["claude-sonnet-5"]["actual_price"] == (3.00, 15.00)
