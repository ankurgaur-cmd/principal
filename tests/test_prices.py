"""Live price refresh: feed application, guard rails, and persistence.

The router selects on catalog prices, so every rule here defends the same
invariant: a feed can update rates, but never silently and never absurdly.
"""

from __future__ import annotations

import json

import pytest

from aigateway.catalog import CATALOG, get_model
from aigateway.prices import PriceFeed


@pytest.fixture(autouse=True)
def restore_catalog():
    """apply() mutates the global catalog; every test leaves it as found."""
    snapshot = dict(CATALOG)
    yield
    CATALOG.clear()
    CATALOG.update(snapshot)


@pytest.fixture
def feed(store) -> PriceFeed:
    return PriceFeed(store, url="https://feed.example/prices.json")


def _entry(price_in: float, price_out: float) -> dict:
    # LiteLLM shape: cost per single token.
    return {
        "input_cost_per_token": price_in / 1e6,
        "output_cost_per_token": price_out / 1e6,
    }


def test_a_changed_rate_is_applied_with_provenance(feed):
    spec = get_model("gpt-5")
    report = feed.apply(
        {spec.vendor_model_id: _entry(spec.price_in_per_mtok * 2, spec.price_out_per_mtok)},
        source="https://feed.example/prices.json",
    )

    assert [u["model"] for u in report["updated"]] == ["gpt-5"]
    updated = get_model("gpt-5")
    assert updated.price_in_per_mtok == spec.price_in_per_mtok * 2
    assert updated.price_verified is True
    assert updated.price_source == "https://feed.example/prices.json"
    assert updated.price_checked  # dated, so staleness stays detectable


def test_an_unchanged_rate_still_bumps_the_freshness_date(feed):
    """Confirmation is information — the checked date is the whole defence
    against silent list-price drift."""
    spec = get_model("claude-opus-5")
    report = feed.apply(
        {spec.vendor_model_id: _entry(spec.price_in_per_mtok, spec.price_out_per_mtok)},
        source="feed",
    )
    assert "claude-opus-5" in report["confirmed"]
    assert get_model("claude-opus-5").price_checked != spec.price_checked


def test_an_absurd_swing_is_reported_not_applied(feed):
    """One bad feed row must not re-route production traffic."""
    spec = get_model("gpt-5-nano")
    report = feed.apply(
        {spec.vendor_model_id: _entry(spec.price_in_per_mtok * 100, spec.price_out_per_mtok)},
        source="feed",
    )
    assert [s["model"] for s in report["suspicious"]] == ["gpt-5-nano"]
    assert get_model("gpt-5-nano").price_in_per_mtok == spec.price_in_per_mtok


def test_models_absent_from_the_feed_are_named_and_untouched(feed):
    before = {k: (m.price_in_per_mtok, m.price_out_per_mtok) for k, m in CATALOG.items()}
    report = feed.apply({}, source="feed")
    assert sorted(report["missing"]) == sorted(CATALOG)
    assert not report["updated"]
    for key, m in CATALOG.items():
        assert (m.price_in_per_mtok, m.price_out_per_mtok) == before[key]


async def test_overrides_survive_a_restart(store):
    """Reverting to code-time prices on restart would make the ledger disagree
    with yesterday's routing decisions."""
    feed = PriceFeed(store, url="https://feed.example/prices.json")
    spec = get_model("gpt-5-mini")
    report = feed.apply(
        {spec.vendor_model_id: _entry(spec.price_in_per_mtok * 1.5, spec.price_out_per_mtok)},
        source="feed",
    )
    await feed._persist(report)

    # Simulate restart: catalog reverts to code-time values.
    CATALOG["gpt-5-mini"] = spec
    fresh = PriceFeed(store, url="https://feed.example/prices.json")
    applied = await fresh.restore()

    assert applied >= 1
    assert get_model("gpt-5-mini").price_in_per_mtok == pytest.approx(
        spec.price_in_per_mtok * 1.5
    )


async def test_router_selects_on_refreshed_prices(feed, settings, store):
    """The point of the feature: a repricing changes routing, immediately."""
    from conftest import make_request

    from aigateway.routing import Router

    router = Router(settings, store, {"anthropic", "openai"})
    before = await router.route(make_request(), "classify")

    # Reprice the winner up 5x (within the sanity bound) — it should lose.
    winner = get_model(before.model.key)
    feed.apply(
        {winner.vendor_model_id: _entry(
            winner.price_in_per_mtok * 5, winner.price_out_per_mtok * 5
        )},
        source="feed",
    )
    after = await router.route(make_request(), "classify")
    assert after.model.key != before.model.key


def test_malformed_feed_entries_are_treated_as_missing(feed):
    spec = get_model("gpt-5")
    report = feed.apply(
        {spec.vendor_model_id: {"input_cost_per_token": "not-a-number"}}, source="feed"
    )
    assert "gpt-5" in report["missing"]


async def test_daily_timer_only_starts_when_configured(store):
    off = PriceFeed(store, url="", refresh_hours=24)
    off.start()
    assert off._task is None

    disabled = PriceFeed(store, url="https://feed.example/x.json", refresh_hours=0)
    disabled.start()
    assert disabled._task is None

    on = PriceFeed(store, url="https://feed.example/x.json", refresh_hours=24)
    on.start()
    assert on._task is not None
    await on.stop()


async def test_last_report_is_persisted_for_the_console(store):
    feed = PriceFeed(store, url="https://feed.example/prices.json")
    spec = get_model("gpt-5")
    report = feed.apply(
        {spec.vendor_model_id: _entry(spec.price_in_per_mtok, spec.price_out_per_mtok)},
        source="feed",
    )
    await feed._persist(report)
    last = await feed.last_report()
    assert last["confirmed"] == 1
    assert json.dumps(last)  # JSON-serialisable for the admin endpoint
