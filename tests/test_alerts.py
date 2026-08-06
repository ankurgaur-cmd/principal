"""System alerts, and the availability gates that raise them.

Two classes of bug live here, both found by switching a model off and watching
traffic go to it anyway:

* an availability gate the *router* honours but some other code path does not,
  which only ever shows up under failure — exactly when the operator is relying
  on the switch;
* an outage reported to the operator and the end user in the same words, which
  leaves the operator unable to diagnose and the user thinking they broke it.
"""

from __future__ import annotations

import pytest
from conftest import make_request

from aigateway.catalog import CATALOG
from aigateway.config import Settings
from aigateway.errors import NoModelsAvailable
from aigateway.observability.alerts import AlertCentre, no_models_available
from aigateway.providers.registry import ProviderRegistry
from aigateway.providers.switchboard import Switchboard
from aigateway.routing import Router


@pytest.fixture
def board() -> Switchboard:
    return Switchboard()


@pytest.fixture
def registry() -> ProviderRegistry:
    reg = ProviderRegistry(Settings(redis_url=None))
    reg._providers = {"openai": object(), "anthropic": object()}
    return reg


# ==========================================================================
# The gap: fallback ignored the switches
# ==========================================================================
def test_the_fallback_chain_skips_switched_off_models(registry, board):
    """The router refusing to route somewhere and the fallback going there
    anyway is the worst kind of inconsistency — it only surfaces under failure,
    which is the one moment the switch has to be trustworthy."""
    primary = CATALOG["gpt-5"]
    unswitched = registry.fallback_chain(primary)
    assert unswitched, "precondition: there is a chain to filter"

    for spec in unswitched:
        board.set_model(spec.key, False)

    filtered = registry.fallback_chain(primary, switchboard=board)
    assert all(board.is_enabled(m.key, m.provider) for m in filtered)
    assert set(m.key for m in filtered) != set(m.key for m in unswitched)


def test_the_fallback_chain_skips_a_switched_off_vendor(registry, board):
    board.set_provider("openai", False)
    chain = registry.fallback_chain(CATALOG["claude-opus-5"], switchboard=board)
    assert all(m.provider != "openai" for m in chain)


def test_the_fallback_chain_skips_unhealthy_models(registry):
    class Health:
        def is_available(self, key):
            return key != "gpt-5-mini"

    chain = registry.fallback_chain(CATALOG["gpt-5"], health=Health())
    assert all(m.key != "gpt-5-mini" for m in chain)


def test_without_the_gates_nothing_is_filtered(registry, board):
    """Guards the default: callers that pass no gates get the old behaviour,
    so this cannot silently change fallback for anyone who has not opted in."""
    board.set_provider("openai", False)
    assert registry.fallback_chain(CATALOG["gpt-5"]) == registry.fallback_chain(
        CATALOG["gpt-5"], switchboard=None
    )


# ==========================================================================
# An operator switch outranks a caller's pin
# ==========================================================================
async def test_a_switched_off_model_cannot_be_pinned(settings, store, board):
    """A pin is a caller instruction; a switch is an operator decision about
    what this deployment may talk to. Anything else means "off" does not mean
    off at the one moment it matters."""
    router = Router(settings, store, {"anthropic", "openai"}, switchboard=board)
    board.set_model("claude-opus-5", False)

    with pytest.raises(NoModelsAvailable) as exc:
        await router.route(make_request(pin_model="claude-opus-5"), "classify")

    assert "switched off" in str(exc.value.detail)
    # It must say how to proceed, not just refuse.
    assert "pin_model" in str(exc.value.detail) or "operator" in str(exc.value.detail)


async def test_a_pin_still_works_when_the_model_is_on(settings, store, board):
    router = Router(settings, store, {"anthropic", "openai"}, switchboard=board)
    decision = await router.route(make_request(pin_model="claude-opus-5"), "classify")
    assert decision.model.key == "claude-opus-5"
    assert decision.pinned is True


# ==========================================================================
# Diagnosing *why* nothing is left
# ==========================================================================
async def test_all_switched_off_is_diagnosed_as_an_operator_action(settings, store, board):
    """"You turned it off" and "it broke" produce identical empty candidate
    sets and could not be more different to act on."""
    router = Router(settings, store, {"anthropic", "openai"}, switchboard=board)
    board.set_provider("anthropic", False)
    board.set_provider("openai", False)

    with pytest.raises(NoModelsAvailable) as exc:
        await router.route(make_request(), "classify")

    assert exc.value.cause == "all_switched_off"
    assert exc.value.status_code == 503


async def test_no_credentials_is_diagnosed_separately(settings, store):
    router = Router(settings, store, set())
    with pytest.raises(NoModelsAvailable) as exc:
        await router.route(make_request(), "classify")
    assert exc.value.cause == "no_credentials"


async def test_the_error_carries_both_audiences(settings, store, board):
    router = Router(settings, store, {"anthropic", "openai"}, switchboard=board)
    board.set_provider("anthropic", False)
    board.set_provider("openai", False)

    with pytest.raises(NoModelsAvailable) as exc:
        await router.route(make_request(), "classify")

    err = exc.value.detail["error"]
    # The operator's version names models and vendors.
    assert "switched off" in err["message"]
    # The user's version names neither.
    user = err["user_message"]
    assert user and "switched off by an operator" not in user
    for jargon in ("gpt-5", "claude", "tier", "intent", "provider"):
        assert jargon not in user.lower(), f"jargon leaked to the end user: {jargon}"


# ==========================================================================
# The alert centre
# ==========================================================================
def test_an_alert_names_a_remedy_and_who_it_is_for():
    centre = AlertCentre()
    centre.raise_alert(no_models_available("all_unhealthy", "detail here", "do this"))
    snap = centre.snapshot()

    assert snap["ok"] is False
    assert snap["severity"] == "critical"
    assert snap["needs_support"] is True, "a failing fleet needs a human"
    alert = snap["active"][0]
    assert alert["remedy"] == "do this"
    assert alert["detail"] != alert["user_message"]


def test_an_operator_switching_things_off_does_not_need_paging():
    """A deliberate action is not an incident. Flagging it as one is how a
    support channel learns to ignore the flag."""
    centre = AlertCentre()
    centre.raise_alert(no_models_available("all_switched_off", "d", "r"))
    assert centre.snapshot()["needs_support"] is False


def test_re_raising_does_not_reset_the_clock():
    """An outage that has lasted twenty minutes should say twenty minutes, not
    restart at zero every time another request hits it."""
    centre = AlertCentre()
    first = centre.raise_alert(no_models_available("all_unhealthy", "d", "r"))
    raised_at = first.raised_at
    for _ in range(4):
        centre.raise_alert(no_models_available("all_unhealthy", "d", "r"))

    active = centre.snapshot()["active"]
    assert len(active) == 1
    assert active[0]["occurrences"] == 5
    assert centre.active[0].raised_at == raised_at


def test_clearing_leaves_a_record():
    """"It was down for ten minutes and came back" is exactly what an operator
    needs to know, and exactly what a live-only status view hides."""
    centre = AlertCentre()
    centre.raise_alert(no_models_available("all_unhealthy", "d", "r"))
    assert centre.clear("no_models_available") == 1

    snap = centre.snapshot()
    assert snap["ok"] is True
    assert snap["active"] == []
    assert snap["recent"], "the outage still happened"
    assert snap["recent"][0]["cleared_at"] is not None


def test_clearing_is_scoped_by_prefix():
    centre = AlertCentre()
    centre.raise_alert(no_models_available("all_unhealthy", "d", "r"))
    assert centre.clear("something_else") == 0
    assert centre.snapshot()["ok"] is False


def test_every_cause_produces_a_message_for_both_audiences():
    for cause in ("all_switched_off", "all_unhealthy", "no_credentials",
                  "no_capable_model", "something_unmapped"):
        alert = no_models_available(cause, "technical detail", "the remedy")
        assert alert.title and alert.user_message and alert.detail
        assert alert.user_message != alert.detail
        # No cause should leave the user reading a stack of model names.
        assert "technical detail" not in alert.user_message


def test_snapshot_is_serialisable():
    import json

    centre = AlertCentre()
    centre.raise_alert(no_models_available("all_unhealthy", "d", "r"))
    json.dumps(centre.snapshot())
