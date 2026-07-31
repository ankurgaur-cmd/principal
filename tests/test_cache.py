"""Cache planning, fingerprinting, and the fan-out pilot."""

from __future__ import annotations

import asyncio

from conftest import make_request

from aigateway.cache import CachePilot, PilotRole, plan_cache, prefix_fingerprint
from aigateway.catalog import get_model
from aigateway.state import MemoryStore


def test_short_prefix_is_not_cacheable_and_says_why():
    """The silent failure this guards against: the API accepts a cache marker
    on a too-short prefix and simply never caches. No error, no hit."""
    haiku = get_model("claude-haiku-4-5")  # 4096-token minimum, the highest
    plan = plan_cache(make_request(system_tokens=500), haiku)
    assert plan.cacheable is False
    assert "below" in plan.reason and "minimum" in plan.reason


def test_min_cacheable_tokens_is_not_monotonic_across_models():
    """A 3k prefix caches on Opus 5 (512) and not on Haiku 4.5 (4096).

    A router that assumes 'cheaper model = cheaper request' gets this wrong.
    """
    req = make_request(system_tokens=3000)
    assert plan_cache(req, get_model("claude-opus-5")).cacheable is True
    assert plan_cache(req, get_model("claude-sonnet-5")).cacheable is True
    assert plan_cache(req, get_model("claude-haiku-4-5")).cacheable is False


def test_system_breakpoint_absorbs_tools():
    """Render order is tools -> system -> messages, so a marker on the last
    system block already covers tools. Spending a second of only four markers
    on tools is waste."""
    req = make_request(system_tokens=5000, tools=3)
    plan = plan_cache(req, get_model("claude-opus-5"))
    assert "system" in plan.breakpoints
    assert "tools" not in plan.breakpoints
    assert len(plan.breakpoints) <= 4


def test_fingerprint_is_model_scoped():
    """Caches are model-scoped, so the same prefix on two models must be two
    distinct fingerprints — otherwise the pilot marks one warm and followers on
    the other model read a cache that does not exist for them."""
    req = make_request(system_tokens=5000)
    assert prefix_fingerprint(req, "claude-opus-5") != prefix_fingerprint(req, "claude-sonnet-5")


def test_fingerprint_ignores_the_volatile_last_turn():
    a = make_request(system_tokens=5000, user_text="question one")
    b = make_request(system_tokens=5000, user_text="a completely different question")
    assert prefix_fingerprint(a, "claude-opus-5") == prefix_fingerprint(b, "claude-opus-5")


def test_fingerprint_is_stable_under_tool_reordering():
    """Tool order churn invalidates the prefix on every provider."""
    from aigateway.schemas import ToolDef

    a = make_request(system_tokens=5000)
    a.tools = [ToolDef(name="beta"), ToolDef(name="alpha")]
    b = make_request(system_tokens=5000)
    b.tools = [ToolDef(name="alpha"), ToolDef(name="beta")]
    assert prefix_fingerprint(a, "claude-opus-5") == prefix_fingerprint(b, "claude-opus-5")


async def test_pilot_serialises_fan_out():
    """Eight parallel sub-agents on one prefix: one writes, seven read."""
    store = MemoryStore()
    pilot = CachePilot(store, enabled=True, wait_ms=2000)
    fingerprint = "fp-fanout"

    async def worker(index: int):
        role = await pilot.acquire(fingerprint, 300)
        if role is PilotRole.PILOT:
            await asyncio.sleep(0.05)  # stand-in for the upstream call
            await pilot.mark_warm(fingerprint, 300)
        return role

    roles = await asyncio.gather(*(worker(i) for i in range(8)))

    assert roles.count(PilotRole.PILOT) == 1
    assert PilotRole.TIMEOUT not in roles
    assert all(r in (PilotRole.PILOT, PilotRole.FOLLOWER, PilotRole.WARM) for r in roles)


async def test_pilot_failure_frees_the_lock():
    store = MemoryStore()
    pilot = CachePilot(store, enabled=True, wait_ms=200)

    assert await pilot.acquire("fp-fail", 300) is PilotRole.PILOT
    await pilot.release_failed("fp-fail")
    # A follower can now become the pilot rather than waiting out the TTL.
    assert await pilot.acquire("fp-fail", 300) is PilotRole.PILOT


async def test_pilot_can_be_disabled():
    pilot = CachePilot(MemoryStore(), enabled=False)
    assert await pilot.acquire("fp", 300) is PilotRole.DISABLED
