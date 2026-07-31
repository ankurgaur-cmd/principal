"""Model pool health and circuit breaking."""

from __future__ import annotations

import time

import pytest

from aigateway.providers.health import Breaker, HealthMonitor, Status


class FakeRegistry:
    def __init__(self, enabled: set[str], provider=None):
        self.enabled = enabled
        self._provider = provider

    def get(self, name):
        return self._provider


class ProbeProvider:
    def __init__(self, ok=True, detail="authenticated"):
        self.ok, self.detail, self.calls = ok, detail, 0

    async def validate(self):
        self.calls += 1
        return self.ok, self.detail


@pytest.fixture
def monitor():
    return HealthMonitor(
        FakeRegistry({"anthropic", "openai"}), failure_threshold=3, cooldown_seconds=60
    )


def test_unexercised_model_is_unknown_not_healthy(monitor):
    """Absence of failure is not evidence of health."""
    assert monitor.status_of("claude-opus-5") is Status.UNKNOWN


def test_unconfigured_provider_reports_unconfigured(monitor):
    m = HealthMonitor(FakeRegistry(set()))
    assert m.status_of("claude-opus-5") is Status.UNCONFIGURED


def test_breaker_opens_after_the_threshold(monitor):
    for _ in range(2):
        monitor.record_failure("claude-opus-5", "HTTP 500")
    assert monitor.is_available("claude-opus-5") is True
    assert monitor.status_of("claude-opus-5") is Status.DEGRADED

    monitor.record_failure("claude-opus-5", "HTTP 500")
    assert monitor.is_available("claude-opus-5") is False
    assert monitor.status_of("claude-opus-5") is Status.UNHEALTHY


def test_success_resets_the_failure_streak(monitor):
    monitor.record_failure("claude-opus-5", "HTTP 500")
    monitor.record_failure("claude-opus-5", "HTTP 500")
    monitor.record_success("claude-opus-5", 120)
    monitor.record_failure("claude-opus-5", "HTTP 500")

    # Two, then a success, then one more must not add up to three in a row.
    assert monitor.is_available("claude-opus-5") is True


def test_cooldown_admits_one_trial_then_closes_on_success(monitor):
    for _ in range(3):
        monitor.record_failure("claude-opus-5", "HTTP 529")
    assert monitor.is_available("claude-opus-5") is False

    monitor._health["claude-opus-5"].opened_at = time.time() - 61  # cooldown elapsed
    assert monitor.is_available("claude-opus-5") is True
    assert monitor._health["claude-opus-5"].breaker is Breaker.HALF_OPEN

    monitor.record_success("claude-opus-5", 90)
    assert monitor._health["claude-opus-5"].breaker is Breaker.CLOSED
    assert monitor.status_of("claude-opus-5") is Status.HEALTHY


def test_failure_during_trial_reopens(monitor):
    for _ in range(3):
        monitor.record_failure("claude-opus-5", "HTTP 500")
    monitor._health["claude-opus-5"].opened_at = time.time() - 61
    monitor.is_available("claude-opus-5")  # -> half open

    monitor.record_failure("claude-opus-5", "HTTP 500")
    assert monitor._health["claude-opus-5"].breaker is Breaker.OPEN
    assert monitor.is_available("claude-opus-5") is False


def test_latency_is_a_rolling_median(monitor):
    for ms in [100, 200, 300, 400, 500]:
        monitor.record_success("claude-sonnet-5", ms)
    assert monitor._health["claude-sonnet-5"].p50_ms == 300
    assert monitor._health["claude-sonnet-5"].success_rate == 1.0


async def test_probe_is_per_provider_not_per_model(monitor):
    """The models endpoint is provider-scoped; probing per model would be N
    identical calls."""
    provider = ProbeProvider(ok=True)
    monitor._registry = FakeRegistry({"anthropic"}, provider)

    await monitor.probe()

    assert provider.calls == 1, "one probe should cover all of a provider's models"
    anthropic_models = [
        m for m in monitor.snapshot() if m["provider"] == "anthropic"
    ]
    assert len(anthropic_models) == 3
    assert all(m["probe_ok"] for m in anthropic_models)


async def test_a_failing_probe_does_not_open_the_breaker(monitor):
    """Probe evidence is weaker than real traffic. A provider-wide blip should
    not take every model out of rotation on its own."""
    monitor._registry = FakeRegistry({"anthropic"}, ProbeProvider(ok=False, detail="401"))
    await monitor.probe()

    assert monitor.is_available("claude-opus-5") is True
    assert monitor._health["claude-opus-5"].breaker is Breaker.CLOSED


def test_manual_reset_closes_breakers(monitor):
    for _ in range(3):
        monitor.record_failure("claude-opus-5", "HTTP 500")
    assert monitor.is_available("claude-opus-5") is False

    monitor.reset("claude-opus-5")
    assert monitor.is_available("claude-opus-5") is True


def test_router_excludes_a_broken_model(settings, store, monitor):
    """Health is a routing input: an open breaker removes the model from the
    candidate set rather than being discovered as an error."""
    from conftest import make_request

    from aigateway.routing import Router

    router = Router(settings, store, {"anthropic"}, health=monitor)

    import asyncio

    healthy = asyncio.run(router.route(make_request(), "architecture"))
    assert healthy.model.key == "claude-opus-5"

    for _ in range(3):
        monitor.record_failure("claude-opus-5", "HTTP 529")

    with pytest.raises(Exception) as exc:
        asyncio.run(router.route(make_request(), "architecture"))
    assert "circuit open" in str(exc.value.detail)
