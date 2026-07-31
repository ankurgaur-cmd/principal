"""Model pool health tracking and circuit breaking.

Two independent signals feed a model's health, and conflating them is a common
mistake:

* **Active probes** — a periodic, free call to each provider's models endpoint.
  Cheap, but only tells you the provider is reachable and the key is valid.
* **Passive observation** — the outcome of real traffic. This is the signal that
  actually matters, because a provider can be reachable while one model is
  overloaded.

Passive evidence outranks probe evidence: a model that just failed three real
requests is unhealthy no matter how cheerfully the models endpoint answers.

The circuit breaker exists so a failing model stops absorbing traffic *and*
stops absorbing latency. Without it every request pays the full timeout before
falling back. States are the standard three:

    CLOSED ──(N consecutive failures)──▶ OPEN
      ▲                                   │
      │                            (cooldown elapses)
      │                                   ▼
      └────(trial succeeds)──────── HALF_OPEN ──(trial fails)──▶ OPEN

HALF_OPEN admits exactly one trial request. The router treats OPEN models as
uncandidates, so breaking a model is a routing decision, not an error path.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field

from ..catalog import CATALOG

log = logging.getLogger(__name__)


class Breaker(enum.StrEnum):
    CLOSED = "closed"  # normal
    OPEN = "open"      # failing; excluded from routing
    HALF_OPEN = "half_open"  # cooldown elapsed; one trial allowed


class Status(enum.StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"        # succeeding, but slow or intermittently failing
    UNHEALTHY = "unhealthy"      # breaker open
    UNCONFIGURED = "unconfigured"  # no credentials for this provider
    UNKNOWN = "unknown"          # configured, never exercised


@dataclass
class ModelHealth:
    model_key: str
    provider: str
    breaker: Breaker = Breaker.CLOSED
    consecutive_failures: int = 0
    total_ok: int = 0
    total_failed: int = 0
    last_ok_at: float | None = None
    last_error: str | None = None
    last_error_at: float | None = None
    opened_at: float | None = None
    probe_ok: bool | None = None
    probe_at: float | None = None
    latencies_ms: list[int] = field(default_factory=list)

    def note_latency(self, ms: int) -> None:
        self.latencies_ms.append(ms)
        if len(self.latencies_ms) > 50:  # rolling window
            self.latencies_ms.pop(0)

    @property
    def p50_ms(self) -> int | None:
        if not self.latencies_ms:
            return None
        ordered = sorted(self.latencies_ms)
        return ordered[len(ordered) // 2]

    @property
    def success_rate(self) -> float | None:
        total = self.total_ok + self.total_failed
        return round(self.total_ok / total, 3) if total else None


class HealthMonitor:
    def __init__(self, registry, *, failure_threshold: int = 3, cooldown_seconds: int = 60):
        self._registry = registry
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._health: dict[str, ModelHealth] = {
            spec.key: ModelHealth(model_key=spec.key, provider=spec.provider)
            for spec in CATALOG.values()
        }
        self._task: asyncio.Task | None = None

    # -- passive signal (real traffic) --------------------------------------
    def record_success(self, model_key: str, latency_ms: int) -> None:
        h = self._health.get(model_key)
        if h is None:
            return
        h.total_ok += 1
        h.consecutive_failures = 0
        h.last_ok_at = time.time()
        h.note_latency(latency_ms)
        if h.breaker is not Breaker.CLOSED:
            log.info("circuit closed for %s after a successful trial", model_key)
            h.breaker = Breaker.CLOSED
            h.opened_at = None

    def record_failure(self, model_key: str, error: str) -> None:
        h = self._health.get(model_key)
        if h is None:
            return
        h.total_failed += 1
        h.consecutive_failures += 1
        h.last_error = error[:200]
        h.last_error_at = time.time()
        if h.consecutive_failures >= self._threshold and h.breaker is not Breaker.OPEN:
            h.breaker = Breaker.OPEN
            h.opened_at = time.time()
            log.warning(
                "circuit opened for %s after %d consecutive failures: %s",
                model_key, h.consecutive_failures, h.last_error,
            )

    # -- breaker gate -------------------------------------------------------
    def is_available(self, model_key: str) -> bool:
        """False only when the breaker is open and still cooling down."""
        h = self._health.get(model_key)
        if h is None:
            return True
        if h.breaker is Breaker.OPEN:
            if h.opened_at and time.time() - h.opened_at >= self._cooldown:
                h.breaker = Breaker.HALF_OPEN
                log.info("circuit half-open for %s; admitting one trial", model_key)
                return True
            return False
        return True

    def status_of(self, model_key: str) -> Status:
        h = self._health[model_key]
        if h.provider not in self._registry.enabled:
            return Status.UNCONFIGURED
        if h.breaker is Breaker.OPEN:
            return Status.UNHEALTHY
        if h.consecutive_failures or h.breaker is Breaker.HALF_OPEN:
            return Status.DEGRADED
        if h.total_ok or h.probe_ok:
            return Status.HEALTHY
        return Status.UNKNOWN

    # -- active probe -------------------------------------------------------
    async def probe(self) -> None:
        """One probe per configured provider, not per model.

        The models endpoint is provider-scoped, so probing per model would be
        N identical calls. The result is fanned out to that provider's models.
        """
        for provider_name in list(self._registry.enabled):
            try:
                provider = self._registry.get(provider_name)
                started = time.perf_counter()
                ok, detail = await provider.validate()
                elapsed = int((time.perf_counter() - started) * 1000)
            except Exception as exc:  # never let a probe kill the loop
                ok, detail, elapsed = False, type(exc).__name__, 0

            for h in self._health.values():
                if h.provider != provider_name:
                    continue
                h.probe_ok = ok
                h.probe_at = time.time()
                if not ok:
                    h.last_error = detail[:200]
                    h.last_error_at = time.time()

            log.debug("probe %s: %s (%d ms)", provider_name, "ok" if ok else detail, elapsed)

    def start(self, interval_seconds: int) -> None:
        if interval_seconds <= 0 or self._task is not None:
            return

        async def loop():
            while True:
                try:
                    await self.probe()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pragma: no cover - defensive
                    log.warning("health probe loop error: %s", exc)
                await asyncio.sleep(interval_seconds)

        self._task = asyncio.create_task(loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    # -- reporting ----------------------------------------------------------
    def snapshot(self) -> list[dict]:
        out = []
        for spec in CATALOG.values():
            h = self._health[spec.key]
            out.append(
                {
                    "model": spec.key,
                    "provider": spec.provider,
                    "tier": spec.tier.name.lower(),
                    "status": self.status_of(spec.key).value,
                    "breaker": h.breaker.value,
                    "consecutive_failures": h.consecutive_failures,
                    "total_ok": h.total_ok,
                    "total_failed": h.total_failed,
                    "success_rate": h.success_rate,
                    "p50_latency_ms": h.p50_ms,
                    "last_error": h.last_error,
                    "probe_ok": h.probe_ok,
                    "price_in_per_mtok": spec.price_in_per_mtok,
                    "price_out_per_mtok": spec.price_out_per_mtok,
                    "context_window": spec.context_window,
                    "min_cacheable_tokens": spec.min_cacheable_tokens,
                    "rate_limit_pool": spec.rate_limit_pool,
                }
            )
        return out

    def reset(self, model_key: str | None = None) -> None:
        """Manually close a breaker. Useful in a demo, and after a known fix."""
        keys = [model_key] if model_key else list(self._health)
        for key in keys:
            if h := self._health.get(key):
                h.breaker = Breaker.CLOSED
                h.consecutive_failures = 0
                h.opened_at = None
