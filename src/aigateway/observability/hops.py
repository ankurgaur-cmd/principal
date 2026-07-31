"""Per-transaction hop trace: origination, every upstream call, every outcome.

A routing decision tells you *what the gateway chose*. This tells you *what
actually happened on the wire* — which is a different question and the one you
need when something is slow, expensive, or wrong.

Every outbound call gets a hop: the classifier's small-model call, each model
attempt including failed ones that triggered a fallback, and any wait the cache
pilot imposed. Failed attempts are the important part — a decision log that only
records the attempt that succeeded hides the two that timed out first, along
with the latency they cost.

Hop 0 is always the origination: who asked, from where, under which trace id.
Without it a trace is a list of calls with no subject.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass


@dataclass
class Hop:
    seq: int
    kind: str  # origin | classifier | model | cache_wait | probe
    label: str
    host: str = ""  # the server actually contacted
    endpoint: str = ""
    model: str = ""
    provider: str = ""
    attempt: int = 1
    status: str = "ok"  # ok | error | refused | skipped | timeout
    http_status: int | None = None
    latency_ms: int = 0
    started_ms: int = 0  # offset from the start of the transaction
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cached_tokens: int = 0
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class TraceContext:
    """Collects hops for one transaction.

    Deliberately not a context manager and deliberately not global: it is
    created per request in the pipeline and passed down explicitly, so there is
    no ambient state to leak between concurrent requests.
    """

    def __init__(self, trace_id: str, origin: dict | None = None):
        self.trace_id = trace_id
        self._t0 = time.perf_counter()
        self.hops: list[Hop] = []
        if origin is not None:
            self.add(
                kind="origin",
                label="client → gateway",
                host=origin.get("client", "local"),
                endpoint=origin.get("endpoint", ""),
                detail=(
                    f"tenant={origin.get('tenant', '?')} "
                    f"agent={origin.get('agent', '?')} "
                    f"session={origin.get('session') or '—'}"
                ),
            )

    def _elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)

    def add(self, **kwargs) -> Hop:
        hop = Hop(seq=len(self.hops), started_ms=self._elapsed_ms(), **kwargs)
        self.hops.append(hop)
        return hop

    def timed(self, **kwargs):
        """Context manager that records a hop and times what happens inside it.

        On an exception it marks the hop failed and re-raises — a hop that
        vanishes because the call it describes threw is exactly the hop you
        needed to see.
        """
        return _TimedHop(self, kwargs)

    @property
    def total_ms(self) -> int:
        return self._elapsed_ms()

    @property
    def upstream_ms(self) -> int:
        """Time spent waiting on other people's servers."""
        return sum(h.latency_ms for h in self.hops if h.kind in ("classifier", "model"))

    def summary(self) -> dict:
        hosts = [h.host for h in self.hops if h.host and h.kind != "origin"]
        return {
            "trace_id": self.trace_id,
            "hops": [h.to_dict() for h in self.hops],
            "hop_count": len(self.hops),
            "hosts_contacted": sorted(set(hosts)),
            "total_ms": self.total_ms,
            "upstream_ms": self.upstream_ms,
            # What the gateway itself cost you, as opposed to the models.
            "gateway_overhead_ms": max(0, self.total_ms - self.upstream_ms),
            "failed_hops": sum(1 for h in self.hops if h.status not in ("ok", "skipped")),
        }


class _TimedHop:
    def __init__(self, trace: TraceContext, kwargs: dict):
        self._trace = trace
        self._kwargs = kwargs
        self._hop: Hop | None = None
        self._started = 0.0

    def __enter__(self) -> Hop:
        self._started = time.perf_counter()
        self._hop = self._trace.add(**self._kwargs)
        return self._hop

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._hop is not None:
            self._hop.latency_ms = int((time.perf_counter() - self._started) * 1000)
            if exc_type is not None and self._hop.status == "ok":
                self._hop.status = "error"
                self._hop.detail = f"{exc_type.__name__}: {exc}"[:200]
                self._hop.http_status = getattr(exc, "status_code", None)
        return False  # never swallow
