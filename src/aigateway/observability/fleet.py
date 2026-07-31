"""Fleet-level view: where the enterprise's traffic actually goes.

A single hop trace answers "what happened to *this* request". This answers the
question you ask second and care about longer: across everything we send, which
vendors and servers are carrying the load, what are they costing us, and how are
they behaving.

It is a rolling in-memory aggregate rather than a query over the JSONL. That is
a deliberate trade: the dashboard needs to be live and cheap to poll, and
re-parsing a growing file on every poll is neither. The JSONL remains the
durable record — this is the hot view over it, and it resets on restart.

The **flow** breakdown is the part that tells a story: each distinct
`tenant → intent → tier → model → host` path with a count, so you can see that
(for example) one agent's summarisation traffic is what is actually driving your
frontier-model spend.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

# Rolling window. Big enough to be representative, small enough that percentile
# maths over it stays trivial.
WINDOW = 500


@dataclass
class Bucket:
    """Stats for one grouping key (a host, a model, a tenant, an intent)."""

    key: str
    requests: int = 0
    errors: int = 0
    refusals: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    latencies: deque[int] = field(default_factory=lambda: deque(maxlen=WINDOW))

    def _pct(self, p: float) -> int | None:
        if not self.latencies:
            return None
        ordered = sorted(self.latencies)
        idx = min(int(len(ordered) * p), len(ordered) - 1)
        return ordered[idx]

    def to_dict(self, total_requests: int) -> dict:
        return {
            "key": self.key,
            "requests": self.requests,
            "share": round(self.requests / total_requests, 4) if total_requests else 0.0,
            "errors": self.errors,
            "refusals": self.refusals,
            "error_rate": round(self.errors / self.requests, 4) if self.requests else 0.0,
            "p50_ms": self._pct(0.50),
            "p95_ms": self._pct(0.95),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cached_tokens": self.cached_tokens,
            "cache_hit_rate": (
                round(self.cached_tokens / self.tokens_in, 4) if self.tokens_in else 0.0
            ),
            "cost_usd": round(self.cost_usd, 6),
            "cost_per_request_usd": (
                round(self.cost_usd / self.requests, 6) if self.requests else 0.0
            ),
        }


class FleetStats:
    """Rolling aggregate over recent transactions."""

    def __init__(self) -> None:
        self.started_at = time.time()
        self.total_requests = 0
        self.by_host: dict[str, Bucket] = {}
        self.by_model: dict[str, Bucket] = {}
        self.by_provider: dict[str, Bucket] = {}
        self.by_tenant: dict[str, Bucket] = {}
        self.by_intent: dict[str, Bucket] = {}
        # tenant → intent → tier → model → host, with a count.
        self.flows: dict[tuple, int] = defaultdict(int)
        self.recent: deque[dict] = deque(maxlen=100)

    @staticmethod
    def _bucket(store: dict[str, Bucket], key: str) -> Bucket:
        if key not in store:
            store[key] = Bucket(key=key)
        return store[key]

    def record(self, rec) -> None:
        """Fold one completed request into the aggregate.

        Takes the RequestRecord rather than a response so failed and refused
        requests are counted too — an availability view that only sees successes
        is worse than none.
        """
        self.total_requests += 1
        ok = rec.outcome == "ok"
        latency = rec.latency_total_ms

        def fold(bucket: Bucket) -> None:
            bucket.requests += 1
            if rec.outcome == "error":
                bucket.errors += 1
            elif rec.outcome == "refusal":
                bucket.refusals += 1
            if ok:
                bucket.latencies.append(latency)
                bucket.tokens_in += rec.prompt_tokens
                bucket.tokens_out += rec.completion_tokens
                bucket.cached_tokens += rec.cache_read_tokens
                bucket.cost_usd += rec.actual_cost_usd

        for host in rec.hosts_contacted or ["(none)"]:
            fold(self._bucket(self.by_host, host))
        if rec.chosen_model:
            fold(self._bucket(self.by_model, rec.chosen_model))
        if rec.provider:
            fold(self._bucket(self.by_provider, rec.provider))
        fold(self._bucket(self.by_tenant, rec.tenant or "(unknown)"))
        fold(self._bucket(self.by_intent, rec.resolved_intent or "unknown"))

        host = (rec.hosts_contacted or ["(none)"])[0]
        self.flows[
            (
                rec.tenant or "?",
                rec.resolved_intent or "unknown",
                rec.tier or "?",
                rec.chosen_model or "(none)",
                host,
            )
        ] += 1

        self.recent.appendleft(
            {
                "trace_id": rec.trace_id,
                "at": rec.timestamp,
                "tenant": rec.tenant,
                "agent": rec.agent,
                "intent": rec.resolved_intent,
                "model": rec.chosen_model,
                "hosts": rec.hosts_contacted,
                "hops": rec.hop_count,
                "upstream_ms": rec.upstream_ms,
                "gateway_ms": rec.gateway_overhead_ms,
                "total_ms": rec.latency_total_ms,
                "cost_usd": round(rec.actual_cost_usd, 6),
                "cached_tokens": rec.cache_read_tokens,
                "outcome": rec.outcome,
            }
        )

    def warnings(self) -> list[dict]:
        """Conditions that make the numbers above misleading.

        The router selects on price, so traffic concentrating on models whose
        price is a placeholder is not a cosmetic issue — the routing itself is
        being decided by a number nobody has confirmed.
        """
        from ..catalog import get_model

        out: list[dict] = []
        if not self.total_requests:
            return out

        unverified = sum(
            b.requests
            for k, b in self.by_model.items()
            if (m := get_model(k)) is not None and not m.price_verified
        )
        if unverified:
            share = unverified / self.total_requests
            out.append(
                {
                    "level": "warning" if share < 0.5 else "serious",
                    "title": f"{share:.0%} of traffic is priced from placeholders",
                    "detail": (
                        "The router picks the cheapest capable model, so an "
                        "unverified price does not cause a small billing error — "
                        "it decides which vendor gets the traffic. Confirm these "
                        "rates before reading anything here as a vendor comparison."
                    ),
                }
            )

        providers = [b for b in self.by_provider.values() if b.requests]
        if len(providers) == 1 and len(self.by_provider) == 1 and self.total_requests >= 3:
            out.append(
                {
                    "level": "warning",
                    "title": f"All traffic is going to one vendor ({providers[0].key})",
                    "detail": (
                        "Either only one vendor is configured, or its catalog "
                        "prices undercut the other at every tier. Check the "
                        "Candidates panel on a single request to see which."
                    ),
                }
            )
        return out

    def snapshot(self, flow_limit: int = 25) -> dict:
        t = self.total_requests
        flows = sorted(self.flows.items(), key=lambda kv: kv[1], reverse=True)[:flow_limit]
        return {
            "warnings": self.warnings(),
            "window_started_at": self.started_at,
            "total_requests": t,
            "total_cost_usd": round(sum(b.cost_usd for b in self.by_model.values()), 6),
            "by_host": [b.to_dict(t) for b in _rank(self.by_host)],
            "by_provider": [b.to_dict(t) for b in _rank(self.by_provider)],
            "by_model": [b.to_dict(t) for b in _rank(self.by_model)],
            "by_tenant": [b.to_dict(t) for b in _rank(self.by_tenant)],
            "by_intent": [b.to_dict(t) for b in _rank(self.by_intent)],
            "flows": [
                {
                    "tenant": k[0],
                    "intent": k[1],
                    "tier": k[2],
                    "model": k[3],
                    "host": k[4],
                    "requests": n,
                    "share": round(n / t, 4) if t else 0.0,
                }
                for k, n in flows
            ],
            "recent": list(self.recent)[:25],
        }

    def reset(self) -> None:
        self.__init__()


def _rank(store: dict[str, Bucket]) -> list[Bucket]:
    return sorted(store.values(), key=lambda b: b.requests, reverse=True)
