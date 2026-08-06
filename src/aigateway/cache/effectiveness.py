"""Did the cache we priced actually turn up?

The router prices every candidate on the assumption that a prefix at or above
``min_cacheable_tokens`` will be cached by the vendor. That assumption is what
makes cache-aware routing work — and it is an assumption, not a measurement.

Measured on this gateway, at an identical 1,095-token prompt:

    gpt-5-mini        1,024 tokens cached      the assumption holds
    claude-sonnet-5   1,650 tokens cached      holds
    gpt-5-nano                0 cached         does not hold
    gpt-5.4-mini              0 cached         does not hold

Two of four never returned a cached token. The router had already discounted
their input cost by 90% on the strength of a warm read that was never going to
arrive — and because the router picks the *cheapest* candidate, a model that
fails to cache is systematically favoured by the discount it does not earn. The
error compounds in the worst direction.

So the assumption is checked against reality. Each (model, cache-state) pair
records whether the vendor actually returned cached tokens when the router
expected them to, and a model with a settled record of not delivering stops
being credited for it. This does not stop the gateway *asking* for a cache — it
costs nothing to ask, and vendors change. It stops the router *pricing* one.

The same three rules as every other feedback loop here:

* **No evidence, no adjustment.** Below ``MIN_SAMPLES`` every model is trusted,
  because the alternative is routing on the order requests happened to arrive.
* **Only the direction that is provably wrong.** A model that delivers is never
  penalised; only one that was priced for a hit and returned a miss.
* **Recoverable.** The window rolls, so a vendor that ships prefix caching next
  month climbs back out on its own.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque

log = logging.getLogger(__name__)

# Observations of one model before its record is allowed to change routing.
MIN_SAMPLES = 4

# Rolling window. Short enough to notice a vendor shipping caching, long enough
# that one flushed cache does not condemn a model.
WINDOW = 40

# At or below this hit rate a model is treated as not caching at all. Not zero:
# a vendor legitimately evicts entries under load, and one miss in ten is a
# cache working normally, not a broken promise.
DEAD_CACHE_RATE = 0.15


class CacheEffectiveness:
    """Observed cache delivery per model.

    In-memory and rolling, like reputation and the fleet view. The durable
    record is the JSONL — ``cache_read_tokens`` is on every line, so this can be
    rebuilt offline by the replay harness.
    """

    def __init__(self, *, window: int = WINDOW, min_samples: int = MIN_SAMPLES):
        self._window = window
        self._min_samples = min_samples
        # model_key -> recent outcomes, True = the vendor returned cached tokens
        self._delivered: dict[str, deque[bool]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def record(self, model_key: str, *, expected_hit: bool, cached_tokens: int) -> None:
        """Note one outcome.

        Only requests where the router *expected* a hit are evidence. A cold
        write returning nothing cached is correct behaviour, not a failure, and
        counting it would condemn every model on its first request.
        """
        if not expected_hit:
            return
        delivered = cached_tokens > 0
        self._delivered[model_key].append(delivered)
        if not delivered:
            log.info(
                "cache miss on %s: router priced a warm read, vendor returned "
                "0 cached tokens",
                model_key,
            )

    def hit_rate(self, model_key: str) -> float | None:
        """Observed delivery rate, or None without enough evidence to say."""
        seen = self._delivered.get(model_key)
        if not seen or len(seen) < self._min_samples:
            return None
        return sum(seen) / len(seen)

    def delivers(self, model_key: str) -> bool:
        """Whether to keep pricing this model as if it caches.

        Trusting by default is deliberate: an unproven model gets the benefit of
        the doubt, and only a settled record of not delivering takes it away.
        """
        rate = self.hit_rate(model_key)
        return True if rate is None else rate > DEAD_CACHE_RATE

    def snapshot(self) -> list[dict]:
        rows = []
        for model_key, seen in sorted(self._delivered.items()):
            rate = self.hit_rate(model_key)
            rows.append(
                {
                    "model": model_key,
                    "samples": len(seen),
                    "misses": sum(1 for d in seen if not d),
                    "hit_rate": round(rate, 3) if rate is not None else None,
                    "trusted": self.delivers(model_key),
                    "status": (
                        "learning"
                        if rate is None
                        else "delivering"
                        if rate > DEAD_CACHE_RATE
                        else "priced as uncached — never returns cached tokens"
                    ),
                    "needs": max(0, self._min_samples - len(seen)),
                }
            )
        return rows

    def reset(self) -> None:
        self._delivered.clear()
