"""Cost pricing and the per-tenant spend ledger.

Pricing is done from *actual* usage, not the routing estimate. The two are
recorded separately so you can measure how wrong the estimator is — an
estimator nobody checks is how budget enforcement quietly stops working.

Cached input is billed at a different rate from fresh input, in both
directions: reads are ~0.1x, and on Anthropic a *write* costs 1.25x (5m TTL) or
2x (1h). A ledger that prices all prompt tokens at the list rate will overstate
a cache-heavy workload and understate the cost of thrashing between models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..catalog import ModelSpec
from ..schemas import Usage


@dataclass
class PricedUsage:
    input_fresh_usd: float
    input_cache_read_usd: float
    input_cache_write_usd: float
    output_usd: float
    # Set by price_usage from the *model's* read multiplier. A hardcoded
    # "read is 0.1x" here overstated savings for any model whose discount
    # differs — the multiplier is catalog data, not a constant.
    cache_savings_usd: float = 0.0

    @property
    def total_usd(self) -> float:
        return (
            self.input_fresh_usd
            + self.input_cache_read_usd
            + self.input_cache_write_usd
            + self.output_usd
        )


def price_usage(usage: Usage, model: ModelSpec, cache_ttl: str = "5m") -> PricedUsage:
    # Same tiered rates the router scored against — if the ledger used the
    # headline rate while the router used the long-context one, spend and
    # estimate would diverge for exactly the largest requests.
    rate_in, rate_out = model.rates_for(usage.prompt_tokens)
    price_in = rate_in / 1_000_000
    price_out = rate_out / 1_000_000

    fresh = max(0, usage.prompt_tokens - usage.cache_read_tokens - usage.cache_write_tokens)
    return PricedUsage(
        input_fresh_usd=fresh * price_in,
        input_cache_read_usd=usage.cache_read_tokens * price_in * model.cache_read_multiplier,
        input_cache_write_usd=usage.cache_write_tokens
        * price_in
        * model.cache_write_multiplier(cache_ttl),
        output_usd=usage.completion_tokens * price_out,
        # What those tokens would have cost fresh, minus what they cost cached.
        cache_savings_usd=usage.cache_read_tokens
        * price_in
        * (1.0 - model.cache_read_multiplier),
    )


class CostLedger:
    """Rolling per-tenant spend, plus attribution keys for chargeback."""

    def __init__(self, store):
        self._store = store

    @staticmethod
    def _day() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _tenant_key(self, tenant: str) -> str:
        return f"spend:{tenant}:{self._day()}"

    def _attr_key(self, tenant: str, agent: str, model: str) -> str:
        return f"spend:{tenant}:{self._day()}:{agent}:{model}"

    async def record(
        self,
        tenant: str,
        agent: str,
        model_key: str,
        priced: PricedUsage,
        reserved_usd: float = 0.0,
    ) -> float:
        """Record actual spend; ``reserved_usd`` is what check() already took.

        With a reservation, only the estimate-vs-actual delta lands here, so
        the tenant counter is never double-charged. Attribution always gets the
        full actual figure — chargeback wants what happened, not the mechanics
        of how enforcement counted it.
        """
        # 48h TTL: long enough to survive a UTC day boundary plus inspection.
        total = await self._store.incr_float(
            self._tenant_key(tenant), priced.total_usd - reserved_usd, ttl=172_800
        )
        await self._store.incr_float(
            self._attr_key(tenant, agent, model_key), priced.total_usd, ttl=172_800
        )
        return total

    async def reserve(self, tenant: str, amount_usd: float) -> float:
        """Atomically add a pending estimate to today's spend; returns the total."""
        return await self._store.incr_float(
            self._tenant_key(tenant), amount_usd, ttl=172_800
        )

    async def release(self, tenant: str, amount_usd: float) -> None:
        """Give a reservation back — the request was rejected or never priced."""
        if amount_usd:
            await self._store.incr_float(
                self._tenant_key(tenant), -amount_usd, ttl=172_800
            )

    async def spend_today(self, tenant: str) -> float:
        return await self._store.get_float(self._tenant_key(tenant))
