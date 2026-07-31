"""Per-tenant budget enforcement.

"Token availability for the enterprise" is four separate mechanisms that get
conflated. This module implements two of them and is explicit about the rest:

* **budget** (here)      — dollars per tenant per day, soft-degrade then fail
* **cost attribution**   — see ``ledger.CostLedger``, keyed tenant/agent/model
* **provider rate limits** — see ``ratelimit.RateLimiter``; note that model
  tiers sit in *separate upstream pools*, so shedding from a frontier model to
  a small one is a genuine capacity lever, not only a cost lever
* **quota fairness across teams** — deliberately not implemented; it needs a
  scheduler, not a counter, and it is the wrong thing to prototype first

Enforcement is soft by default: over-budget tenants are pushed down to the
cheapest capable tier and told so in the response, rather than being cut off
mid-workflow. Degradation is visible in ``x_gateway.degraded`` precisely so an
agent can react instead of silently getting worse answers.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..catalog import Tier
from ..config import Settings
from ..errors import BudgetExceeded
from .ledger import CostLedger


@dataclass
class BudgetVerdict:
    allowed: bool
    tier_ceiling: Tier | None  # None = no ceiling imposed
    spend_usd: float
    limit_usd: float
    utilisation: float
    needs_exact_preflight: bool
    message: str = ""


class BudgetGuard:
    def __init__(self, settings: Settings, store, ledger: CostLedger):
        self._s = settings
        self._store = store
        self._ledger = ledger

    async def limit_for(self, tenant: str) -> float:
        override = await self._store.get(f"budget:{tenant}:daily_usd")
        return float(override) if override else self._s.default_tenant_daily_usd

    async def set_limit(self, tenant: str, usd: float) -> None:
        await self._store.set(f"budget:{tenant}:daily_usd", str(usd))

    async def check(self, tenant: str, estimated_usd: float) -> BudgetVerdict:
        spend = await self._ledger.spend_today(tenant)
        limit = await self.limit_for(tenant)
        projected = spend + estimated_usd
        utilisation = projected / limit if limit > 0 else 0.0

        # Only pay for an exact token count when the answer might actually
        # change the outcome. An extra round trip on every request is not free.
        needs_exact = utilisation >= self._s.preflight_exact_threshold

        if projected <= limit:
            return BudgetVerdict(True, None, spend, limit, utilisation, needs_exact)

        if self._s.budget_mode == "hard":
            raise BudgetExceeded(
                f"tenant '{tenant}' daily budget exhausted "
                f"(${spend:.4f} spent of ${limit:.2f}); request rejected"
            )

        # Soft mode: degrade before failing. If even the cheapest tier cannot
        # fit, there is nothing left to degrade to.
        if spend >= limit:
            raise BudgetExceeded(
                f"tenant '{tenant}' daily budget fully exhausted "
                f"(${spend:.4f} of ${limit:.2f}); degradation exhausted"
            )

        return BudgetVerdict(
            allowed=True,
            tier_ceiling=Tier.LIGHT,
            spend_usd=spend,
            limit_usd=limit,
            utilisation=utilisation,
            needs_exact_preflight=needs_exact,
            message=(
                f"over projected budget (${projected:.4f} of ${limit:.2f}); "
                f"degraded to the cheapest capable tier"
            ),
        )
