"""Rate limiting: inbound per-tenant, and outbound per upstream pool.

The second half is the one people forget. Upstream models sit in **separate
rate-limit pools** — a frontier model does not draw from its predecessor's
bucket — so tracking headroom per pool lets the router treat "this pool is
saturated" as a routing input rather than as a 429 to retry blindly.
"""

from __future__ import annotations

from ..config import Settings
from ..errors import RateLimited


class RateLimiter:
    def __init__(self, settings: Settings, store):
        self._s = settings
        self._store = store

    async def limit_for(self, tenant: str) -> int:
        override = await self._store.get(f"rpm:{tenant}:limit")
        return int(override) if override else self._s.default_tenant_rpm

    async def check_tenant(self, tenant: str) -> int:
        """Fixed-window counter. Cheap and adequate; swap for a sliding window
        or token bucket if burst shaping matters to you."""
        limit = await self.limit_for(tenant)
        count = await self._store.incr_window(f"rpm:{tenant}", 60)
        if count > limit:
            raise RateLimited(
                f"tenant '{tenant}' exceeded {limit} requests/min", retry_after=60
            )
        return count

    async def note_upstream(self, pool: str) -> int:
        return await self._store.incr_window(f"pool:{pool}", 60)

    async def pool_pressure(self, pool: str) -> int:
        """Requests observed against an upstream pool in the current window.

        The router reads this when ``pool_rpm_limits`` bounds the pool: at the
        ceiling the pool's models leave the candidate set, and past 80% their
        score is penalised so load sheds before the 429s start. The honest
        limit still lives on the provider side — the configured ceiling is
        operator knowledge of the account tier, not discovery.
        """
        bucket_key = f"pool:{pool}"
        import time

        bucket = int(time.time() // 60)
        raw = await self._store.get(f"{bucket_key}:{bucket}")
        return int(raw or 0)
