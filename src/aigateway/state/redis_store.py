"""Redis-backed store, plus the factory that falls back to in-memory."""

from __future__ import annotations

import logging
import time

from .memory import MemoryStore

log = logging.getLogger(__name__)


class RedisStore:
    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis

        self._r = aioredis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> str | None:
        return await self._r.get(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        await self._r.set(key, value, ex=ttl)

    async def delete(self, key: str) -> None:
        await self._r.delete(key)

    async def incr_float(self, key: str, amount: float, ttl: int | None = None) -> float:
        pipe = self._r.pipeline()
        pipe.incrbyfloat(key, amount)
        if ttl:
            # NX so a rolling budget window isn't extended by every write.
            pipe.expire(key, ttl, nx=True)
        result = await pipe.execute()
        return float(result[0])

    async def get_float(self, key: str) -> float:
        return float(await self._r.get(key) or 0.0)

    async def incr_window(self, key: str, window_seconds: int) -> int:
        bucket = int(time.time() // window_seconds)
        wkey = f"{key}:{bucket}"
        pipe = self._r.pipeline()
        pipe.incr(wkey)
        pipe.expire(wkey, window_seconds, nx=True)
        result = await pipe.execute()
        return int(result[0])

    async def try_lock(self, key: str, ttl_seconds: int) -> bool:
        return bool(await self._r.set(key, "1", ex=ttl_seconds, nx=True))

    async def unlock(self, key: str) -> None:
        await self._r.delete(key)

    async def close(self) -> None:
        await self._r.aclose()


def build_store(redis_url: str | None):
    """Redis when configured, in-memory otherwise.

    The in-memory path is a development convenience. Running more than one
    worker without Redis silently multiplies every tenant's budget and rate
    limit by the worker count.
    """
    if not redis_url:
        log.warning(
            "REDIS_URL unset — using in-memory state. Budgets and rate limits "
            "are only correct with a single worker."
        )
        return MemoryStore()
    try:
        return RedisStore(redis_url)
    except Exception as exc:  # pragma: no cover - depends on local redis
        log.error("redis unavailable (%s); falling back to in-memory state", exc)
        return MemoryStore()
