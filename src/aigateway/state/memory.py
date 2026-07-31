"""In-process store. Correct for a single worker only."""

from __future__ import annotations

import asyncio
import time


class MemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, tuple[str, float | None]] = {}
        self._lock = asyncio.Lock()

    def _live(self, key: str) -> str | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        value, expires = entry
        if expires is not None and expires < time.monotonic():
            self._data.pop(key, None)
            return None
        return value

    async def get(self, key: str) -> str | None:
        async with self._lock:
            return self._live(key)

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        async with self._lock:
            expires = time.monotonic() + ttl if ttl else None
            self._data[key] = (value, expires)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)

    async def incr_float(self, key: str, amount: float, ttl: int | None = None) -> float:
        async with self._lock:
            current = float(self._live(key) or 0.0)
            new = current + amount
            existing = self._data.get(key)
            expires = existing[1] if existing and existing[1] else (
                time.monotonic() + ttl if ttl else None
            )
            self._data[key] = (str(new), expires)
            return new

    async def get_float(self, key: str) -> float:
        async with self._lock:
            return float(self._live(key) or 0.0)

    async def incr_window(self, key: str, window_seconds: int) -> int:
        bucket = int(time.time() // window_seconds)
        wkey = f"{key}:{bucket}"
        async with self._lock:
            current = int(self._live(wkey) or 0)
            current += 1
            self._data[wkey] = (str(current), time.monotonic() + window_seconds)
            return current

    async def try_lock(self, key: str, ttl_seconds: int) -> bool:
        async with self._lock:
            if self._live(key) is not None:
                return False
            self._data[key] = ("1", time.monotonic() + ttl_seconds)
            return True

    async def unlock(self, key: str) -> None:
        await self.delete(key)

    async def close(self) -> None:
        self._data.clear()
