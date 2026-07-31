"""Shared-state interface.

Budgets, rate limits, session stickiness, and the cache-pilot lock all need
state. Behind one interface so a laptop run needs no infrastructure and a
scaled deployment stays correct: two workers with in-memory budget counters
do not have half a budget each, they have two full budgets.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class StateStore(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl: int | None = None) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def incr_float(self, key: str, amount: float, ttl: int | None = None) -> float:
        """Atomically add to a float counter and return the new value."""
        ...

    async def get_float(self, key: str) -> float: ...

    async def incr_window(self, key: str, window_seconds: int) -> int:
        """Increment a fixed-window counter, returning the count in this window."""
        ...

    async def try_lock(self, key: str, ttl_seconds: int) -> bool:
        """Acquire an exclusive lock. False if already held."""
        ...

    async def unlock(self, key: str) -> None: ...

    async def close(self) -> None: ...
