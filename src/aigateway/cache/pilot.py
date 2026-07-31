"""Cache pilot: the fan-out fix.

A cache entry only becomes readable once the first response *begins streaming*.
So when a multi-agent orchestrator fires N sub-agents in parallel against the
same system prompt and tool set, all N miss — every one of them pays a cold
write. With N=8 and a 20k-token prefix that is 8 full-price writes instead of
1 write and 7 reads at 0.1x.

The fix is small: the first caller to see an unseen prefix becomes the *pilot*
and proceeds immediately. Everyone else on that prefix becomes a *follower* and
waits, briefly, for the pilot to mark the prefix warm — then proceeds and reads
the cache the pilot just wrote.

Followers never block indefinitely. On timeout they proceed anyway and simply
pay what they would have paid without this module, so a stuck pilot degrades
to the status quo rather than to an outage.
"""

from __future__ import annotations

import asyncio
import enum
import logging

log = logging.getLogger(__name__)


class PilotRole(enum.StrEnum):
    PILOT = "pilot"  # first to see this prefix; writes the cache
    FOLLOWER = "follower"  # waited for the pilot, reads the cache
    WARM = "warm"  # prefix already warm; no waiting needed
    TIMEOUT = "timeout"  # pilot never landed; proceeding cold
    DISABLED = "disabled"


class CachePilot:
    def __init__(self, store, enabled: bool = True, wait_ms: int = 4000):
        self._store = store
        self._enabled = enabled
        self._wait_ms = wait_ms

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> bool:
        """Toggle at runtime; returns the previous value.

        Exists so the demo console can run the same fan-out with and without
        the pilot and show the difference in cache writes.
        """
        previous = self._enabled
        self._enabled = value
        return previous

    @staticmethod
    def _warm_key(fp: str) -> str:
        return f"cache:warm:{fp}"

    @staticmethod
    def _lock_key(fp: str) -> str:
        return f"cache:pilot:{fp}"

    async def acquire(self, fingerprint: str, ttl_seconds: int) -> PilotRole:
        if not self._enabled or not fingerprint:
            return PilotRole.DISABLED

        if await self._store.get(self._warm_key(fingerprint)):
            return PilotRole.WARM

        # Lock TTL is a safety valve: if the pilot dies mid-flight, followers
        # get a shot at becoming the pilot rather than waiting forever.
        lock_ttl = max(2, self._wait_ms // 1000 + 2)
        if await self._store.try_lock(self._lock_key(fingerprint), lock_ttl):
            return PilotRole.PILOT

        deadline = asyncio.get_running_loop().time() + self._wait_ms / 1000
        delay = 0.05
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(delay)
            if await self._store.get(self._warm_key(fingerprint)):
                return PilotRole.FOLLOWER
            delay = min(delay * 1.6, 0.4)

        log.info("cache pilot timeout for %s; proceeding cold", fingerprint[:12])
        return PilotRole.TIMEOUT

    async def mark_warm(self, fingerprint: str, ttl_seconds: int) -> None:
        """Release the followers.

        Ideally this fires when the upstream response *starts* — the provider
        cache is readable from the first token, so followers need not wait for
        the full body. On the unary path there is no first-token hook, so it is
        called on completion instead; the streaming path is where this becomes
        worth tightening, and is the obvious next optimisation here.
        """
        if not self._enabled or not fingerprint:
            return
        await self._store.set(self._warm_key(fingerprint), "1", ttl=ttl_seconds)
        await self._store.unlock(self._lock_key(fingerprint))

    async def release_failed(self, fingerprint: str) -> None:
        """Pilot failed — free the lock so a follower can take over."""
        if not self._enabled or not fingerprint:
            return
        await self._store.unlock(self._lock_key(fingerprint))
