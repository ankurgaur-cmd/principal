"""Cache pilot: the fan-out fix.

A cache entry only becomes readable once the first response *begins streaming*.
So when a multi-agent orchestrator fires N sub-agents in parallel against the
same system prompt and tool set, all N miss — every one of them pays a cold
write. With N=8 and a 20k-token prefix that is 8 full-price writes instead of
1 write and 7 reads at 0.1x.

The fix is small: the first caller to see an unseen prefix becomes the *pilot*
and proceeds immediately. Everyone else on that prefix becomes a *follower* and
waits for the pilot to mark the prefix warm — then proceeds and reads the cache
the pilot just wrote.

Three timing rules make this actually work against real upstream latencies
(tens of seconds, not the four the original defaults assumed):

* **The pilot holds its lock for as long as it is genuinely in flight.** The
  lock TTL is a dead-pilot detector, not a time budget — so the pilot
  heartbeats it while working (``holding``), and a crashed pilot's lock still
  expires in seconds. A fixed short TTL instead re-elected a second pilot
  mid-flight, which is precisely the duplicate write this module exists to
  prevent.
* **Followers wait scaled to the model's observed latency**, not to a constant.
  ``acquire`` takes a per-call ``wait_ms`` so the pipeline can pass p50-derived
  patience; a flat 4s against an 18-58s call meant every follower timed out and
  paid the cold write anyway — the pilot only ever helped stragglers.
* **Warmth is marked at the first token where the transport allows** (the
  streaming path), because the provider cache is readable from first token.
  The unary path still marks on completion — it has no earlier hook.

Followers never block indefinitely. On timeout they proceed anyway and simply
pay what they would have paid without this module, so a stuck pilot degrades
to the status quo rather than to an outage.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import logging

log = logging.getLogger(__name__)

# How long a pilot lock survives without a heartbeat. Short, deliberately: this
# only has to outlive a heartbeat interval, not an upstream call.
LOCK_TTL_SECONDS = 6


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

    async def acquire(
        self, fingerprint: str, ttl_seconds: int, wait_ms: int | None = None
    ) -> PilotRole:
        """Race for the pilot seat, or wait for whoever holds it.

        ``wait_ms`` overrides the configured follower patience. Callers who
        know the chosen model's observed latency should pass it — patience
        only pays off if it covers the pilot's actual time to first token.
        """
        if not self._enabled or not fingerprint:
            return PilotRole.DISABLED

        if await self._store.get(self._warm_key(fingerprint)):
            return PilotRole.WARM

        if await self._store.try_lock(self._lock_key(fingerprint), LOCK_TTL_SECONDS):
            return PilotRole.PILOT

        budget_ms = self._wait_ms if wait_ms is None else wait_ms
        deadline = asyncio.get_running_loop().time() + budget_ms / 1000
        delay = 0.05
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(delay)
            if await self._store.get(self._warm_key(fingerprint)):
                return PilotRole.FOLLOWER
            # The lock going free without warmth means the pilot died. Take
            # the seat rather than waiting out a timeout nobody will end.
            if await self._store.try_lock(self._lock_key(fingerprint), LOCK_TTL_SECONDS):
                return PilotRole.PILOT
            delay = min(delay * 1.6, 0.4)

        log.info("cache pilot timeout for %s; proceeding cold", fingerprint[:12])
        return PilotRole.TIMEOUT

    @contextlib.asynccontextmanager
    async def holding(self, fingerprint: str, role: PilotRole):
        """Keep the pilot lock alive while the upstream call is in flight.

        A no-op for every role but PILOT. The heartbeat re-arms the lock at
        half its TTL, so a live pilot is never deposed, and a dead one loses
        the seat within seconds — the two failure modes a fixed TTL conflates.
        """
        if role is not PilotRole.PILOT or not self._enabled or not fingerprint:
            yield
            return

        lock_key = self._lock_key(fingerprint)

        async def beat() -> None:
            while True:
                await asyncio.sleep(LOCK_TTL_SECONDS / 2)
                await self._store.set(lock_key, "1", ttl=LOCK_TTL_SECONDS)

        task = asyncio.create_task(beat())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def mark_warm(self, fingerprint: str, ttl_seconds: int) -> None:
        """Release the followers.

        Call this as early as truth allows: the provider cache is readable from
        the first token, so the streaming path fires this on the first chunk.
        The unary path has no first-token hook and calls it on completion.
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
