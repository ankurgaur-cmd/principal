"""Operator switchboard: turn models and vendors on and off at runtime.

Deliberately separate from ``HealthMonitor``, even though both end up gating the
same routing decision. They answer different questions:

* **Health** is *observed state* — the model is failing, so stop sending to it.
  It heals on its own when the circuit closes.
* **The switchboard** is an *operator decision* — I have turned this off, and it
  stays off until I turn it back on. Nothing heals it.

Collapsing the two would mean a circuit breaker could silently re-enable
something a human deliberately disabled, which is exactly the surprise you do
not want at 3am. Keeping them apart also makes the routing explanation honest:
"circuit open (unhealthy)" and "switched off by an operator" are different
reasons and read differently in the console.

The main use is testing the routing logic: switch a vendor off and watch where
the traffic actually goes.
"""

from __future__ import annotations

import json
import logging

from ..catalog import CATALOG

log = logging.getLogger(__name__)


class Switchboard:
    """On/off state for models and providers, shared through the store.

    The queries stay synchronous over a local mirror — the router calls them
    per candidate. The mirror is refreshed from the store once per routing
    decision (``refresh``) and written through on every mutation (``save``),
    so with Redis every worker sees the same switches. Without a store it
    degrades to the old per-process behaviour, which is correct for the only
    deployment shape that has no store: a single dev process.

    With Redis this state also survives a restart. That is the point, not an
    accident: "switched off" is an operator decision, and a decision that
    silently un-makes itself when a pod restarts is exactly the 3am surprise
    this class exists to prevent. ``reset`` is the deliberate way back on.
    """

    _KEY = "switchboard:state"

    def __init__(self, store=None) -> None:
        self._store = store
        self._disabled_models: set[str] = set()
        self._disabled_providers: set[str] = set()

    # -- store sync ----------------------------------------------------------
    async def refresh(self) -> None:
        """Adopt the shared state. No-op without a store."""
        if self._store is None:
            return
        raw = await self._store.get(self._KEY)
        if raw is None:
            return  # nothing persisted yet; the local view stands
        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return
        self._disabled_models = set(data.get("models", []))
        self._disabled_providers = set(data.get("providers", []))

    async def save(self) -> None:
        """Publish the local view. Call after every mutation."""
        if self._store is None:
            return
        await self._store.set(
            self._KEY,
            json.dumps(
                {
                    "models": sorted(self._disabled_models),
                    "providers": sorted(self._disabled_providers),
                }
            ),
        )

    # -- queries -----------------------------------------------------------
    def is_enabled(self, model_key: str, provider: str) -> bool:
        return provider not in self._disabled_providers and model_key not in self._disabled_models

    def reason(self, model_key: str, provider: str) -> str | None:
        """Why this model is unavailable, or None if it is available."""
        if provider in self._disabled_providers:
            return f"vendor '{provider}' switched off by an operator"
        if model_key in self._disabled_models:
            return "switched off by an operator"
        return None

    # -- mutations ---------------------------------------------------------
    def set_model(self, model_key: str, enabled: bool) -> None:
        if model_key not in CATALOG:
            from ..errors import NoCapableModel

            raise NoCapableModel(f"unknown model '{model_key}'")
        if enabled:
            self._disabled_models.discard(model_key)
        else:
            self._disabled_models.add(model_key)
        log.info("model %s %s", model_key, "enabled" if enabled else "disabled")

    def set_provider(self, provider: str, enabled: bool) -> None:
        known = {m.provider for m in CATALOG.values()}
        if provider not in known:
            from ..errors import NoCapableModel

            raise NoCapableModel(f"unknown provider '{provider}'")
        if enabled:
            self._disabled_providers.discard(provider)
        else:
            self._disabled_providers.add(provider)
        log.info("provider %s %s", provider, "enabled" if enabled else "disabled")

    def reset(self) -> None:
        self._disabled_models.clear()
        self._disabled_providers.clear()

    # -- reporting ---------------------------------------------------------
    def state(self) -> dict:
        providers = sorted({m.provider for m in CATALOG.values()})
        return {
            "providers": [
                {"provider": p, "enabled": p not in self._disabled_providers}
                for p in providers
            ],
            "models": [
                {
                    "model": m.key,
                    "provider": m.provider,
                    "tier": m.tier.name.lower(),
                    "enabled": self.is_enabled(m.key, m.provider),
                    # Distinguishes "I turned this model off" from "I turned the
                    # whole vendor off", which look identical otherwise.
                    "disabled_by_provider": m.provider in self._disabled_providers,
                }
                for m in CATALOG.values()
            ],
            "disabled_models": sorted(self._disabled_models),
            "disabled_providers": sorted(self._disabled_providers),
        }
