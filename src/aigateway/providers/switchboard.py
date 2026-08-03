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

import logging

from ..catalog import CATALOG

log = logging.getLogger(__name__)


class Switchboard:
    """In-memory on/off state for models and providers.

    Not persisted: a restart returns to "everything the credentials allow",
    which is the safe default. If you need a disable to survive a restart, take
    the credential away instead — that is a stronger statement than a toggle.
    """

    def __init__(self) -> None:
        self._disabled_models: set[str] = set()
        self._disabled_providers: set[str] = set()

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
