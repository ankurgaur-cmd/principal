"""Provider registry and cross-vendor fallback ordering."""

from __future__ import annotations

import logging

from ..catalog import CATALOG, ModelSpec, Tier, get_model
from ..config import Settings

log = logging.getLogger(__name__)


BUILDERS = {
    "anthropic": lambda key: _build_anthropic(key),
    "openai": lambda key: _build_openai(key),
}


def _build_anthropic(key: str):
    from .anthropic_provider import AnthropicProvider

    return AnthropicProvider(key)


def _build_openai(key: str):
    from .openai_provider import OpenAIProvider

    return OpenAIProvider(key)


def mask(key: str) -> str:
    """Render a key for display. Never return the key itself, anywhere."""
    if not key:
        return ""
    return f"{key[:3]}…{key[-4:]}" if len(key) > 10 else "…"


class ProviderRegistry:
    def __init__(self, settings: Settings):
        self._providers: dict[str, object] = {}
        self._masked: dict[str, str] = {}
        self._source: dict[str, str] = {}

        for name, key in (
            ("anthropic", settings.anthropic_api_key),
            ("openai", settings.openai_api_key),
        ):
            if key:
                self._providers[name] = BUILDERS[name](key)
                self._masked[name] = mask(key)
                self._source[name] = "env"

        if not self._providers:
            log.warning(
                "no provider credentials configured — the gateway will route and "
                "score requests but cannot serve them. Add keys to .env, or paste "
                "them into the console at / (dev mode only)."
            )

    # -- runtime credential management -------------------------------------
    def use_ambient(self, provider: str) -> None:
        """Build a provider client with no explicit key.

        The SDKs resolve credentials on their own — for Anthropic that chain is
        ANTHROPIC_API_KEY → ANTHROPIC_AUTH_TOKEN → an OAuth profile created by
        ``ant auth login``. So a machine already authenticated for CLI use can
        serve traffic without a key ever being pasted anywhere.
        """
        if provider not in BUILDERS:
            from ..errors import NoCapableModel

            raise NoCapableModel(f"unknown provider '{provider}'")
        self._providers[provider] = BUILDERS[provider](None)
        self._masked[provider] = "(ambient)"
        self._source[provider] = "ambient"
        log.info("using ambient credentials for %s", provider)

    def set_credentials(self, provider: str, api_key: str, source: str = "runtime") -> None:
        """Hot-swap a provider's client.

        The key is held in memory on the client object and nowhere else — not
        logged, not echoed back, not written to disk unless the caller
        explicitly asks for that separately.
        """
        if provider not in BUILDERS:
            from ..errors import NoCapableModel

            raise NoCapableModel(f"unknown provider '{provider}'")
        self._providers[provider] = BUILDERS[provider](api_key)
        self._masked[provider] = mask(api_key)
        self._source[provider] = source
        log.info("credentials set for %s (%s)", provider, self._masked[provider])

    def clear_credentials(self, provider: str) -> None:
        self._providers.pop(provider, None)
        self._masked.pop(provider, None)
        self._source.pop(provider, None)

    def status(self) -> list[dict]:
        """Masked credential status. Never includes a key."""
        return [
            {
                "provider": name,
                "configured": name in self._providers,
                "masked_key": self._masked.get(name, ""),
                "source": self._source.get(name, ""),
                "models": sorted(m.key for m in CATALOG.values() if m.provider == name),
            }
            for name in BUILDERS
        ]

    @property
    def enabled(self) -> set[str]:
        return set(self._providers)

    def get(self, provider: str):
        if provider not in self._providers:
            from ..errors import UpstreamError

            raise UpstreamError(f"provider '{provider}' is not configured", 503)
        return self._providers[provider]

    def for_model(self, model_key: str):
        spec = get_model(model_key)
        if spec is None:
            from ..errors import NoCapableModel

            raise NoCapableModel(f"unknown model '{model_key}'")
        return self.get(spec.provider)

    def fallback_chain(self, model: ModelSpec, max_hops: int = 2) -> list[ModelSpec]:
        """Ordered fallbacks for an upstream failure.

        Same vendor first, deliberately. A cross-vendor hop mid-conversation
        discards the warm prompt cache *and* any provider-native state, so it is
        reserved for the case where the whole vendor is unreachable.
        """
        chain: list[ModelSpec] = []

        same_vendor = sorted(
            (
                m
                for m in CATALOG.values()
                if m.provider == model.provider
                and m.key != model.key
                and m.tier >= min(model.tier, Tier.STANDARD)
            ),
            key=lambda m: (m.tier, m.price_in_per_mtok),
        )
        chain.extend(same_vendor)

        cross_vendor = sorted(
            (
                m
                for m in CATALOG.values()
                if m.provider != model.provider
                and m.provider in self._providers
                and m.tier >= model.tier
            ),
            key=lambda m: (m.tier, m.price_in_per_mtok),
        )
        chain.extend(cross_vendor)

        return [m for m in chain if m.provider in self._providers][:max_hops]
