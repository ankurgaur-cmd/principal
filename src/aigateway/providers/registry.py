"""Provider registry and cross-vendor fallback ordering."""

from __future__ import annotations

import logging

from ..catalog import CATALOG, ModelSpec, Tier, get_model
from ..config import Settings

log = logging.getLogger(__name__)


BUILDERS = {
    "anthropic": lambda key, kind="api_key": _build_anthropic(key, kind),
    "openai": lambda key, kind="api_key": _build_openai(key, kind),
}

# Which credential shapes each vendor actually honours. OpenAI offers no
# subscription-token path to the API — a ChatGPT plan is not API access —
# and pretending otherwise here would fail with a 401 the user cannot fix.
CREDENTIAL_KINDS = {
    "anthropic": ("api_key", "oauth_token"),
    "openai": ("api_key",),
}


def _build_anthropic(key: str, kind: str = "api_key"):
    from .anthropic_provider import AnthropicProvider

    if kind == "oauth_token":
        return AnthropicProvider(auth_token=key)
    return AnthropicProvider(key)


def _build_openai(key: str, kind: str = "api_key"):
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
        self._kind: dict[str, str] = {}

        for name, key, kind in (
            ("anthropic", settings.anthropic_api_key, "api_key"),
            # A subscription OAuth token from the environment counts as a
            # configured provider — only if no API key claims the slot first.
            ("anthropic", getattr(settings, "anthropic_auth_token", None), "oauth_token"),
            ("openai", settings.openai_api_key, "api_key"),
        ):
            if key and name not in self._providers:
                self._providers[name] = BUILDERS[name](key, kind)
                self._masked[name] = mask(key)
                self._source[name] = "env"
                self._kind[name] = kind

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
        self._kinds()[provider] = "ambient"
        log.info("using ambient credentials for %s", provider)

    def set_credentials(
        self, provider: str, api_key: str, source: str = "runtime",
        kind: str = "api_key",
    ) -> None:
        """Hot-swap a provider's client.

        The key is held in memory on the client object and nowhere else — not
        logged, not echoed back, not written to disk unless the caller
        explicitly asks for that separately.
        """
        if provider not in BUILDERS:
            from ..errors import NoCapableModel

            raise NoCapableModel(f"unknown provider '{provider}'")
        if kind not in CREDENTIAL_KINDS.get(provider, ("api_key",)):
            from ..errors import NoCapableModel

            raise NoCapableModel(
                f"{provider} does not accept '{kind}' credentials — "
                f"supported: {', '.join(CREDENTIAL_KINDS[provider])}"
            )
        self._providers[provider] = BUILDERS[provider](api_key, kind)
        self._masked[provider] = mask(api_key)
        self._source[provider] = source
        self._kinds()[provider] = kind
        log.info(
            "credentials set for %s (%s, %s)", provider, self._masked[provider], kind
        )

    def clear_credentials(self, provider: str) -> None:
        self._providers.pop(provider, None)
        self._masked.pop(provider, None)
        self._source.pop(provider, None)
        self._kinds().pop(provider, None)

    def _kinds(self) -> dict[str, str]:
        # getattr-guarded because tests replace __init__ wholesale.
        if not hasattr(self, "_kind"):
            self._kind = {}
        return self._kind

    def status(self) -> list[dict]:
        """Masked credential status. Never includes a key."""
        return [
            {
                "provider": name,
                "configured": name in self._providers,
                "masked_key": self._masked.get(name, ""),
                "source": self._source.get(name, ""),
                "kind": self._kinds().get(name, "api_key" if name in self._providers else ""),
                "accepts": list(CREDENTIAL_KINDS.get(name, ("api_key",))),
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

    def fallback_chain(
        self,
        model: ModelSpec,
        max_hops: int = 2,
        *,
        switchboard=None,
        health=None,
    ) -> list[ModelSpec]:
        """Ordered fallbacks for an upstream failure.

        Same vendor first, deliberately. A cross-vendor hop mid-conversation
        discards the warm prompt cache *and* any provider-native state, so it is
        reserved for the case where the whole vendor is unreachable.

        **The same availability gates the router applies must apply here.** This
        chain used to filter on credentials and tier alone, so a model an
        operator had explicitly switched off — or one whose circuit breaker was
        open — could still be handed live traffic the moment the primary
        stumbled. The router refusing to route somewhere and the fallback going
        there anyway is the worst kind of inconsistency: it only shows up under
        failure, which is exactly when an operator is relying on the switch.
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

        def available(m: ModelSpec) -> bool:
            if m.provider not in self._providers:
                return False
            # An operator's decision outranks an observation, and both outrank
            # convenience. Checked in that order for the same reason the router
            # checks them in that order.
            if switchboard is not None and not switchboard.is_enabled(m.key, m.provider):
                return False
            if health is not None and not health.is_available(m.key):
                return False
            return True

        return [m for m in chain if available(m)][:max_hops]
