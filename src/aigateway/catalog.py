"""Model catalog: capability matrix, pricing, and cache economics.

This is the single source of truth the router scores against. Two properties
matter more than they look:

* ``min_cacheable_tokens`` is **not monotonic across generations**. A 3K-token
  prefix caches on Opus 5 and Sonnet 5 and silently does not on Haiku 4.5.
  A router that ignores this will "save" money by picking Haiku and then pay
  full price on every input token.
* ``supports_sampling_params`` is false for the current Anthropic frontier
  models — sending ``temperature``/``top_p``/``top_k`` is a hard 400. The
  gateway strips them rather than surfacing a vendor error to an agent that
  did nothing wrong.

Anthropic figures are first-party API rates. **OpenAI entries are
config-driven placeholders** — verify ids and prices against OpenAI's current
pricing page before you trust the cost ledger for chargeback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Tier(IntEnum):
    """Capability tiers. Ordering is meaningful — the router compares them."""

    LIGHT = 1  # classification, extraction, short-form
    STANDARD = 2  # most production work, tool orchestration
    HEAVY = 3  # long-horizon agentic, hard reasoning, deep code work


class Capability:
    TOOLS = "tools"
    VISION = "vision"
    STRUCTURED_OUTPUTS = "structured_outputs"
    EXTENDED_THINKING = "extended_thinking"
    EXPLICIT_CACHE_BREAKPOINTS = "explicit_cache_breakpoints"
    AUTO_PREFIX_CACHE = "auto_prefix_cache"


@dataclass(frozen=True)
class ModelSpec:
    key: str  # gateway-facing id
    provider: str  # "anthropic" | "openai"
    vendor_model_id: str
    tier: Tier

    price_in_per_mtok: float
    price_out_per_mtok: float

    context_window: int
    max_output_tokens: int

    # Cache economics. Anthropic: reads ~0.1x, writes 1.25x (5m) / 2x (1h).
    cache_read_multiplier: float = 0.1
    cache_write_multiplier_5m: float = 1.25
    cache_write_multiplier_1h: float = 2.0
    # Prefixes shorter than this never cache. No error — just a silent miss.
    min_cacheable_tokens: int = 1024

    capabilities: frozenset[str] = field(default_factory=frozenset)

    # Vendor quirks the adapters must honour.
    supports_sampling_params: bool = True
    thinking_default_on: bool = False
    # Some models reject thinking:disabled above a given effort level.
    thinking_disable_max_effort: str | None = None
    # Separate rate-limit pool from sibling models (affects load-shedding).
    rate_limit_pool: str = "default"

    # False means the price above is a placeholder, not a confirmed rate.
    # This matters more than it looks: the router picks the cheapest capable
    # model, so a wrong price does not cause a small billing error — it
    # silently sends *all* traffic in that tier to the wrong vendor.
    price_verified: bool = True

    def supports(self, cap: str) -> bool:
        return cap in self.capabilities

    def cache_write_multiplier(self, ttl: str) -> float:
        return self.cache_write_multiplier_1h if ttl == "1h" else self.cache_write_multiplier_5m


_ANTHROPIC_CAPS = frozenset(
    {
        Capability.TOOLS,
        Capability.VISION,
        Capability.STRUCTURED_OUTPUTS,
        Capability.EXTENDED_THINKING,
        Capability.EXPLICIT_CACHE_BREAKPOINTS,
    }
)

_OPENAI_CAPS = frozenset(
    {
        Capability.TOOLS,
        Capability.VISION,
        Capability.STRUCTURED_OUTPUTS,
        Capability.AUTO_PREFIX_CACHE,
    }
)


CATALOG: dict[str, ModelSpec] = {
    # ---------------- Anthropic ----------------
    "claude-haiku-4-5": ModelSpec(
        key="claude-haiku-4-5",
        provider="anthropic",
        vendor_model_id="claude-haiku-4-5",
        tier=Tier.LIGHT,
        price_in_per_mtok=1.00,
        price_out_per_mtok=5.00,
        context_window=200_000,
        max_output_tokens=64_000,
        min_cacheable_tokens=4096,  # note: highest minimum of the three
        capabilities=_ANTHROPIC_CAPS - {Capability.EXTENDED_THINKING},
        supports_sampling_params=True,
        rate_limit_pool="anthropic-haiku",
    ),
    "claude-sonnet-5": ModelSpec(
        key="claude-sonnet-5",
        provider="anthropic",
        vendor_model_id="claude-sonnet-5",
        tier=Tier.STANDARD,
        # $3/$15 list; introductory $2/$10 runs through 2026-08-31.
        price_in_per_mtok=3.00,
        price_out_per_mtok=15.00,
        context_window=1_000_000,
        max_output_tokens=128_000,
        min_cacheable_tokens=1024,
        capabilities=_ANTHROPIC_CAPS,
        supports_sampling_params=False,  # non-default values are a 400
        thinking_default_on=True,
        rate_limit_pool="anthropic-sonnet-5",
    ),
    "claude-opus-5": ModelSpec(
        key="claude-opus-5",
        provider="anthropic",
        vendor_model_id="claude-opus-5",
        tier=Tier.HEAVY,
        price_in_per_mtok=5.00,
        price_out_per_mtok=25.00,
        context_window=1_000_000,
        max_output_tokens=128_000,
        min_cacheable_tokens=512,  # lowest minimum — short prefixes cache here
        capabilities=_ANTHROPIC_CAPS,
        supports_sampling_params=False,
        thinking_default_on=True,
        thinking_disable_max_effort="high",  # disabling above `high` is a 400
        # Does NOT draw from the combined Opus 4.x pool.
        rate_limit_pool="anthropic-opus-5",
    ),
    # ---------------- OpenAI (verify ids + pricing) ----------------
    "gpt-5-nano": ModelSpec(
        key="gpt-5-nano",
        provider="openai",
        vendor_model_id="gpt-5-nano",
        tier=Tier.LIGHT,
        price_in_per_mtok=0.05,
        price_out_per_mtok=0.40,
        context_window=400_000,
        max_output_tokens=128_000,
        cache_read_multiplier=0.1,
        cache_write_multiplier_5m=1.0,  # automatic caching: no write premium
        cache_write_multiplier_1h=1.0,
        min_cacheable_tokens=1024,
        capabilities=_OPENAI_CAPS,
        rate_limit_pool="openai-nano",
        price_verified=False,
    ),
    "gpt-5-mini": ModelSpec(
        key="gpt-5-mini",
        provider="openai",
        vendor_model_id="gpt-5-mini",
        tier=Tier.STANDARD,
        price_in_per_mtok=0.25,
        price_out_per_mtok=2.00,
        context_window=400_000,
        max_output_tokens=128_000,
        cache_write_multiplier_5m=1.0,
        cache_write_multiplier_1h=1.0,
        min_cacheable_tokens=1024,
        capabilities=_OPENAI_CAPS,
        rate_limit_pool="openai-mini",
        price_verified=False,
    ),
    "gpt-5": ModelSpec(
        key="gpt-5",
        provider="openai",
        vendor_model_id="gpt-5",
        tier=Tier.HEAVY,
        price_in_per_mtok=1.25,
        price_out_per_mtok=10.00,
        context_window=400_000,
        max_output_tokens=128_000,
        cache_write_multiplier_5m=1.0,
        cache_write_multiplier_1h=1.0,
        min_cacheable_tokens=1024,
        capabilities=_OPENAI_CAPS,
        rate_limit_pool="openai-flagship",
        price_verified=False,
    ),
}


def get_model(key: str) -> ModelSpec | None:
    return CATALOG.get(key)


def models_at_or_above(tier: Tier) -> list[ModelSpec]:
    return [m for m in CATALOG.values() if m.tier >= tier]


def available_models(enabled_providers: set[str]) -> list[ModelSpec]:
    return [m for m in CATALOG.values() if m.provider in enabled_providers]


def unverified_prices() -> list[str]:
    """Models whose price is a placeholder.

    Worth surfacing loudly: the router selects on price, so an unverified rate
    does not produce a rounding error in the ledger — it can route an entire
    tier to the wrong vendor and look deliberate while doing it.
    """
    return [m.key for m in CATALOG.values() if not m.price_verified]
