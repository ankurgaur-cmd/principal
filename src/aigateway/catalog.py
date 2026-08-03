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

All prices were verified on 2026-07-31 against each vendor's published pricing
page (see ``price_source`` / ``price_checked`` on each entry). Model IDs were
confirmed against the live ``/v1/models`` endpoint for the configured accounts.

Re-verify periodically. The router selects on price, so a stale rate does not
produce a small billing error — it re-routes whole tiers. ``stale_prices()``
catches promotional rates that have lapsed; nothing catches a silent list-price
change except checking.
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
    price_source: str = ""
    price_checked: str = ""  # ISO date the rate was last confirmed
    # Promotional rates expire. A dated price that lapses silently is the same
    # failure mode as a placeholder, just delayed — so record the date and warn.
    price_expires: str | None = None
    price_after_expiry: tuple[float, float] | None = None
    price_note: str = ""

    # Some vendors charge more once a request crosses a context threshold —
    # OpenAI roughly doubles above 272K. A router that ignores this will
    # under-price large-context requests by 2x and pick the wrong model for
    # exactly the workloads where the bill is biggest.
    # Anthropic deliberately has no long-context premium: the full 1M window
    # is billed at the standard rate.
    long_context_threshold: int | None = None
    price_in_long_per_mtok: float | None = None
    price_out_long_per_mtok: float | None = None

    # Context windows are harder to verify than prices — vendors publish them
    # less consistently. False means the value is a conservative guess, which
    # only ever excludes a model from a large request (safe), never includes
    # one that cannot serve it.
    context_verified: bool = True

    def rates_for(self, input_tokens: int) -> tuple[float, float]:
        """Input/output rates that apply at this request size."""
        if (
            self.long_context_threshold is not None
            and input_tokens > self.long_context_threshold
            and self.price_in_long_per_mtok is not None
        ):
            return self.price_in_long_per_mtok, self.price_out_long_per_mtok
        return self.price_in_per_mtok, self.price_out_per_mtok

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


ANTHROPIC_PRICING = "https://platform.claude.com/docs/en/about-claude/pricing"
OPENAI_PRICING = "https://developers.openai.com/api/docs/pricing"
CHECKED = "2026-08-03"


CATALOG: dict[str, ModelSpec] = {
    # ================= Anthropic =================
    # Verified 2026-07-31 against https://platform.claude.com/docs/en/about-claude/pricing
    # Cache multipliers are published as exact rates and match 1.25x / 2x / 0.1x.
    "claude-haiku-4-5": ModelSpec(
        key="claude-haiku-4-5",
        provider="anthropic",
        vendor_model_id="claude-haiku-4-5",
        tier=Tier.LIGHT,
        price_in_per_mtok=1.00,
        price_out_per_mtok=5.00,
        context_window=200_000,
        max_output_tokens=64_000,
        min_cacheable_tokens=4096,  # highest minimum of the three — not monotonic
        capabilities=_ANTHROPIC_CAPS - {Capability.EXTENDED_THINKING},
        supports_sampling_params=True,
        rate_limit_pool="anthropic-haiku",
        price_source=ANTHROPIC_PRICING,
        price_checked=CHECKED,
    ),
    "claude-sonnet-5": ModelSpec(
        key="claude-sonnet-5",
        provider="anthropic",
        vendor_model_id="claude-sonnet-5",
        tier=Tier.STANDARD,
        # Introductory rate, in effect through 2026-08-31. Standard is $3/$15.
        price_in_per_mtok=2.00,
        price_out_per_mtok=10.00,
        price_expires="2026-08-31",
        price_after_expiry=(3.00, 15.00),
        price_note="introductory pricing; reverts to $3/$15 on 2026-09-01",
        context_window=1_000_000,
        max_output_tokens=128_000,
        min_cacheable_tokens=1024,
        capabilities=_ANTHROPIC_CAPS,
        supports_sampling_params=False,  # non-default values are a 400
        thinking_default_on=True,
        rate_limit_pool="anthropic-sonnet-5",
        price_source=ANTHROPIC_PRICING,
        price_checked=CHECKED,
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
        rate_limit_pool="anthropic-opus-5",  # separate pool from Opus 4.x
        price_source=ANTHROPIC_PRICING,
        price_checked=CHECKED,
    ),

    # ================= OpenAI =================
    # Verified 2026-07-31 against https://developers.openai.com/api/docs/pricing
    # OpenAI caches prefixes automatically: cached input is 0.1x and there is
    # no separate write premium, so the write multiplier is 1.0 rather than
    # Anthropic's 1.25x. That asymmetry is real and the router prices it.
    # NOTE: prices are verified; context windows below are not — the published
    # model page did not list these versions. They are conservative.
    "gpt-5-nano": ModelSpec(
        key="gpt-5-nano",
        provider="openai",
        vendor_model_id="gpt-5-nano",
        tier=Tier.LIGHT,
        price_in_per_mtok=0.05,
        price_out_per_mtok=0.40,
        context_window=272_000,
        max_output_tokens=128_000,
        cache_write_multiplier_5m=1.0,
        cache_write_multiplier_1h=1.0,
        min_cacheable_tokens=1024,
        capabilities=_OPENAI_CAPS,
        rate_limit_pool="openai-nano",
        long_context_threshold=272_000,
        price_in_long_per_mtok=0.1,
        price_out_long_per_mtok=0.8,
        context_verified=False,
        price_source=OPENAI_PRICING,
        price_checked=CHECKED,
    ),
    "gpt-5.4-nano": ModelSpec(
        key="gpt-5.4-nano",
        provider="openai",
        vendor_model_id="gpt-5.4-nano",
        tier=Tier.LIGHT,
        price_in_per_mtok=0.20,
        price_out_per_mtok=1.25,
        context_window=272_000,
        max_output_tokens=128_000,
        cache_write_multiplier_5m=1.0,
        cache_write_multiplier_1h=1.0,
        min_cacheable_tokens=1024,
        capabilities=_OPENAI_CAPS,
        rate_limit_pool="openai-nano",
        long_context_threshold=272_000,
        price_in_long_per_mtok=0.4,
        price_out_long_per_mtok=2.5,
        context_verified=False,
        price_source=OPENAI_PRICING,
        price_checked=CHECKED,
    ),
    "gpt-5-mini": ModelSpec(
        key="gpt-5-mini",
        provider="openai",
        vendor_model_id="gpt-5-mini",
        tier=Tier.STANDARD,
        price_in_per_mtok=0.25,
        price_out_per_mtok=2.00,
        context_window=272_000,
        max_output_tokens=128_000,
        cache_write_multiplier_5m=1.0,
        cache_write_multiplier_1h=1.0,
        min_cacheable_tokens=1024,
        capabilities=_OPENAI_CAPS,
        rate_limit_pool="openai-mini",
        long_context_threshold=272_000,
        price_in_long_per_mtok=0.5,
        price_out_long_per_mtok=4.0,
        context_verified=False,
        price_source=OPENAI_PRICING,
        price_checked=CHECKED,
    ),
    "gpt-5.4-mini": ModelSpec(
        key="gpt-5.4-mini",
        provider="openai",
        vendor_model_id="gpt-5.4-mini",
        tier=Tier.STANDARD,
        price_in_per_mtok=0.75,
        price_out_per_mtok=4.50,
        context_window=272_000,
        max_output_tokens=128_000,
        cache_write_multiplier_5m=1.0,
        cache_write_multiplier_1h=1.0,
        min_cacheable_tokens=1024,
        capabilities=_OPENAI_CAPS,
        rate_limit_pool="openai-mini",
        long_context_threshold=272_000,
        price_in_long_per_mtok=1.5,
        price_out_long_per_mtok=9.0,
        context_verified=False,
        price_source=OPENAI_PRICING,
        price_checked=CHECKED,
    ),
    "gpt-5": ModelSpec(
        key="gpt-5",
        provider="openai",
        vendor_model_id="gpt-5",
        tier=Tier.HEAVY,
        price_in_per_mtok=1.25,
        price_out_per_mtok=10.00,
        context_window=272_000,
        max_output_tokens=128_000,
        cache_write_multiplier_5m=1.0,
        cache_write_multiplier_1h=1.0,
        min_cacheable_tokens=1024,
        capabilities=_OPENAI_CAPS,
        rate_limit_pool="openai-flagship",
        long_context_threshold=272_000,
        price_in_long_per_mtok=2.5,
        price_out_long_per_mtok=20.0,
        context_verified=False,
        price_source=OPENAI_PRICING,
        price_checked=CHECKED,
    ),
    "gpt-5.4": ModelSpec(
        key="gpt-5.4",
        provider="openai",
        vendor_model_id="gpt-5.4",
        tier=Tier.HEAVY,
        price_in_per_mtok=2.50,
        price_out_per_mtok=15.00,
        context_window=272_000,
        max_output_tokens=128_000,
        cache_write_multiplier_5m=1.0,
        cache_write_multiplier_1h=1.0,
        min_cacheable_tokens=1024,
        capabilities=_OPENAI_CAPS,
        rate_limit_pool="openai-flagship",
        long_context_threshold=272_000,
        price_in_long_per_mtok=5.0,
        price_out_long_per_mtok=30.0,
        context_verified=False,
        price_source=OPENAI_PRICING,
        price_checked=CHECKED,
    ),
    # --- newly provisioned on this account (checked 2026-08-03) ---
    # These are all pricier than the gpt-5 generation, so price alone will not
    # select them. That is correct and deliberate: if they are better, the
    # quality reputation will earn them traffic. Asserting it up front would be
    # a guess dressed as a policy.
    "gpt-5.6-luna": ModelSpec(
        key="gpt-5.6-luna",
        provider="openai",
        vendor_model_id="gpt-5.6-luna",
        tier=Tier.LIGHT,
        price_in_per_mtok=0.20,
        price_out_per_mtok=1.20,
        long_context_threshold=272_000,
        price_in_long_per_mtok=0.40,
        price_out_long_per_mtok=1.80,
        context_window=1_050_000,
        max_output_tokens=128_000,
        cache_write_multiplier_5m=1.0,
        cache_write_multiplier_1h=1.0,
        min_cacheable_tokens=1024,
        capabilities=_OPENAI_CAPS,
        rate_limit_pool="openai-luna",
        price_source=OPENAI_PRICING,
        price_checked=CHECKED,
    ),
    "gpt-5.6-terra": ModelSpec(
        key="gpt-5.6-terra",
        provider="openai",
        vendor_model_id="gpt-5.6-terra",
        tier=Tier.STANDARD,
        price_in_per_mtok=2.00,
        price_out_per_mtok=12.00,
        long_context_threshold=272_000,
        price_in_long_per_mtok=4.00,
        price_out_long_per_mtok=18.00,
        context_window=1_050_000,
        max_output_tokens=128_000,
        cache_write_multiplier_5m=1.0,
        cache_write_multiplier_1h=1.0,
        min_cacheable_tokens=1024,
        capabilities=_OPENAI_CAPS,
        rate_limit_pool="openai-terra",
        price_source=OPENAI_PRICING,
        price_checked=CHECKED,
    ),
    "gpt-5.6-sol": ModelSpec(
        key="gpt-5.6-sol",
        provider="openai",
        vendor_model_id="gpt-5.6-sol",
        tier=Tier.HEAVY,
        price_in_per_mtok=5.00,
        price_out_per_mtok=30.00,
        long_context_threshold=272_000,
        price_in_long_per_mtok=10.00,
        price_out_long_per_mtok=45.00,
        context_window=1_050_000,
        max_output_tokens=128_000,
        cache_write_multiplier_5m=1.0,
        cache_write_multiplier_1h=1.0,
        min_cacheable_tokens=1024,
        capabilities=_OPENAI_CAPS,
        rate_limit_pool="openai-sol",
        price_source=OPENAI_PRICING,
        price_checked=CHECKED,
    ),
}


def get_model(key: str) -> ModelSpec | None:
    return CATALOG.get(key)


def models_at_or_above(tier: Tier) -> list[ModelSpec]:
    return [m for m in CATALOG.values() if m.tier >= tier]


def available_models(enabled_providers: set[str]) -> list[ModelSpec]:
    return [m for m in CATALOG.values() if m.provider in enabled_providers]


def unverified_prices() -> list[str]:
    """Models whose price is a placeholder rather than a confirmed rate."""
    return [m.key for m in CATALOG.values() if not m.price_verified]


def catalog_warnings() -> list[str]:
    """Internal inconsistencies in the catalog itself.

    The one that actually bit: a long-context price tier whose threshold equals
    the model's context window is unreachable — the tier can never apply, which
    means one of the two numbers is wrong. Since the router selects on price,
    silently carrying a contradictory rate is worse than saying so.
    """
    out: list[str] = []
    for m in CATALOG.values():
        if m.long_context_threshold and m.long_context_threshold >= m.context_window:
            out.append(
                f"{m.key}: long-context pricing above {m.long_context_threshold:,} tokens "
                f"can never apply — the context window is {m.context_window:,}. Either the "
                f"window is larger than recorded (likely: the vendor publishes a tier for it) "
                f"or the tier does not exist. context_verified="
                f"{m.context_verified}."
            )
        if not m.context_verified:
            out.append(f"{m.key}: context window is an unverified conservative estimate.")
    return out


def stale_prices(today: str | None = None) -> list[dict]:
    """Models whose promotional rate has lapsed.

    The router selects on price, so a rate that quietly expires re-routes
    traffic without anyone changing a line of code. Checked at startup.
    """
    import datetime

    now = today or datetime.date.today().isoformat()
    out = []
    for m in CATALOG.values():
        if m.price_expires and now > m.price_expires:
            out.append(
                {
                    "model": m.key,
                    "expired_on": m.price_expires,
                    "catalog_price": (m.price_in_per_mtok, m.price_out_per_mtok),
                    "actual_price": m.price_after_expiry,
                }
            )
    return out
