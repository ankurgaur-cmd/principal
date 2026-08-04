"""Runtime configuration.

Every knob here corresponds to a decision recorded in ARCHITECTURE.md. If you change
a default, change the rationale there too — the defaults are load-bearing.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="GATEWAY_", extra="ignore"
    )

    # Credentials keep their conventional names rather than taking the
    # GATEWAY_ prefix — an explicit alias, because relying on os.getenv would
    # miss values that live only in .env and never reach the process env.
    anthropic_api_key: str | None = Field(
        None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "GATEWAY_ANTHROPIC_API_KEY")
    )
    openai_api_key: str | None = Field(
        None, validation_alias=AliasChoices("OPENAI_API_KEY", "GATEWAY_OPENAI_API_KEY")
    )

    # -- state --
    redis_url: str | None = Field(
        None, validation_alias=AliasChoices("REDIS_URL", "GATEWAY_REDIS_URL")
    )

    # -- auth --
    auth_mode: Literal["dev", "jwt"] = "dev"
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"

    # -- routing --
    session_ttl_seconds: int = 300
    cache_aware_routing: bool = True
    escalate_only: bool = True
    llm_classifier_enabled: bool = True
    classifier_model: str = "claude-haiku-4-5"
    classifier_min_confidence: float = 0.6

    # -- cache --
    cache_pilot_enabled: bool = True
    cache_pilot_wait_ms: int = 4000
    cache_ttl: Literal["5m", "1h"] = "5m"
    semantic_cache_enabled: bool = False

    # -- model pool health --
    # Active probe cadence. 0 disables probing; passive observation of real
    # traffic still drives the breaker.
    health_probe_interval_seconds: int = 30
    # Consecutive real-traffic failures before a model is taken out of rotation.
    breaker_failure_threshold: int = 3
    # How long a broken model stays out before one trial request is admitted.
    breaker_cooldown_seconds: int = 60

    # -- vendor preference --
    # Score multiplier per provider, applied to the estimated cost. 1.0 is
    # neutral (pure price). Below 1.0 favours a vendor, above 1.0 penalises it.
    #
    # This exists because routing on sticker price alone will always concentrate
    # traffic on whoever is cheapest per token, and that is not always the right
    # business decision — you may have a committed-spend agreement, a data
    # -residency constraint, or a quality preference that price cannot express.
    # Example: GATEWAY_VENDOR_WEIGHTS='{"anthropic": 0.6, "openai": 1.0}'
    vendor_weights: dict[str, float] = {}

    # -- quality-adjusted routing --
    # Feed observed response quality back into model selection. A model that
    # keeps failing a given intent becomes more expensive to choose.
    quality_routing_enabled: bool = True
    # Observations of one (model, intent) pair to keep. Old failures age out.
    quality_window: int = 50
    # Below this many observations the multiplier is exactly 1.0 — penalising
    # on one bad response would be superstition.
    quality_min_samples: int = 5
    # Ceiling on the penalty, so a bad patch cannot exile a model forever.
    quality_max_penalty: float = 4.0
    # Fraction of requests that ignore reputation entirely. Without this a
    # penalised model never gets traffic, so it can never recover — the
    # feedback loop becomes a ratchet.
    quality_exploration_rate: float = 0.05

    # -- response quality --
    # Deterministic checks are always on and free. The LLM grader is a real
    # billable call per request, so it is opt-in.
    quality_judge_enabled: bool = False
    quality_judge_model: str = "claude-haiku-4-5"

    # -- optional work, switchable ------------------------------------------
    #
    # Measure before turning any of these off. On this gateway every local
    # stage *together* costs about 1.3ms against an upstream call of 18-58
    # seconds, so switching them off buys roughly a thousandth of a percent and
    # costs you the ability to explain what happened. They exist for the case
    # where you are running the gateway hot and want the serving path minimal,
    # not as a latency fix — the levers that move latency are `max_tokens`,
    # `effort`, and which model you land on, in that order.
    #
    # `auto_size_max_tokens` is the exception that genuinely matters: it sizes
    # an absent output budget to the intent instead of a single global default,
    # and output budget is what dominates wall-clock. Turning it off restores
    # the old one-size default and will make short work slow again.
    auto_size_max_tokens: bool = True
    # Learned latency baselines, and the colours the console derives from them.
    latency_baselines_enabled: bool = True
    # Per-transaction hop trace: origination and every upstream call.
    hop_trace_enabled: bool = True
    # Deterministic response checks. Free, and they are what makes a claimed
    # saving falsifiable — turning these off is the one that actually costs you
    # something.
    quality_checks_enabled: bool = True
    # Effort scoring on every response.
    effort_tracking_enabled: bool = True

    # -- governance --
    default_tenant_daily_usd: float = 50.0
    default_tenant_rpm: int = 600
    budget_mode: Literal["soft", "hard"] = "soft"
    preflight_exact_threshold: float = 0.85

    # -- observability --
    record_path: str = "./var/records.jsonl"
    log_level: str = "INFO"

    def model_post_init(self, __context) -> None:  # noqa: D105
        # An empty string in .env should read as "unset", not as a blank key.
        for field in ("anthropic_api_key", "openai_api_key", "redis_url"):
            if getattr(self, field) == "":
                object.__setattr__(self, field, None)


@lru_cache
def get_settings() -> Settings:
    return Settings()
