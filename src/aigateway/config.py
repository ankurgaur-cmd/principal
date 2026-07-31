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
