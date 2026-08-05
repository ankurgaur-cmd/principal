"""Wire schemas.

Two layers on purpose:

* ``ChatCompletionRequest``/``Response`` — the OpenAI-compatible surface your
  agents talk to. They point ``base_url`` at the gateway and nothing else changes.
* ``CanonicalRequest`` — the internal, vendor-neutral representation the router
  and adapters operate on. Vendor-specific escapes live in ``vendor_overrides``
  so the neutral core never becomes a lowest-common-denominator schema.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# OpenAI-compatible surface
# --------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class GatewayExtensions(BaseModel):
    """Gateway-specific hints, namespaced so the request stays OpenAI-valid.

    Sent as ``{"x_gateway": {...}}`` in the request body; unknown to vendors.
    """

    session_id: str | None = Field(
        None,
        description="Groups turns of one workflow. Routing is sticky per session; "
        "omitting it forfeits cache-aware routing.",
    )
    intent: str | None = Field(
        None,
        description="Caller-declared intent. Accepted as given, but overridden when "
        "the request is plainly heavier than the label — never lighter.",
    )
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    cache_hints: list[Literal["system", "tools", "history", "last_turn"]] | None = Field(
        None,
        description="Where the gateway should place cache breakpoints. Translated to "
        "cache_control on Anthropic; ignored on providers with automatic caching.",
    )
    pin_model: str | None = Field(
        None, description="Bypass the router. Logged as a routing bypass."
    )
    max_tier: Literal["light", "standard", "heavy"] | None = None
    vendor_overrides: dict[str, Any] = Field(default_factory=dict)


class ChatCompletionRequest(BaseModel):
    model: str = Field(
        "auto",
        description="'auto' delegates to the router. A catalog key pins that model.",
    )
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None
    user: str | None = None
    x_gateway: GatewayExtensions = Field(default_factory=GatewayExtensions)

    def resolved_max_tokens(self) -> int:
        # Don't lowball: hitting the cap truncates mid-thought and forces a retry.
        # Only used when the caller gave no budget *and* the intent has no policy
        # default — see `chose_max_tokens`.
        return self.max_tokens or self.max_completion_tokens or 16_000

    def chose_max_tokens(self) -> bool:
        """Did the caller actually pick a budget, or are we defaulting?

        Output budget is what governs latency, so the difference matters: a
        caller's number is a decision to respect, and an absent one should be
        sized to the work rather than to a single global fallback.
        """
        return self.max_tokens is not None or self.max_completion_tokens is not None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Surfaced explicitly: cache_read stuck at zero is the #1 symptom of a
    # broken caching design, and it is invisible in the standard OpenAI shape.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class GatewayMeta(BaseModel):
    """Routing transparency, returned on every response."""

    trace_id: str
    session_id: str | None = None
    declared_intent: str | None = None
    resolved_intent: str
    intent_confidence: float
    intent_source: str
    chosen_model: str
    provider: str
    routing_reason: str
    tier: str
    cache_state: str
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    cache_savings_usd: float = 0.0
    fallback_chain: list[str] = Field(default_factory=list)
    degraded: bool = False
    latency_ms: int = 0
    # The road not taken: what else could have served this, and at what price.
    # Surfaced so a routing decision can be argued with, not just observed.
    considered: list[dict[str, Any]] = Field(default_factory=list)
    pilot_role: str = ""
    # Did the answer actually come back usable? See quality.py — a failure here
    # is evidence the routing decision was wrong, not just a bad response.
    quality: dict[str, Any] = Field(default_factory=dict)
    # Full hop trace: origination plus every upstream call, including the
    # attempts that failed and triggered a fallback.
    trace: dict[str, Any] = Field(default_factory=dict)
    prefix_tokens_est: int = 0
    volatile_tokens_est: int = 0
    dropped_params: list[str] = Field(default_factory=list)


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
    x_gateway: GatewayMeta


# --------------------------------------------------------------------------
# Internal canonical representation
# --------------------------------------------------------------------------
class ToolDef(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool = False


class CanonicalRequest(BaseModel):
    """Vendor-neutral request. Adapters compile this per provider."""

    system: list[str] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)
    tools: list[ToolDef] = Field(default_factory=list)
    tool_choice: Any = None
    response_schema: dict[str, Any] | None = None
    max_tokens: int = 16_000
    max_tokens_explicit: bool = False
    effort: str | None = None
    stream: bool = False

    # Sampling params are carried but may be dropped per model — see catalog.
    temperature: float | None = None
    top_p: float | None = None

    session_id: str | None = None
    intent_hint: str | None = None
    cache_hints: list[str] = Field(default_factory=lambda: ["system", "tools"])
    pin_model: str | None = None
    max_tier: str | None = None
    vendor_overrides: dict[str, Any] = Field(default_factory=dict)

    def sorted_tools(self) -> list[ToolDef]:
        """Deterministic tool order.

        Tool definitions render at position 0 of the prompt. Any churn in their
        order invalidates the entire prefix cache, on every provider.
        """
        return sorted(self.tools, key=lambda t: t.name)


class ProviderResponse(BaseModel):
    text: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str = "stop"
    model: str
    usage: Usage
    refusal_category: str | None = None
    raw_stop_reason: str | None = None
