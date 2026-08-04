"""Intent taxonomy and the intent -> tier policy table.

**This is placeholder taxonomy and you should replace it.** A routing table is
only as good as its intent classes, and the right classes are the ones your
agents actually emit — not a generic list. Treat this as the shape to fill in,
and expect the first week of `records.jsonl` to tell you what the real classes
are.

``min_tier`` is a floor, not a target. The router still picks the cheapest model
at or above the floor, so widening the catalog is how you get cheaper, not
editing this table downward.

``max_tokens`` is the third column and the one that governs **latency**. Output
budget is what a request actually spends its wall-clock on — measured on this
gateway, the same prompt took 18s at 1,200 tokens, 27s at 4,000 and 58s at 8,000,
while every gateway-local stage together took 1.3ms. Handing a classification the
same budget as an architecture review makes it slow for no benefit, and handing a
review a classification's budget makes it come back empty. One global default
cannot be right for both, so the budget belongs here, next to the tier.

A caller-supplied ``max_tokens`` always wins; this is the default when none is
given.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..catalog import Tier


@dataclass(frozen=True)
class IntentPolicy:
    intent: str
    min_tier: Tier
    effort: str
    # Default output budget. Sized to the reasoning floor for this intent's
    # effort plus room to answer — see quality.REASONING_FLOOR_BY_EFFORT.
    max_tokens: int = 4000
    # Escalate one tier when the request carries these signals.
    escalate_on_tools: bool = False
    notes: str = ""


INTENT_POLICY: dict[str, IntentPolicy] = {
    # ---- light: bounded, well-specified, schema-shaped ----
    "classify": IntentPolicy(
        "classify", Tier.LIGHT, "low", max_tokens=600, notes="label selection"
    ),
    "extract": IntentPolicy(
        "extract", Tier.LIGHT, "low", max_tokens=1200, notes="structured field extraction from text"
    ),
    "summarize": IntentPolicy("summarize", Tier.LIGHT, "low", max_tokens=1500),
    "translate": IntentPolicy("translate", Tier.LIGHT, "low", max_tokens=2000),
    "format": IntentPolicy(
        "format", Tier.LIGHT, "low", max_tokens=2000, notes="rewrite/reshape, no new reasoning"
    ),
    # ---- standard: most production traffic ----
    "qa": IntentPolicy(
        "qa", Tier.STANDARD, "medium", max_tokens=5000, notes="grounded question answering"
    ),
    "chat": IntentPolicy("chat", Tier.STANDARD, "medium", max_tokens=5000),
    "plan": IntentPolicy(
        "plan", Tier.STANDARD, "high", max_tokens=8000, escalate_on_tools=True,
        notes="decompose a task"
    ),
    "tool_orchestration": IntentPolicy(
        "tool_orchestration", Tier.STANDARD, "high", max_tokens=8000,
        notes="multi-tool sequencing"
    ),
    "analysis": IntentPolicy("analysis", Tier.STANDARD, "high", max_tokens=8000),
    "code_write": IntentPolicy(
        "code_write", Tier.STANDARD, "high", max_tokens=8000, escalate_on_tools=True
    ),
    # ---- heavy: reasoning depth is the product ----
    "code_review": IntentPolicy(
        "code_review",
        Tier.HEAVY,
        "high",
        max_tokens=8000,
        notes="recall matters; do not filter severity at the finding stage",
    ),
    "hard_debug": IntentPolicy("hard_debug", Tier.HEAVY, "xhigh", max_tokens=14000),
    "architecture": IntentPolicy("architecture", Tier.HEAVY, "xhigh", max_tokens=14000),
    "long_horizon_agentic": IntentPolicy(
        "long_horizon_agentic",
        Tier.HEAVY,
        "xhigh",
        max_tokens=16000,
        notes="give the full spec up front; expect multi-minute turns",
    ),
    # ---- fallback ----
    "unknown": IntentPolicy(
        "unknown", Tier.STANDARD, "medium", max_tokens=5000,
        notes="classifier abstained; middle tier is safest"
    ),
}

_TIER_NAMES = {"light": Tier.LIGHT, "standard": Tier.STANDARD, "heavy": Tier.HEAVY}


def tier_from_name(name: str | None) -> Tier | None:
    return _TIER_NAMES.get((name or "").lower())


def policy_for(intent: str) -> IntentPolicy:
    return INTENT_POLICY.get(intent, INTENT_POLICY["unknown"])
