"""Intent taxonomy and the intent -> tier policy table.

**This is placeholder taxonomy and you should replace it.** A routing table is
only as good as its intent classes, and the right classes are the ones your
agents actually emit — not a generic list. Treat this as the shape to fill in,
and expect the first week of `records.jsonl` to tell you what the real classes
are.

``min_tier`` is a floor, not a target. The router still picks the cheapest model
at or above the floor, so widening the catalog is how you get cheaper, not
editing this table downward.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..catalog import Tier


@dataclass(frozen=True)
class IntentPolicy:
    intent: str
    min_tier: Tier
    effort: str
    # Escalate one tier when the request carries these signals.
    escalate_on_tools: bool = False
    notes: str = ""


INTENT_POLICY: dict[str, IntentPolicy] = {
    # ---- light: bounded, well-specified, schema-shaped ----
    "classify": IntentPolicy("classify", Tier.LIGHT, "low", notes="label selection"),
    "extract": IntentPolicy(
        "extract", Tier.LIGHT, "low", notes="structured field extraction from text"
    ),
    "summarize": IntentPolicy("summarize", Tier.LIGHT, "low"),
    "translate": IntentPolicy("translate", Tier.LIGHT, "low"),
    "format": IntentPolicy("format", Tier.LIGHT, "low", notes="rewrite/reshape, no new reasoning"),
    # ---- standard: most production traffic ----
    "qa": IntentPolicy("qa", Tier.STANDARD, "medium", notes="grounded question answering"),
    "chat": IntentPolicy("chat", Tier.STANDARD, "medium"),
    "plan": IntentPolicy(
        "plan", Tier.STANDARD, "high", escalate_on_tools=True, notes="decompose a task"
    ),
    "tool_orchestration": IntentPolicy(
        "tool_orchestration", Tier.STANDARD, "high", notes="multi-tool sequencing"
    ),
    "analysis": IntentPolicy("analysis", Tier.STANDARD, "high"),
    "code_write": IntentPolicy("code_write", Tier.STANDARD, "high", escalate_on_tools=True),
    # ---- heavy: reasoning depth is the product ----
    "code_review": IntentPolicy(
        "code_review",
        Tier.HEAVY,
        "high",
        notes="recall matters; do not filter severity at the finding stage",
    ),
    "hard_debug": IntentPolicy("hard_debug", Tier.HEAVY, "xhigh"),
    "architecture": IntentPolicy("architecture", Tier.HEAVY, "xhigh"),
    "long_horizon_agentic": IntentPolicy(
        "long_horizon_agentic",
        Tier.HEAVY,
        "xhigh",
        notes="give the full spec up front; expect multi-minute turns",
    ),
    # ---- fallback ----
    "unknown": IntentPolicy(
        "unknown", Tier.STANDARD, "medium", notes="classifier abstained; middle tier is safest"
    ),
}

_TIER_NAMES = {"light": Tier.LIGHT, "standard": Tier.STANDARD, "heavy": Tier.HEAVY}


def tier_from_name(name: str | None) -> Tier | None:
    return _TIER_NAMES.get((name or "").lower())


def policy_for(intent: str) -> IntentPolicy:
    return INTENT_POLICY.get(intent, INTENT_POLICY["unknown"])
