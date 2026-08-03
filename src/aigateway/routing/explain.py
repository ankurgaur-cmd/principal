"""Turn a routing decision into an explanation a human can read.

The router's ``reason`` string is a log line — precise, dense, and useless to
anyone who hasn't read the router source. This module produces the other
artifact: a short narrative that answers the question people actually ask, which
is *"why that model and not a cheaper one?"*.

It lives outside the router deliberately. The router's job is to decide; making
the decision legible is a separate concern with a separate audience, and mixing
them would put presentation strings in the scoring path.

The structure is fixed at five beats, because a routing decision always has the
same shape:

    1. what you asked for      the intent, and how we worked it out
    2. what that needs         the capability floor it implies
    3. who could do it         who qualified, and who was ruled out and why
    4. why this one            the single deciding factor
    5. what it cost            the price, and the price of the road not taken
"""

from __future__ import annotations

from ..catalog import Tier

# What each tier means in terms of the work, not the hardware.
TIER_MEANING = {
    Tier.LIGHT: "quick, well-specified work where a small model is enough",
    Tier.STANDARD: "everyday production work that needs solid reasoning",
    Tier.HEAVY: "deep reasoning, where getting it right matters more than the price",
}

# How each intent reads in plain English.
INTENT_PLAIN = {
    "classify": "sort something into a category",
    "extract": "pull structured fields out of text",
    "summarize": "condense something",
    "translate": "translate text",
    "format": "reshape or rewrite text without new reasoning",
    "qa": "answer a question from the material given",
    "chat": "hold a conversation",
    "plan": "break a task into steps",
    "tool_orchestration": "coordinate several tools",
    "analysis": "analyse something in depth",
    "code_write": "write or change code",
    "code_review": "review code for defects",
    "hard_debug": "track down a difficult bug",
    "architecture": "make a system-design judgement",
    "long_horizon_agentic": "run a long, multi-step task autonomously",
    "unknown": "something the classifier could not confidently label",
}

SOURCE_PLAIN = {
    "declared": "your agent told us directly",
    "rules": "matched a rule (keywords, request shape, size)",
    "embedding": "closest match among labelled examples",
    "llm": "a small, cheap model read it and labelled it",
    "llm-cached": "a small model labelled this shape earlier; reused that answer",
    "default": "nothing was confident, so we defaulted to the middle tier",
}

CACHE_PLAIN = {
    "warm_read": (
        "Reusing cached context",
        "This session already used this model, so the shared part of the prompt "
        "is cached — it bills at about a tenth of the normal rate.",
    ),
    "cold_write": (
        "Building the cache",
        "First request on this prompt, so the shared part is being written to "
        "the cache. It costs about 25% extra now and roughly 90% less on every "
        "later request in this session.",
    ),
    "uncached": (
        "Not cached",
        "The shared part of this prompt is too short to cache on this model, so "
        "every request pays full price for it.",
    ),
}

EXCLUSION_PLAIN = {
    "below the tier this task needs": "not capable enough for this task",
    "provider not configured": "no API key for this provider",
    "circuit open (unhealthy)": "taken out of rotation — it has been failing",
    "switched off by an operator": "switched off by you",
    "no tool support": "cannot use tools, and this request needs them",
    "no structured outputs": "cannot guarantee the JSON shape this request asks for",
}


def _humanise_exclusion(reason: str) -> str:
    if reason in EXCLUSION_PLAIN:
        return EXCLUSION_PLAIN[reason]
    if reason.startswith("vendor '") and "switched off" in reason:
        return reason.replace("switched off by an operator", "switched off by you")
    if reason.startswith("context "):
        return "context window too small for this prompt"
    if reason.startswith("max_output "):
        return "cannot produce a response this long"
    return reason


def _fmt(usd: float) -> str:
    return f"${usd:.4f}" if usd >= 0.0001 else f"${usd:.6f}"


def explain(decision, intent_confidence: float, intent_source: str) -> dict:
    """Build the narrative. Pure presentation — no decisions are made here."""
    model = decision.model
    intent = decision.intent or "unknown"
    qualified = decision.considered
    runner_up = next(
        (c for c in sorted(qualified, key=lambda c: c.cost_usd) if c.model.key != model.key),
        None,
    )

    # -- 4. the single deciding factor --------------------------------------
    if decision.pinned:
        verdict = "You pinned this model"
        why = "The request named a specific model, so the router was bypassed entirely."
    elif decision.degraded:
        verdict = "Budget forced a smaller model"
        why = (
            "This tenant is over its spending limit, so the router was capped to "
            "the cheapest tier that could still do the job."
        )
    elif decision.sticky:
        verdict = "Kept the model this session was already using"
        why = (
            f"This session already warmed up {model.key}. Switching to another model "
            "would throw that cached context away and pay to build it again — usually "
            "more than the cheaper per-token price would save."
        )
    elif len(qualified) == 1:
        verdict = "It was the only model that qualified"
        why = "Every other model was ruled out; see below."
    elif any(c.quality_multiplier > 1.0 for c in qualified):
        penalised = [c for c in qualified if c.quality_multiplier > 1.0]
        worst = max(penalised, key=lambda c: c.quality_multiplier)
        verdict = "Cheapest once past quality was taken into account"
        why = (
            f"{model.key} won on cost adjusted for observed quality. "
            f"{worst.model.key} is cheaper per token but has succeeded on only "
            f"{(worst.quality_success_rate or 0):.0%} of recent '{intent}' "
            f"requests, so it effectively costs "
            f"{worst.quality_multiplier:.1f}x its sticker price — you would "
            f"expect to retry it."
        )
    elif runner_up:
        est = decision.estimated_cost_usd
        ratio = runner_up.cost_usd / est if est else 1
        verdict = f"Cheapest of the {len(qualified)} models that qualified"
        why = (
            f"{model.key} is estimated at {_fmt(decision.estimated_cost_usd)} for this "
            f"request. The next option, {runner_up.model.key}, would be "
            f"{_fmt(runner_up.cost_usd)} — about {ratio:.1f}× more for the same task."
        )
    else:
        verdict = "Cheapest model that qualified"
        why = f"Estimated {_fmt(decision.estimated_cost_usd)} for this request."

    cache_title, cache_body = CACHE_PLAIN.get(
        decision.cache_state, ("Cache", decision.cache_state)
    )

    tier = decision.required_tier
    steps = [
        {
            "n": 1,
            "title": "What you asked for",
            "value": intent.replace("_", " "),
            "detail": (
                f"Read as a request to {INTENT_PLAIN.get(intent, intent)}. "
                f"Worked out because {SOURCE_PLAIN.get(intent_source, intent_source)} "
                f"({round(intent_confidence * 100)}% confident)."
            ),
        },
        {
            "n": 2,
            "title": "What that needs",
            "value": f"{tier.name.lower()} tier or better",
            "detail": (
                f"That is {TIER_MEANING.get(tier, '')}. Models below that tier "
                "were not considered, however cheap they are."
            ),
        },
        {
            "n": 3,
            "title": "Who could do it",
            "value": f"{len(qualified)} qualified, {len(decision.excluded)} ruled out",
            "detail": (
                "Each qualifying model was priced for this exact request — "
                "including what the cache would cost it, or save it."
            ),
        },
        {
            "n": 4,
            "title": "Why this one",
            "value": verdict,
            "detail": why,
        },
        {
            "n": 5,
            "title": "What it costs",
            "value": f"{_fmt(decision.estimated_cost_usd)} estimated",
            "detail": f"{cache_title}. {cache_body}",
        },
    ]

    return {
        "headline": (
            f"Sent to {model.key} — {verdict.lower()}."
            if not decision.pinned
            else f"Sent to {model.key} because you pinned it."
        ),
        "steps": steps,
        "excluded": [
            {**e, "plain": _humanise_exclusion(e["reason"])} for e in decision.excluded
        ],
        "technical": decision.reason,
    }
