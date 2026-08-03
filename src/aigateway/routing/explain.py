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

# How each exclusion category reads, and what it is really saying.
#
# The important one is `tier`. A model kept out by the tier floor is not
# "incapable" — it is a smaller model than this *particular task* was judged to
# need. Saying "not capable enough" is both inaccurate and a slur on a model
# that may be the right answer for the next request. State the rule instead.
EXCLUSION_LABEL = {
    "tier": "Below the required intelligence tier",
    "capacity": "Not enough capacity for this request",
    "capability": "Missing a capability this request needs",
    "no_credentials": "No credentials configured",
    "unhealthy": "Out of rotation — failing",
    "switched_off": "Switched off by you",
}


def _humanise_exclusion(entry: dict) -> str:
    """Say precisely why, in terms of the rule that actually applied."""
    reason, kind = entry.get("reason", ""), entry.get("kind", "")
    tier, required = entry.get("tier", "?"), entry.get("required_tier", "?")

    if kind == "tier":
        # Factual: this model's tier, and the tier the task calls for.
        return (
            f"{tier}-tier model; this task was judged to need {required} or better"
        )
    if reason.startswith("context "):
        return "context window too small to hold this prompt"
    if reason.startswith("max_output "):
        return "cannot produce a response this long"
    if reason == "no tool support":
        return "does not support tool calling, which this request uses"
    if reason == "no structured outputs":
        return "cannot guarantee the JSON shape this request asks for"
    if reason == "provider not configured":
        return f"no API key configured for {entry.get('provider', 'this vendor')}"
    if reason == "circuit open (unhealthy)":
        return "taken out of rotation — it has been failing recently"
    if "switched off" in reason:
        return (
            f"vendor {entry.get('provider', '')} switched off by you"
            if reason.startswith("vendor ")
            else "switched off by you"
        )
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
    # Naming the dimension matters: "cheapest" and "only one left" and "already
    # warm" are different kinds of answer, and conflating them hides which lever
    # you would pull to change the outcome.
    decided_on = "cost"
    if decision.pinned:
        decided_on = "pin"
        verdict = "You pinned this model"
        why = "The request named a specific model, so the router was bypassed entirely."
    elif decision.degraded:
        decided_on = "budget"
        verdict = "Budget forced a smaller model"
        why = (
            "This tenant is over its spending limit, so the router was capped to "
            "the cheapest tier that could still do the job."
        )
    elif decision.sticky:
        decided_on = "cache"
        verdict = "Kept the model this session was already using"
        why = (
            f"This session already warmed up {model.key}. Switching to another model "
            "would throw that cached context away and pay to build it again — usually "
            "more than the cheaper per-token price would save."
        )
    elif len(qualified) == 1:
        decided_on = "availability"
        verdict = "It was the only model that qualified"
        why = "Every other model was ruled out; see below."
    elif any(c.quality_multiplier > 1.0 for c in qualified):
        penalised = [c for c in qualified if c.quality_multiplier > 1.0]
        worst = max(penalised, key=lambda c: c.quality_multiplier)
        decided_on = "quality"
        verdict = "Cheapest once quality was taken into account"
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

    # --- the qualified field, across vendors ------------------------------
    # "Which models could have done this, and what separated them" is the
    # question a routing decision should answer. Listing only the winner, or
    # only the losers, answers neither half.
    ranked = sorted(qualified, key=lambda c: c.cost_usd)
    cheapest = ranked[0].cost_usd if ranked else 0.0
    qualified_rows = []
    for c in ranked:
        chosen = c.model.key == model.key
        if chosen:
            note = "selected"
        elif c.quality_multiplier > 1.0 and c.raw_cost_usd < decision.estimated_cost_usd:
            note = (
                f"cheaper per token, but {c.quality_multiplier:.1f}× penalty from "
                f"observed quality on '{intent}'"
            )
        elif cheapest and c.cost_usd > cheapest:
            note = f"{c.cost_usd / cheapest:.1f}× the cost of the cheapest option"
        else:
            note = "equally priced"
        qualified_rows.append(
            {
                "model": c.model.key,
                "provider": c.model.provider,
                "tier": c.model.tier.name.lower(),
                "cost_usd": round(c.cost_usd, 6),
                "raw_cost_usd": round(c.raw_cost_usd, 6),
                "cache_state": c.cache_state,
                "quality_multiplier": round(c.quality_multiplier, 3),
                "quality_samples": c.quality_samples,
                "chosen": chosen,
                "note": note,
            }
        )

    vendors = sorted({c.model.provider for c in qualified})

    # --- the dimensions that were actually weighed ------------------------
    dimensions = [
        {
            "name": "Intelligence required",
            "value": f"{tier.name.lower()} tier or better",
            "detail": f"Set by the intent '{intent}'. "
                      f"{len(decision.excluded)} model(s) fell outside it or were unavailable.",
        },
        {
            "name": "Availability",
            "value": f"{len(qualified)} model(s) across {len(vendors)} vendor(s)",
            "detail": "Credentials present, circuit closed, not switched off, and "
                      "large enough context for this prompt."
                      + (f" Vendors in play: {', '.join(vendors)}." if vendors else ""),
        },
        {
            "name": "Cost",
            "value": _fmt(decision.estimated_cost_usd),
            "detail": "Priced for this exact request — token counts, the vendor's "
                      "rate at this context size, and what the cache costs or saves.",
        },
        {
            "name": "Cache economics",
            "value": cache_title,
            "detail": cache_body,
        },
        {
            "name": "Observed quality",
            "value": (
                "no penalties in play"
                if all(c.quality_multiplier == 1.0 for c in qualified)
                else "penalties applied"
            ),
            "detail": "Models that recently failed this kind of task are priced at "
                      "their expected cost — succeeding half the time means about "
                      "twice the attempts, so about twice the price.",
        },
    ]

    return {
        "headline": (
            f"Sent to {model.key} — {verdict.lower()}."
            if not decision.pinned
            else f"Sent to {model.key} because you pinned it."
        ),
        "decided_on": decided_on,
        "steps": steps,
        "qualified": qualified_rows,
        "vendors_in_play": vendors,
        "dimensions": dimensions,
        "excluded": [{**e, "plain": _humanise_exclusion(e)} for e in decision.excluded],
        "technical": decision.reason,
    }
