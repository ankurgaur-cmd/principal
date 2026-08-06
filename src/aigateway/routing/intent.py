"""Layered intent classification.

Cost discipline is the whole point: an LLM classifier on *every* request adds a
latency floor and a per-call cost floor to traffic you were trying to make
cheaper. So the layers run cheapest-first and stop as soon as one is confident:

  L0  declared hint      free       the caller usually already knows
  L1  deterministic rules free       schema present, tool count, length, keywords
  L2  embedding classifier ~free     nearest-neighbour over labelled exemplars
  L3  small-model call    cheap      only on low confidence

In practice L0+L1 should absorb the large majority of traffic. Watch the
``intent_source`` field in the records: if L3 is firing often, your rules or
exemplars need work, not your budget.

L2 ships as a null implementation on purpose — plugging in an embedding
provider is a deployment decision, and a stub that silently returns garbage is
worse than one that abstains.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

from .policy import INTENT_POLICY

log = logging.getLogger(__name__)

KNOWN_INTENTS = sorted(INTENT_POLICY.keys())

# Fingerprint of the label set. Part of the L3 cache key, so changing the
# taxonomy invalidates every cached label rather than leaving stale ones to
# expire on their own schedule.
_TAXONOMY_VERSION = hashlib.sha256(",".join(KNOWN_INTENTS).encode()).hexdigest()[:8]


@dataclass
class IntentResult:
    intent: str
    confidence: float
    # declared | declared-overridden | rules | embedding | llm | llm-cached | default
    source: str
    rationale: str = ""
    # True when this label came from an *absence* of signal rather than
    # evidence for it. "Short and tool-free, so probably chat" is a sensible
    # default and a terrible reason to overrule a caller who said otherwise.
    fallback: bool = False


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float] | None: ...


class NullEmbedder:
    """Abstains. Replace with a real embedder to enable L2."""

    async def embed(self, text: str) -> list[float] | None:
        return None


# --------------------------------------------------------------------------
# L1: deterministic rules
# --------------------------------------------------------------------------
# Keyword evidence. Every pattern is tested and the matches are *scored* — the
# first version returned on first match, which made list order silently decide
# the answer: "review this stack trace and debug the race condition" matched
# `code_review` because it happened to be listed first, and `hard_debug` never
# got a look in. Order is a terrible tie-break because nobody reading the list
# knows it is one.
#
# `weight` is how much one hit is worth. Phrases that only ever appear in one
# kind of request score higher than words that show up everywhere: "race
# condition" is decisive, "review" is not — you review a diff, a document, a
# plan, or a decision.
_KEYWORDS: list[tuple[str, re.Pattern[str], float]] = [
    ("code_review", re.compile(r"\breview\b", re.I), 0.6),
    (
        "code_review",
        re.compile(r"\b(audit|find bugs?|code smell|vulnerab|security|injection|xss)", re.I),
        1.0,
    ),
    ("hard_debug", re.compile(r"\bdebug\b", re.I), 0.8),
    (
        "hard_debug",
        re.compile(r"\b(stack ?trace|flaky|race condition|deadlock|segfault|heisenbug)", re.I),
        1.2,
    ),
    ("architecture", re.compile(r"\b(architect|design doc|system design)", re.I), 1.2),
    ("architecture", re.compile(r"\btrade-?offs?\b", re.I), 0.8),
    ("code_write", re.compile(r"\b(implement|refactor)\b", re.I), 0.9),
    ("code_write", re.compile(r"\bwrite (a |the )?(function|class|test|script)", re.I), 1.2),
    ("summarize", re.compile(r"\b(summari[sz]e|tl;?dr|condense|key points)", re.I), 1.2),
    ("translate", re.compile(r"\btranslate\b", re.I), 1.4),
    ("classify", re.compile(r"\b(classif|categor|label this|sentiment)", re.I), 1.2),
    ("extract", re.compile(r"\b(extract|parse out|pull the fields)", re.I), 1.2),
    ("plan", re.compile(r"\b(break (this )?down|steps to|roadmap)", re.I), 1.0),
    ("plan", re.compile(r"\bplan\b", re.I), 0.7),
]

# Phrases that mean the *opposite* of what the keyword suggests. "Do not
# summarise" matched `summarize` before this existed.
_NEGATIONS = re.compile(
    r"\b(do ?n[o']?t|no need to|without|avoid|rather than|instead of)\s+$", re.I
)


def score_keywords(text: str) -> dict[str, float]:
    """Total keyword evidence per intent. Every pattern is tried."""
    scores: dict[str, float] = {}
    for intent, pattern, weight in _KEYWORDS:
        for match in pattern.finditer(text):
            # Look at the ~24 characters before the hit for a negation.
            if _NEGATIONS.search(text[max(0, match.start() - 24) : match.start()]):
                continue
            scores[intent] = scores.get(intent, 0.0) + weight
    return scores

# Overruling an explicit caller declaration is a stronger action than merely
# being confident enough to stop classifying, so it needs a higher bar. Without
# the gap, one weak keyword ("...this plan...") was enough to overturn a label
# a human had written down on purpose.
_OVERRIDE_CONFIDENCE = 0.7

# Below this, a request is too small to be doing anything heavy.
_SHORT_REQUEST_TOKENS = 400
# A schema-shaped request bigger than this is doing more than filling fields.
_SCHEMA_EXTRACTION_CEILING = 1600
# Above this, it is carrying enough context that light-tier is a bad bet.
_LARGE_CONTEXT_TOKENS = 25_000


def _text_of(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    if isinstance(msg.content, list):
        return " ".join(p.get("text", "") for p in msg.content if isinstance(p, dict))
    return ""


def _last_user_text(canonical) -> str:
    """The most recent user message alone."""
    for msg in reversed(canonical.messages):
        if msg.role == "user":
            return _text_of(msg)
    return ""


# How many recent user turns contribute evidence, and how fast older ones fade.
_EVIDENCE_TURNS = 3
_TURN_DECAY = 0.5


def user_turns(canonical, limit: int = _EVIDENCE_TURNS) -> list[str]:
    """The last `limit` user messages, most recent first."""
    turns = [_text_of(m) for m in reversed(canonical.messages) if m.role == "user"]
    return [t for t in turns[:limit] if t.strip()]


def conversation_scores(canonical) -> dict[str, float]:
    """Keyword evidence across recent turns, weighted towards the newest.

    Classifying on the last message alone is wrong in exactly the case that
    matters most — a multi-turn session. "yes, do that" and "keep going" carry
    no signal at all, and the request they continue was a code review three
    messages ago. Reading only the last turn sends that continuation to a light
    model.

    Older turns fade rather than counting equally: the conversation *has* moved
    on, so a review two turns back is weaker evidence than a translation right
    now, but it is not zero evidence.
    """
    totals: dict[str, float] = {}
    for depth, text in enumerate(user_turns(canonical)):
        decay = _TURN_DECAY**depth
        for intent, score in score_keywords(text).items():
            totals[intent] = totals.get(intent, 0.0) + score * decay
    return totals


def classify_by_rules(canonical, prefix_tokens: int, volatile_tokens: int) -> IntentResult | None:
    total = prefix_tokens + volatile_tokens

    # A response schema plus a short prompt is extraction, near-definitionally.
    if canonical.response_schema and total < _SCHEMA_EXTRACTION_CEILING:
        return IntentResult("extract", 0.9, "rules", "response schema + short prompt")

    # Many tools in play means orchestration, whatever the prose says.
    if len(canonical.tools) >= 5:
        return IntentResult(
            "tool_orchestration", 0.8, "rules", f"{len(canonical.tools)} tools declared"
        )

    # Keyword evidence, scored across recent turns rather than first-match-wins.
    scores = conversation_scores(canonical)
    if scores:
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

        # Keyword hits are suggestive, not conclusive. Confidence drops below
        # the escalation threshold in two cases, so L3 gets a look: a large
        # request (a big-context prompt labelled "summarize" may be doing
        # something much harder), and a close contest between two intents,
        # which is precisely when a cheap label is most likely to be wrong.
        contested = runner_up >= best_score * 0.75
        if total >= _LARGE_CONTEXT_TOKENS:
            conf = 0.55
        elif contested:
            conf = 0.5
        else:
            conf = min(0.75, 0.55 + 0.1 * best_score)

        detail = f"keyword evidence: {best} ({best_score:.1f})"
        if contested:
            detail += f", contested by {ranked[1][0]} ({runner_up:.1f})"
        return IntentResult(best, conf, "rules", detail)

    if total < _SHORT_REQUEST_TOKENS and not canonical.tools:
        return IntentResult(
            "chat", 0.7, "rules", "short, tool-free request", fallback=True
        )

    if total > _LARGE_CONTEXT_TOKENS and canonical.tools:
        return IntentResult(
            "long_horizon_agentic", 0.7, "rules", "large context with tools"
        )

    return None


# --------------------------------------------------------------------------
# L3: small-model classifier
# --------------------------------------------------------------------------
_CLASSIFIER_SYSTEM = (
    "You label API requests with a single intent for a model router. "
    "Reply with the intent and a confidence between 0 and 1. "
    "Choose the cheapest intent that plausibly covers the request — "
    "over-labelling wastes money on an unnecessarily large model.\n"
    "Valid intents: " + ", ".join(KNOWN_INTENTS)
)

_CLASSIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": KNOWN_INTENTS},
        "confidence": {"type": "number"},
    },
    "required": ["intent", "confidence"],
    "additionalProperties": False,
}


class IntentClassifier:
    def __init__(
        self,
        store,
        provider_registry,
        *,
        enabled: bool = True,
        model_key: str = "claude-haiku-4-5",
        min_confidence: float = 0.6,
        embedder: Embedder | None = None,
        timeout_seconds: float = 4.0,
    ):
        self._store = store
        self._registry = provider_registry
        self._enabled = enabled
        self._model_key = model_key
        self._min_confidence = min_confidence
        self._embedder = embedder or NullEmbedder()
        self._timeout_seconds = timeout_seconds

    async def classify(
        self, canonical, prefix_tokens: int, volatile_tokens: int
    ) -> IntentResult:
        # L0 — the caller told us. Trusted, but checked.
        #
        # A declared hint used to be accepted at confidence 1.0 with no scrutiny
        # at all, which quietly contradicted the documented contract ("verified,
        # and overridable, by the gateway") and made the cheapest possible
        # mistake also the easiest one: an agent that labels a code review as
        # `classify` — by a copy-paste, a stale constant, or a template that
        # never got updated — gets a light model and a bad answer, and nothing
        # anywhere says so.
        #
        # The check is deliberately narrow. The caller usually *does* know best,
        # so the hint is only overridden when the request's own shape contradicts
        # it in the one direction that costs quality: the declared intent asks
        # for a smaller model than the evidence supports. Declaring something
        # *heavier* than the evidence is left alone — that is the caller
        # spending their own money on caution, which is their call.
        if canonical.intent_hint:
            hint = canonical.intent_hint.lower()
            if hint in INTENT_POLICY:
                return self._verify_declared(hint, canonical, prefix_tokens, volatile_tokens)
            log.warning("unknown declared intent %r; falling through", hint)

        # L1 — rules.
        result = classify_by_rules(canonical, prefix_tokens, volatile_tokens)

        # A pinned request has already decided which model serves it, and the
        # router bypasses intent-based selection entirely for it. Paying a full
        # upstream round trip to L3 would buy nothing but a nicer label for the
        # reputation bucket — measured at ~1.5s, on 100% of pinned traffic.
        if canonical.pin_model:
            return result or IntentResult(
                "unknown", 0.3, "default", "pinned request; intent not needed to route"
            )
        if result and result.confidence >= self._min_confidence:
            return result

        # L2 — embeddings (abstains unless an embedder is wired up).
        embedded = await self._classify_by_embedding(canonical)
        if embedded and embedded.confidence >= self._min_confidence:
            return embedded

        # L3 — small-model call, memoised by request shape. The registry check
        # is live: credentials can arrive after startup.
        if self._enabled and self._registry.enabled:
            llm = await self._classify_by_llm(canonical)
            if llm:
                return llm

        return result or IntentResult(
            "unknown", 0.3, "default", "no layer produced a confident label"
        )

    def _verify_declared(
        self, hint: str, canonical, prefix_tokens: int, volatile_tokens: int
    ) -> IntentResult:
        """Accept the caller's label unless the request contradicts it downward."""
        declared_tier = INTENT_POLICY[hint].min_tier
        evidence = classify_by_rules(canonical, prefix_tokens, volatile_tokens)

        # No evidence, or only a structural default, is no contradiction at all.
        if evidence is None or evidence.fallback:
            return IntentResult(hint, 1.0, "declared", "caller-declared intent")

        evidence_tier = INTENT_POLICY[evidence.intent].min_tier
        if evidence_tier <= declared_tier:
            # Agrees, or the caller asked for more headroom than we would.
            return IntentResult(
                hint,
                1.0,
                "declared",
                f"caller-declared intent, consistent with request shape "
                f"({evidence.rationale})",
            )

        # The request looks heavier than the label. Two guards before
        # overruling a human: the evidence must be positive rather than a
        # structural default, and it must clear a bar higher than ordinary
        # classification confidence.
        if evidence.fallback or evidence.confidence < _OVERRIDE_CONFIDENCE:
            return IntentResult(
                hint,
                0.8,
                "declared",
                f"caller-declared intent, kept despite weaker signs of "
                f"{evidence.intent} ({evidence.rationale})",
            )

        log.info(
            "declared intent %r overridden: request looks like %r (%s)",
            hint, evidence.intent, evidence.rationale,
        )
        return IntentResult(
            evidence.intent,
            evidence.confidence,
            "declared-overridden",
            f"you declared '{hint}' ({declared_tier.name.lower()} tier), but the "
            f"request looks like {evidence.intent} needing "
            f"{evidence_tier.name.lower()} — {evidence.rationale}. Routed on the "
            f"evidence; pin_model or max_tier if you meant it.",
        )

    async def _classify_by_embedding(self, canonical) -> IntentResult | None:
        vector = await self._embedder.embed(_last_user_text(canonical)[:2000])
        if vector is None:
            return None
        # Wire your labelled-exemplar index in here. Returning None keeps the
        # layer honest until that exists.
        return None

    async def _classify_by_llm(self, canonical) -> IntentResult | None:
        text = _last_user_text(canonical)[:2000]
        if not text:
            return None

        # The cache key carries everything that could change the answer, not
        # just the text. Keyed on text alone, a stale label survived for an hour
        # after the taxonomy or the classifier model changed — and a label that
        # is quietly wrong for an hour is worse than one recomputed for a cent.
        key = "intent:" + hashlib.sha256(
            "\x00".join((_TAXONOMY_VERSION, self._model_key, text)).encode()
        ).hexdigest()[:24]
        if cached := await self._store.get(key):
            data = json.loads(cached)
            return IntentResult(
                data["intent"],
                data["confidence"],
                "llm-cached",
                "a small model labelled this exact text earlier; reused that answer",
            )

        try:
            provider = self._registry.for_model(self._model_key)
            # A hard ceiling of our own. The classifier sits in front of the
            # call it is labelling, so a slow one delays every request and
            # inflates time-to-first-token — the number a drop-in gateway is
            # judged on. Better to fall back to the rules guess than to let a
            # degraded classifier set the floor for the whole gateway.
            raw = await asyncio.wait_for(
                provider.classify(
                    model_key=self._model_key,
                    system=_CLASSIFIER_SYSTEM,
                    text=text,
                    schema=_CLASSIFIER_SCHEMA,
                ),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            log.warning(
                "llm classifier exceeded %.1fs; falling back to the rules guess",
                self._timeout_seconds,
            )
            return None
        except Exception as exc:
            # A classifier failure must never fail the request it was labelling.
            log.warning("llm classifier failed: %s", exc)
            return None

        if not raw or raw.get("intent") not in INTENT_POLICY:
            return None

        await self._store.set(key, json.dumps(raw), ttl=3600)
        return IntentResult(
            raw["intent"],
            float(raw.get("confidence", 0.7)),
            "llm",
            f"{self._model_key} read the request and labelled it",
        )
