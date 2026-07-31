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

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

from .policy import INTENT_POLICY

log = logging.getLogger(__name__)

KNOWN_INTENTS = sorted(INTENT_POLICY.keys())


@dataclass
class IntentResult:
    intent: str
    confidence: float
    source: str  # "declared" | "rules" | "embedding" | "llm" | "default"
    rationale: str = ""


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float] | None: ...


class NullEmbedder:
    """Abstains. Replace with a real embedder to enable L2."""

    async def embed(self, text: str) -> list[float] | None:
        return None


# --------------------------------------------------------------------------
# L1: deterministic rules
# --------------------------------------------------------------------------
_KEYWORDS: list[tuple[str, re.Pattern[str]]] = [
    ("code_review", re.compile(r"\b(review|audit|find bugs?|code smell|vulnerab)", re.I)),
    ("hard_debug", re.compile(r"\b(debug|stack ?trace|flaky|race condition|segfault)", re.I)),
    ("architecture", re.compile(r"\b(architect|design doc|trade-?off|system design)", re.I)),
    (
        "code_write",
        re.compile(r"\b(implement|refactor|write (a |the )?(function|class|test))", re.I),
    ),
    ("summarize", re.compile(r"\b(summari[sz]e|tl;?dr|condense|key points)", re.I)),
    ("translate", re.compile(r"\btranslate\b", re.I)),
    ("classify", re.compile(r"\b(classif|categor|label this|sentiment)", re.I)),
    ("extract", re.compile(r"\b(extract|parse out|pull the fields)", re.I)),
    ("plan", re.compile(r"\b(plan|break (this )?down|steps to|roadmap)", re.I)),
]

# Below this, a request is too small to be doing anything heavy.
_SHORT_REQUEST_TOKENS = 400
# Above this, it is carrying enough context that light-tier is a bad bet.
_LARGE_CONTEXT_TOKENS = 25_000


def _last_user_text(canonical) -> str:
    for msg in reversed(canonical.messages):
        if msg.role == "user":
            if isinstance(msg.content, str):
                return msg.content
            if isinstance(msg.content, list):
                return " ".join(
                    p.get("text", "") for p in msg.content if isinstance(p, dict)
                )
    return ""


def classify_by_rules(canonical, prefix_tokens: int, volatile_tokens: int) -> IntentResult | None:
    total = prefix_tokens + volatile_tokens
    text = _last_user_text(canonical)

    # A response schema plus a short prompt is extraction, near-definitionally.
    if canonical.response_schema and total < _SHORT_REQUEST_TOKENS * 4:
        return IntentResult("extract", 0.9, "rules", "response schema + short prompt")

    # Many tools in play means orchestration, whatever the prose says.
    if len(canonical.tools) >= 5:
        return IntentResult(
            "tool_orchestration", 0.8, "rules", f"{len(canonical.tools)} tools declared"
        )

    for intent, pattern in _KEYWORDS:
        if pattern.search(text):
            # Keyword hits are suggestive, not conclusive — confidence is set
            # below the LLM-escalation threshold when the request is large,
            # because a big-context request labelled "summarize" may still be
            # doing something much harder.
            conf = 0.75 if total < _LARGE_CONTEXT_TOKENS else 0.55
            return IntentResult(intent, conf, "rules", f"keyword match: {intent}")

    if total < _SHORT_REQUEST_TOKENS and not canonical.tools:
        return IntentResult("chat", 0.7, "rules", "short, tool-free request")

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
    ):
        self._store = store
        self._registry = provider_registry
        self._enabled = enabled
        self._model_key = model_key
        self._min_confidence = min_confidence
        self._embedder = embedder or NullEmbedder()

    async def classify(
        self, canonical, prefix_tokens: int, volatile_tokens: int
    ) -> IntentResult:
        # L0 — the caller told us.
        if canonical.intent_hint:
            hint = canonical.intent_hint.lower()
            if hint in INTENT_POLICY:
                return IntentResult(hint, 1.0, "declared", "caller-declared intent")
            log.warning("unknown declared intent %r; falling through", hint)

        # L1 — rules.
        result = classify_by_rules(canonical, prefix_tokens, volatile_tokens)
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

        key = "intent:" + hashlib.sha256(text.encode()).hexdigest()[:24]
        if cached := await self._store.get(key):
            data = json.loads(cached)
            return IntentResult(data["intent"], data["confidence"], "llm-cached")

        try:
            provider = self._registry.for_model(self._model_key)
            raw = await provider.classify(
                model_key=self._model_key,
                system=_CLASSIFIER_SYSTEM,
                text=text,
                schema=_CLASSIFIER_SCHEMA,
            )
        except Exception as exc:
            # A classifier failure must never fail the request it was labelling.
            log.warning("llm classifier failed: %s", exc)
            return None

        if not raw or raw.get("intent") not in INTENT_POLICY:
            return None

        await self._store.set(key, json.dumps(raw), ttl=3600)
        return IntentResult(raw["intent"], float(raw.get("confidence", 0.7)), "llm")
