"""Intent classification — the layer everything else is downstream of.

Every routing decision, every cost estimate and every tier floor starts from a
label produced here, so a wrong label is not a wrong label: it is the wrong
model, at the wrong price, for the rest of the request. This file did not exist
until the classifier was reworked, which is its own finding.

The tests are organised by layer, and most of them are about the ways a cheap
classifier is confidently wrong rather than the happy path.
"""

from __future__ import annotations

import pytest
from conftest import make_request

from aigateway.catalog import Tier
from aigateway.routing.intent import (
    IntentClassifier,
    classify_by_rules,
    conversation_scores,
    score_keywords,
)
from aigateway.routing.policy import INTENT_POLICY
from aigateway.schemas import ChatMessage, ToolDef


class StubProvider:
    """Stands in for a vendor adapter. Only `classify` is ever called here."""

    def __init__(self):
        self.calls = []

    async def classify(self, **kw):
        self.calls.append(kw)
        return {"intent": "chat", "confidence": 0.9}


class StubRegistry:
    """The classifier needs exactly two things: is anything configured, and
    which adapter serves this model."""

    def __init__(self):
        self.provider = StubProvider()

    @property
    def enabled(self):
        return {"anthropic", "openai"}

    def for_model(self, model_key: str):
        return self.provider

    def all(self):
        return [self.provider]


@pytest.fixture
def registry() -> StubRegistry:
    return StubRegistry()


def req(*user_texts, tools=0, schema=None, hint=None, assistant_between=True):
    """A canonical request whose user turns are `user_texts`, oldest first."""
    canonical = make_request()
    messages = []
    for i, text in enumerate(user_texts):
        messages.append(ChatMessage(role="user", content=text))
        if assistant_between and i < len(user_texts) - 1:
            messages.append(ChatMessage(role="assistant", content="Understood."))
    canonical.messages = messages
    canonical.tools = [
        ToolDef(name=f"t{i}", description="", parameters={}) for i in range(tools)
    ]
    canonical.response_schema = schema
    canonical.intent_hint = hint
    return canonical


def rules(*args, prefix=100, volatile=100, **kw):
    return classify_by_rules(req(*args, **kw), prefix, volatile)


# ==========================================================================
# L1 — keyword scoring
# ==========================================================================
def test_every_pattern_is_scored_not_just_the_first_that_matches():
    """List order used to decide the answer. Nobody reading a list of patterns
    knows that its order is load-bearing, which is what made it a bug."""
    scores = score_keywords("review this stack trace and debug the race condition")
    assert "code_review" in scores and "hard_debug" in scores
    assert scores["hard_debug"] > scores["code_review"], (
        "specific evidence should beat a generic word, whatever the list order"
    )


def test_the_contested_case_resolves_to_the_stronger_evidence():
    result = rules("Please review this stack trace — there is a race condition.")
    assert result.intent == "hard_debug"


def test_a_decisive_phrase_outweighs_a_generic_word():
    """You review a diff, a document, a plan or a decision. 'Race condition' is
    only ever one thing."""
    assert score_keywords("race condition")["hard_debug"] > score_keywords("review")["code_review"]


def test_repeated_evidence_accumulates():
    once = score_keywords("summarize this")["summarize"]
    twice = score_keywords("summarize this, then summarize that")["summarize"]
    assert twice > once


def test_a_negated_keyword_does_not_count():
    """'Do not summarise' matched `summarize` before this existed."""
    assert "summarize" not in score_keywords("Do not summarize; give me the full text.")
    assert "summarize" not in score_keywords("Answer in full without summarizing.")


def test_a_plain_keyword_still_counts():
    assert score_keywords("Summarize this thread")["summarize"] > 0


# ==========================================================================
# L1 — evidence across turns
# ==========================================================================
def test_a_continuation_inherits_the_intent_of_the_conversation():
    """The case that matters most and the one a last-message classifier gets
    wrong every time: 'yes, do that' carries no signal, and the request it
    continues was a code review two turns ago."""
    scores = conversation_scores(
        req("Review this authentication middleware for vulnerabilities.", "yes, do that")
    )
    assert scores.get("code_review", 0) > 0


def test_recent_turns_outweigh_older_ones():
    """The conversation has moved on — older evidence fades rather than counting
    equally, or a session could never change subject."""
    scores = conversation_scores(req("Review this diff for vulnerabilities.", "Translate it."))
    assert scores["translate"] > scores["code_review"]


def test_evidence_does_not_reach_back_forever():
    """Only the recent window contributes; an hour-old topic is not this task."""
    scores = conversation_scores(
        req("Translate this.", "ok", "and now?", "Summarize the result.")
    )
    assert "translate" not in scores


def test_a_signal_free_conversation_produces_no_keyword_evidence():
    assert conversation_scores(req("yes", "go on", "ok")) == {}


# ==========================================================================
# L1 — shape rules, which outrank prose
# ==========================================================================
def test_a_schema_on_a_short_prompt_is_extraction():
    result = rules("Pull out the fields.", schema={"type": "object"})
    assert result.intent == "extract"
    assert result.confidence >= 0.9


def test_a_schema_on_a_large_request_is_not_assumed_to_be_extraction():
    """A schema-shaped request carrying 30k tokens is doing more than filling
    in fields, and calling it `extract` would floor it at the light tier."""
    result = classify_by_rules(req("Return JSON.", schema={"type": "object"}), 30_000, 500)
    assert result is None or result.intent != "extract"


def test_many_tools_means_orchestration_whatever_the_prose_says():
    result = rules("Summarize this.", tools=6)
    assert result.intent == "tool_orchestration"


def test_a_short_tool_free_request_is_chat():
    assert rules("hello there").intent == "chat"


def test_a_large_context_with_tools_is_long_horizon():
    result = classify_by_rules(req("Carry on.", tools=2), 30_000, 500)
    assert result.intent == "long_horizon_agentic"


def test_no_signal_at_all_abstains():
    """Abstaining hands the decision to a layer that might actually know."""
    assert classify_by_rules(req("qwerty asdf"), 5_000, 100) is None


# ==========================================================================
# L1 — confidence, which decides whether L3 gets a look
# ==========================================================================
MIN_CONFIDENCE = 0.6


def test_a_close_contest_escalates_rather_than_guessing():
    """Two intents neck and neck is exactly when a cheap label is most likely
    wrong, so it must fall below the escalation threshold."""
    result = rules("Review the plan and plan the review.")
    assert result.confidence < MIN_CONFIDENCE
    assert "contested" in result.rationale


def test_a_large_request_never_settles_on_keywords_alone():
    """A big-context prompt saying 'summarize' may be doing something far
    harder, and light-tier is an expensive place to be wrong."""
    result = classify_by_rules(req("Summarize this."), 30_000, 500)
    assert result.confidence < MIN_CONFIDENCE


def test_clear_uncontested_evidence_is_confident_enough_to_stop():
    result = rules("Translate this paragraph into French.")
    assert result.intent == "translate"
    assert result.confidence >= MIN_CONFIDENCE


def test_confidence_never_exceeds_the_ceiling_for_keyword_evidence():
    """Keywords are suggestive, never conclusive — no pile of them should reach
    the certainty of a declared intent."""
    result = rules("translate translate translate translate translate")
    assert result.confidence <= 0.75


# ==========================================================================
# L0 — the declared hint, trusted but checked
# ==========================================================================
@pytest.fixture
def classifier(store, registry):
    return IntentClassifier(store, registry, enabled=False, min_confidence=MIN_CONFIDENCE)


async def test_a_declared_intent_is_accepted(classifier):
    result = await classifier.classify(req("anything", hint="translate"), 100, 100)
    assert result.intent == "translate"
    assert result.source == "declared"
    assert result.confidence == 1.0


async def test_an_unknown_declared_intent_falls_through(classifier):
    result = await classifier.classify(req("Translate this.", hint="not-an-intent"), 100, 100)
    assert result.source != "declared"
    assert result.intent == "translate"


async def test_a_declaration_that_understates_the_work_is_overridden(classifier):
    """The cheapest possible mistake and the easiest one to make: a template
    that labels everything `classify` gets a light model for a hard debug, and
    without this nothing anywhere says so."""
    result = await classifier.classify(
        req("There is a race condition and a deadlock in this stack trace.", hint="classify"),
        100,
        100,
    )
    assert result.intent == "hard_debug"
    assert result.source == "declared-overridden"
    # It has to say what it did and how to insist.
    assert "classify" in result.rationale
    assert "max_tier" in result.rationale or "pin_model" in result.rationale


async def test_a_declaration_that_overstates_the_work_is_left_alone(classifier):
    """Declaring something heavier than the evidence is the caller spending
    their own money on caution. That is their call, not ours."""
    result = await classifier.classify(req("Translate this.", hint="architecture"), 100, 100)
    assert result.intent == "architecture"
    assert result.source == "declared"


async def test_weak_contrary_evidence_does_not_overrule_the_caller(classifier):
    """The caller usually knows best. Only evidence strong enough that we would
    have escalated on it anyway is grounds for overriding."""
    result = await classifier.classify(
        req("Have a quick look at this plan.", hint="classify"), 100, 100
    )
    assert result.source == "declared"
    assert result.intent == "classify"


async def test_an_override_only_ever_moves_up_a_tier(classifier):
    """The check exists to protect quality, not to second-guess spend."""
    for hint in INTENT_POLICY:
        result = await classifier.classify(
            req("Refactor this function and write the tests.", hint=hint), 100, 100
        )
        if result.source == "declared-overridden":
            assert INTENT_POLICY[result.intent].min_tier > INTENT_POLICY[hint].min_tier


# ==========================================================================
# L3 — the small-model call
# ==========================================================================
async def test_the_llm_layer_is_skipped_when_the_rules_are_confident(store, registry):
    """L3 costs money and latency on traffic you were trying to make cheaper."""
    calls = []
    classifier = IntentClassifier(store, registry, min_confidence=MIN_CONFIDENCE)

    async def spy(**kw):
        calls.append(kw)
        return {"intent": "chat", "confidence": 0.9}

    for provider in registry.all():
        provider.classify = spy

    await classifier.classify(req("Translate this paragraph into French."), 100, 100)
    assert calls == []


async def test_a_classifier_failure_never_fails_the_request(store, registry):
    classifier = IntentClassifier(store, registry, min_confidence=MIN_CONFIDENCE)

    async def boom(**kw):
        raise RuntimeError("classifier is down")

    for provider in registry.all():
        provider.classify = boom

    result = await classifier.classify(req("qwerty asdf zxcv"), 5_000, 100)
    assert result.source == "default"
    assert result.intent == "unknown"


async def test_the_cache_key_changes_with_the_classifier_model(store, registry):
    """A label cached against one model must not be served for another."""
    keys = []

    async def label(**kw):
        return {"intent": "chat", "confidence": 0.9}

    for provider in registry.all():
        provider.classify = label

    original_set = store.set

    async def spy_set(key, value, ttl=None):
        keys.append(key)
        return await original_set(key, value, ttl)

    store.set = spy_set

    for model in ("claude-haiku-4-5", "gpt-5-nano"):
        c = IntentClassifier(store, registry, model_key=model, min_confidence=MIN_CONFIDENCE)
        await c.classify(req("qwerty asdf zxcv"), 5_000, 100)

    assert len(set(keys)) == 2, "same text, different model, must not share a cache entry"


async def test_the_cache_key_changes_with_the_taxonomy(store, registry):
    """Changing the label set must invalidate cached labels rather than leaving
    stale ones to expire on their own schedule."""
    from aigateway.routing import intent as intent_module

    async def label(**kw):
        return {"intent": "chat", "confidence": 0.9}

    for provider in registry.all():
        provider.classify = label

    c = IntentClassifier(store, registry, min_confidence=MIN_CONFIDENCE)
    await c.classify(req("qwerty asdf zxcv"), 5_000, 100)
    before = [k for k in store._data if k.startswith("intent:")]

    original = intent_module._TAXONOMY_VERSION
    try:
        intent_module._TAXONOMY_VERSION = "different"
        await c.classify(req("qwerty asdf zxcv"), 5_000, 100)
    finally:
        intent_module._TAXONOMY_VERSION = original

    after = [k for k in store._data if k.startswith("intent:")]
    assert len(after) == len(before) + 1


async def test_a_cached_label_says_it_was_cached(store, registry):
    async def label(**kw):
        return {"intent": "chat", "confidence": 0.9}

    for provider in registry.all():
        provider.classify = label

    c = IntentClassifier(store, registry, min_confidence=MIN_CONFIDENCE)
    await c.classify(req("qwerty asdf zxcv"), 5_000, 100)
    second = await c.classify(req("qwerty asdf zxcv"), 5_000, 100)

    assert second.source == "llm-cached"
    assert second.rationale, "a label with no explanation cannot be audited"


async def test_a_label_outside_the_taxonomy_is_rejected(store, registry):
    async def nonsense(**kw):
        return {"intent": "make_coffee", "confidence": 0.99}

    for provider in registry.all():
        provider.classify = nonsense

    c = IntentClassifier(store, registry, min_confidence=MIN_CONFIDENCE)
    result = await c.classify(req("qwerty asdf zxcv"), 5_000, 100)
    assert result.intent in INTENT_POLICY


# ==========================================================================
# The contract every layer shares
# ==========================================================================
async def test_every_result_is_a_known_intent_with_a_reason(classifier):
    """A label the router cannot price, or that nobody can explain, is useless."""
    for canonical, p, v in [
        (req("Translate this."), 100, 100),
        (req("qwerty"), 5_000, 100),
        (req("Summarize.", hint="summarize"), 100, 100),
        (req("Review this.", tools=6), 100, 100),
        (req("Anything"), 30_000, 500),
    ]:
        result = await classifier.classify(canonical, p, v)
        assert result.intent in INTENT_POLICY
        assert 0.0 <= result.confidence <= 1.0
        assert result.source
        assert result.rationale, f"{result.source} produced no rationale"


def test_the_taxonomy_and_the_policy_table_cannot_drift():
    """The classifier's label set and the router's policy table are the same
    thing seen from two sides; a label with no policy has no tier floor."""
    from aigateway.routing.intent import KNOWN_INTENTS

    assert set(KNOWN_INTENTS) == set(INTENT_POLICY)


def test_the_fallback_intent_is_never_the_cheapest_tier():
    """Abstention must not be a discount. If the classifier does not know what
    the work is, guessing light is the expensive kind of wrong."""
    assert INTENT_POLICY["unknown"].min_tier >= Tier.STANDARD
