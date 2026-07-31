"""Canonicalisation, intent rules, and the replay harness."""

from __future__ import annotations

from conftest import make_request

from aigateway.pipeline import canonicalise
from aigateway.replay import replay
from aigateway.routing.intent import classify_by_rules
from aigateway.schemas import ChatCompletionRequest, ChatMessage, GatewayExtensions


def test_leading_system_messages_are_hoisted():
    """System content is the most stable part of the prompt and therefore the
    natural home for the first cache breakpoint. Left inline, the boundary is
    buried."""
    req = ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content="you are helpful"),
            ChatMessage(role="user", content="hi"),
        ]
    )
    canonical = canonicalise(req)
    assert canonical.system == ["you are helpful"]
    assert [m.role for m in canonical.messages] == ["user"]


def test_mid_conversation_system_message_keeps_its_position():
    """Hoisting it would change the prefix ahead of the whole conversation and
    invalidate every cached turn."""
    req = ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content="base"),
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="system", content="terse mode on"),
        ]
    )
    canonical = canonicalise(req)
    assert canonical.system == ["base"]
    assert [m.role for m in canonical.messages] == ["user", "system"]


def test_explicit_model_id_is_treated_as_a_pin():
    req = ChatCompletionRequest(
        model="claude-opus-5", messages=[ChatMessage(role="user", content="x")]
    )
    assert canonicalise(req).pin_model == "claude-opus-5"


def test_auto_model_delegates_to_the_router():
    req = ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="x")])
    assert canonicalise(req).pin_model is None


def test_gateway_extensions_survive_canonicalisation():
    req = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="x")],
        x_gateway=GatewayExtensions(
            session_id="s1", intent="code_review", effort="xhigh", cache_hints=["system"]
        ),
    )
    canonical = canonicalise(req)
    assert canonical.session_id == "s1"
    assert canonical.intent_hint == "code_review"
    assert canonical.effort == "xhigh"
    assert canonical.cache_hints == ["system"]


def test_tools_are_converted_from_openai_shape():
    req = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="x")],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "search the web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ],
    )
    canonical = canonicalise(req)
    assert len(canonical.tools) == 1
    assert canonical.tools[0].name == "search"


# -- intent rules ----------------------------------------------------------
def test_schema_plus_short_prompt_is_extraction():
    req = make_request(user_text="get the name")
    req.response_schema = {"type": "object"}
    result = classify_by_rules(req, 50, 20)
    assert result and result.intent == "extract"


def test_many_tools_implies_orchestration():
    result = classify_by_rules(make_request(tools=6), 200, 50)
    assert result and result.intent == "tool_orchestration"


def test_keyword_confidence_drops_on_large_context():
    """A 'summarize' keyword over 100k tokens of context may be doing something
    much harder than summarising, so it should not short-circuit the classifier."""
    small = classify_by_rules(make_request(user_text="summarize this"), 100, 50)
    large = classify_by_rules(make_request(user_text="summarize this"), 100_000, 50)
    assert small.confidence > large.confidence


# -- replay ----------------------------------------------------------------
def _record(**kwargs) -> dict:
    base = {
        "trace_id": "t",
        "outcome": "ok",
        "resolved_intent": "summarize",
        "chosen_model": "claude-haiku-4-5",
        "session_id": "s1",
        "prefix_tokens_est": 20_000,
        "volatile_tokens_est": 200,
        "completion_tokens": 500,
    }
    base.update(kwargs)
    return base


def test_replay_shows_frontier_baseline_is_more_expensive():
    records = [_record(trace_id=str(i)) for i in range(20)]
    results = {r.name: r for r in replay(records)}

    assert results["as_recorded"].requests == 20
    assert results["always_frontier"].total_usd > results["as_recorded"].total_usd


def test_replay_credits_stickiness_with_cache_reads():
    """Same session, repeated prefix: the sticky variant should accumulate
    cache reads that the no-stickiness variant does not."""
    records = [_record(trace_id=str(i), resolved_intent="code_write") for i in range(10)]
    results = {r.name: r for r in replay(records)}

    sticky = results["policy_floor_sticky"]
    assert sticky.cache_read_tokens > 0


def test_replay_skips_failed_requests():
    records = [_record(trace_id="1"), _record(trace_id="2", outcome="error")]
    results = {r.name: r for r in replay(records)}
    assert results["as_recorded"].requests == 1
