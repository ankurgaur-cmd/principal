"""Response quality checks.

These exist because the gateway can prove it picked a cheaper model but cannot,
without them, tell you the cheaper model was good enough — which makes every
claimed saving unfalsifiable.
"""

from __future__ import annotations

import pytest
from conftest import make_request

from aigateway.quality import (
    REASONING_FLOOR_BY_EFFORT,
    assess,
    budget_starves_the_answer,
    effort_that_fits,
)
from aigateway.schemas import ProviderResponse, Usage


def _resp(text="fine", finish="stop", tools=None, **usage):
    return ProviderResponse(
        text=text,
        tool_calls=tools or [],
        finish_reason=finish,
        model="gpt-5-nano",
        usage=Usage(**{"prompt_tokens": 100, "completion_tokens": 40, **usage}),
    )


@pytest.fixture
async def decision(router):
    return await router.route(make_request(), "classify")


async def test_a_good_response_passes(router, decision):
    report = assess(make_request(), _resp(), decision)
    assert report.verdict == "pass"
    assert report.routing_ok is True


async def test_empty_answer_from_exhausted_budget_is_a_failure(router, decision):
    """The bug that made responses look invisible: a reasoning model spends the
    whole output budget thinking and returns nothing."""
    report = assess(make_request(), _resp(text="", finish="length"), decision)

    assert report.verdict == "fail"
    assert report.routing_ok is False
    check = report.failures[0]
    assert check.id == "reasoning_starved"
    # The message has to say what to do about it, not just that it happened,
    # and quote the floor for the effort actually used.
    assert str(REASONING_FLOOR_BY_EFFORT[decision.effort]) in check.detail.replace(",", "")
    assert "max_tokens" in check.detail


async def test_truncated_answer_is_a_failure(router, decision):
    report = assess(make_request(), _resp(text="Here is the ans", finish="length"), decision)
    assert report.failures[0].id == "truncated"


async def test_empty_without_truncation_is_reported_differently(router, decision):
    report = assess(make_request(), _resp(text="", finish="stop"), decision)
    assert report.failures[0].id == "empty_response"


async def test_invalid_json_against_a_requested_schema_fails(router, decision):
    req = make_request()
    req.response_schema = {"type": "object", "required": ["name"]}
    report = assess(req, _resp(text="not json at all"), decision)
    assert report.failures[0].id == "schema_invalid"


async def test_missing_required_fields_fail(router, decision):
    req = make_request()
    req.response_schema = {"type": "object", "required": ["name", "email"]}
    report = assess(req, _resp(text='{"name": "Ada"}'), decision)

    failure = report.failures[0]
    assert failure.id == "schema_incomplete"
    assert "email" in failure.detail


async def test_valid_structured_output_passes(router, decision):
    req = make_request()
    req.response_schema = {"type": "object", "required": ["name"]}
    report = assess(req, _resp(text='{"name": "Ada"}'), decision)
    assert report.routing_ok is True


async def test_malformed_tool_arguments_fail(router, decision):
    """A weaker model emitting broken tool JSON is a classic too-low-tier sign."""
    bad = [{"function": {"name": "search", "arguments": "{not json"}}]
    report = assess(make_request(), _resp(text="", tools=bad), decision)

    failure = report.failures[0]
    assert failure.id == "tool_args_invalid"
    assert "search" in failure.title


async def test_wellformed_tool_calls_pass(router, decision):
    good = [{"function": {"name": "search", "arguments": '{"q": "x"}'}}]
    report = assess(make_request(), _resp(text="", tools=good), decision)
    assert report.routing_ok is True


async def test_a_missed_cache_is_a_warning_not_a_failure(router):
    """The answer is fine; the router's cost model was wrong. Different problem."""
    req = make_request(session_id="q-cache", system_tokens=20_000)
    decision = await router.route(req, "code_write")
    await router.remember("q-cache", decision.model)
    warm = await router.route(req, "code_write")
    assert warm.cache_state == "warm_read"

    report = assess(req, _resp(cache_read_tokens=0), warm)
    ids = [c.id for c in report.warnings]
    assert "cache_missed" in ids
    assert report.routing_ok is True


async def test_cache_hit_is_reported(router, decision):
    report = assess(make_request(), _resp(cache_read_tokens=8000), decision)
    assert any(c.id == "cache_hit" for c in report.checks)


async def test_thin_answer_for_a_demanding_intent_warns(router):
    d = await router.route(make_request(), "code_review")
    report = assess(make_request(), _resp(text="Looks fine."), d)

    assert any(c.id == "thin_for_intent" for c in report.warnings)
    assert report.routing_ok is True, "a short answer is suspicious, not disqualifying"


async def test_verdict_ranks_failure_above_warning(router, decision):
    report = assess(make_request(), _resp(text="", finish="length", cache_read_tokens=0), decision)
    assert report.verdict == "fail"


async def test_summary_is_serialisable(router, decision):
    import json

    json.dumps(assess(make_request(), _resp(), decision).summary())


# -- the effort ladder -----------------------------------------------------
def test_reasoning_floor_scales_with_effort():
    """A single flat floor was too low for high-effort work: requests cleared
    it and still came back empty. Reasoning depth consumes the budget, so the
    threshold has to move with it."""
    floors = [REASONING_FLOOR_BY_EFFORT[e] for e in ("low", "medium", "high", "xhigh", "max")]
    assert floors == sorted(floors), "floors must increase with effort"
    assert len(set(floors)) == len(floors), "each level needs its own floor"


def test_floors_reflect_what_reasoning_models_actually_spend():
    """These were guessed once and were 5-10x too low, which made every heavy
    request return an empty answer. Measured on one heavy prompt, the *cheapest*
    effort level spent 1,937 tokens on gpt-5 and 2,828 on claude-opus-5 before
    any visible output. A floor under that is not a floor."""
    assert REASONING_FLOOR_BY_EFFORT["low"] >= 2828


def test_a_budget_under_the_lowest_floor_is_flagged_before_the_call():
    """Stepping effort down rescues a tight budget only until the bottom rung.
    Below that there is nothing left to trade, so say so before spending."""
    assert budget_starves_the_answer(1200) is True
    assert budget_starves_the_answer(REASONING_FLOOR_BY_EFFORT["low"] - 1) is True
    assert budget_starves_the_answer(REASONING_FLOOR_BY_EFFORT["low"]) is False
    assert budget_starves_the_answer(8000) is False


def test_a_budget_exactly_at_the_floor_does_not_keep_its_effort():
    """The bug a user hit: 8,000 tokens at effort 'high' against a floor of
    8,000. Arithmetically there is nothing left for the answer, but the check
    read as satisfied and the request came back empty."""
    from aigateway.quality import REASONING_FLOOR_BY_EFFORT as F

    assert effort_that_fits("high", F["high"]) == "medium"
    assert effort_that_fits("medium", F["medium"]) == "low"


def test_the_suggested_budget_is_never_the_one_that_just_failed():
    """The advice was "raise max_tokens above ~8000 for effort 'high'" to
    someone who had sent exactly 8,000 — circular, and still wrong."""
    from aigateway.quality import budget_for_effort

    for effort in ("low", "medium", "high", "xhigh", "max"):
        assert budget_for_effort(effort) > REASONING_FLOOR_BY_EFFORT[effort]


def test_effort_steps_down_to_fit_the_budget():
    """Respects the caller's cap rather than raising it — their budget is their
    decision, how much goes to reasoning is ours."""
    assert effort_that_fits("high", 9500) == "high"
    assert effort_that_fits("high", 6500) == "medium"
    assert effort_that_fits("high", 4500) == "low"
    assert effort_that_fits("max", 100) == "low", "never below the bottom rung"


def test_effort_is_never_stepped_up():
    """Fitting is a downgrade to protect the answer, not licence to spend more."""
    assert effort_that_fits("low", 100_000) == "low"
    assert effort_that_fits("medium", 100_000) == "medium"


def test_unknown_effort_passes_through():
    assert effort_that_fits("bogus", 100) == "bogus"
