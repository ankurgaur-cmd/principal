"""The plain-language explanation.

These assert on *comprehensibility*, which is unusual for a test suite but is
the whole point of the module: the explanation exists because the router's own
`reason` string is unreadable to anyone who hasn't read the router.
"""

from __future__ import annotations

import pytest
from conftest import make_request

from aigateway.routing import explain


@pytest.fixture
async def decision(router):
    return await router.route(make_request(user_text="Review this diff"), "code_review")


async def test_explanation_has_all_five_beats(router):
    d = await router.route(make_request(), "code_review")
    ex = explain(d, 0.75, "rules")

    assert [s["n"] for s in ex["steps"]] == [1, 2, 3, 4, 5]
    assert [s["title"] for s in ex["steps"]] == [
        "What you asked for",
        "What that needs",
        "Who could do it",
        "Why this one",
        "What it costs",
    ]


async def test_headline_names_the_model_and_the_reason(router):
    d = await router.route(make_request(), "classify")
    ex = explain(d, 0.9, "rules")

    assert d.model.key in ex["headline"]
    assert ex["headline"].endswith(".")


async def test_no_jargon_leaks_into_the_narrative(router):
    """Router-internal vocabulary is exactly what makes the log line unreadable."""
    d = await router.route(make_request(system_tokens=20_000), "code_review")
    ex = explain(d, 0.75, "rules")

    prose = " ".join(s["detail"] + s["value"] for s in ex["steps"]) + ex["headline"]
    for jargon in ("cold_write", "warm_read", "min_tier", "floor=", "cost_usd", "tier_ceiling"):
        assert jargon not in prose, f"leaked jargon: {jargon}"

    # No unrendered markdown either — the detail is injected as HTML.
    assert "*" not in prose


async def test_tier_exclusions_state_the_rule_not_a_capability_slur(router):
    """A model kept out by the tier floor is not incapable — it is smaller than
    this particular task was judged to need. Saying 'not capable enough' is both
    inaccurate and unfair to a model that may be right for the next request."""
    d = await router.route(make_request(), "architecture")
    ex = explain(d, 0.8, "rules")

    assert ex["excluded"], "heavy-tier work should rule several models out"
    for e in ex["excluded"]:
        assert e["plain"]
        assert "not capable" not in e["plain"], f"capability slur: {e['plain']}"
        assert "incapable" not in e["plain"]

    tier_rows = [e for e in ex["excluded"] if e["kind"] == "tier"]
    assert tier_rows
    for e in tier_rows:
        # States the model's own tier and the tier required.
        assert e["tier"] in e["plain"]
        assert e["required_tier"] in e["plain"]


async def test_qualified_set_spans_vendors_and_says_what_separated_them(router):
    """The question a routing decision should answer is 'which models could
    have done this, and what separated them' — not just who won."""
    d = await router.route(make_request(), "code_review")
    ex = explain(d, 0.8, "rules")

    assert len(ex["qualified"]) >= 2
    assert sum(1 for q in ex["qualified"] if q["chosen"]) == 1
    assert len(ex["vendors_in_play"]) >= 2, "cross-vendor comparison expected"

    chosen = next(q for q in ex["qualified"] if q["chosen"])
    assert chosen["note"] == "selected"
    for other in (q for q in ex["qualified"] if not q["chosen"]):
        assert other["note"], "every alternative needs a reason it lost"
        assert other["provider"] and other["tier"]


async def test_the_deciding_dimension_is_named(router):
    """'cheapest', 'only one left' and 'already warm' are different answers.
    Conflating them hides which lever changes the outcome."""
    cheapest = await router.route(make_request(), "code_review")
    assert explain(cheapest, 0.8, "rules")["decided_on"] == "cost"

    pinned = await router.route(make_request(pin_model="claude-opus-5"), "classify")
    assert explain(pinned, 1.0, "declared")["decided_on"] == "pin"

    from aigateway.catalog import Tier

    degraded = await router.route(make_request(), "architecture", cost_ceiling_tier=Tier.LIGHT)
    assert explain(degraded, 0.8, "rules")["decided_on"] == "budget"


async def test_dimensions_cover_every_axis_the_router_weighed(router):
    d = await router.route(make_request(), "code_review")
    names = {dim["name"] for dim in explain(d, 0.8, "rules")["dimensions"]}

    assert names == {
        "Intelligence required",
        "Availability",
        "Cost",
        "Cache economics",
        "Observed quality",
    }


async def test_cheapest_verdict_quotes_the_runner_up(router):
    """'Why not the other one' is the question people actually ask."""
    d = await router.route(make_request(), "code_review")
    ex = explain(d, 0.75, "rules")
    why = ex["steps"][3]["detail"]

    others = [c for c in d.considered if c.model.key != d.model.key]
    if others:
        assert "×" in why and "more for the same task" in why


async def test_sticky_decision_explains_the_cache_tradeoff(router):
    """The least intuitive decision the router makes needs the clearest prose."""
    req = make_request(session_id="s-explain", system_tokens=20_000)
    first = await router.route(req, "code_write")
    await router.remember("s-explain", first.model)

    second = await router.route(
        make_request(session_id="s-explain", system_tokens=20_000), "summarize"
    )
    ex = explain(second, 0.8, "rules")

    assert second.sticky is True
    verdict = ex["steps"][3]
    assert "already" in verdict["value"].lower() or "already" in verdict["detail"].lower()
    assert "cached" in verdict["detail"]


async def test_pinned_decision_says_so_plainly(router):
    d = await router.route(make_request(pin_model="claude-opus-5"), "classify")
    ex = explain(d, 1.0, "declared")

    assert d.pinned is True
    assert "pinned" in ex["headline"]


async def test_degraded_decision_blames_the_budget(router):
    from aigateway.catalog import Tier

    d = await router.route(make_request(), "architecture", cost_ceiling_tier=Tier.LIGHT)
    ex = explain(d, 0.8, "rules")

    assert "budget" in ex["steps"][3]["value"].lower()


async def test_technical_reason_is_kept_but_separated(router):
    d = await router.route(make_request(), "classify")
    ex = explain(d, 0.9, "rules")

    # Still available for logs and debugging — just not the headline.
    assert ex["technical"] == d.reason
    assert "intent=" in ex["technical"]
