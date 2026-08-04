"""Effort-adjusted routing.

The multiplier decides which model wins, so a wrong effort score does not make
a dashboard misleading — it misroutes traffic and bills for it. These tests are
mostly about the two ways this goes wrong: punishing a model for being handed
hard work, and inventing evidence where there is none.
"""

from __future__ import annotations

import pytest

from aigateway.routing.effort import (
    MAX_SIGNAL_EFFORT,
    MAX_TOTAL_EFFORT,
    EffortModel,
    EffortObservation,
    EffortSignal,
    Norms,
)
from aigateway.routing.reputation import Reputation


def obs(**kw) -> EffortObservation:
    base = {"model_key": "gpt-5", "intent": "code_review"}
    return EffortObservation(**{**base, **kw})


# -- the unit --------------------------------------------------------------
def test_a_clean_call_costs_no_extra_effort():
    """The whole scheme is only safe if the default is 'no adjustment'."""
    scored = EffortModel().score(obs(completion_tokens=400, visible_tokens=400))
    assert scored["extra_effort"] == 0.0
    assert scored["contributions"] == []


def test_a_fallback_is_literally_one_extra_call():
    """Not an estimate — a count. Two attempts cost two calls."""
    assert EffortModel().score(obs(attempts=2))["extra_effort"] == 1.0
    assert EffortModel().score(obs(attempts=3))["extra_effort"] == 2.0


def test_an_empty_answer_is_a_whole_call_thrown_away():
    scored = EffortModel().score(obs(empty=True, truncated=True, completion_tokens=1200))
    names = [c["signal"] for c in scored["contributions"]]
    assert "wasted_call" in names
    assert "truncation" not in names, "an empty answer is not also a truncated one"


def test_a_truncated_answer_costs_less_than_a_wasted_one():
    """A cut-off answer needs a continuation against a warm context; an empty
    one bought nothing at all."""
    m = EffortModel()
    assert (
        m.score(obs(truncated=True, completion_tokens=800, visible_tokens=800))["extra_effort"]
        < m.score(obs(empty=True))["extra_effort"]
    )


# -- confounding -----------------------------------------------------------
def test_effort_is_relative_to_the_intent_not_an_absolute_budget():
    """6,000 tokens is profligate for a classification and frugal for an
    architecture review. An absolute threshold would punish whichever model
    gets handed the hard problems — exactly backwards."""
    m = EffortModel()
    heavy = m.score(obs(completion_tokens=6000), Norms(completion_tokens=6000))
    light = m.score(obs(completion_tokens=6000), Norms(completion_tokens=600))

    assert heavy["extra_effort"] == 0.0, "normal for this kind of work"
    assert light["extra_effort"] > 0.0, "ten times the norm is real cost"


def test_turns_are_judged_against_the_norm_for_that_intent():
    """A debugging session takes more turns than a translation. Charging the
    model for that would route every hard problem to whoever has least history."""
    m = EffortModel()
    assert m.score(obs(turns_to_goal=6), Norms(turns=6.0))["extra_effort"] == 0.0
    assert m.score(obs(turns_to_goal=6), Norms(turns=2.0))["extra_effort"] > 0.0


# -- no data is not zero ---------------------------------------------------
def test_a_signal_without_data_stays_silent_rather_than_scoring_zero():
    """'No opinion' and 'no effort' are different claims. Conflating them means
    a model looks good precisely because nobody measured it."""
    scored = EffortModel().score(obs())
    silent = {s["signal"] for s in scored["silent"]}

    assert "turns_to_goal" in silent
    assert "rejection" in silent
    assert scored["extra_effort"] == 0.0


def test_signals_awaiting_data_say_what_they_are_waiting_for():
    """An empty column that names what is missing beats a signal that quietly
    is not there."""
    for row in EffortModel().table():
        if row["status"] == "awaiting data":
            assert row["needs"], f"{row['name']} does not say what it needs"
            assert "/admin/effort" in row["needs"]


def test_norms_are_needed_before_a_normalised_signal_speaks():
    scored = EffortModel().score(obs(completion_tokens=99999), Norms())
    assert all(c["signal"] != "token_overrun" for c in scored["contributions"])


# -- the open table --------------------------------------------------------
def test_a_new_signal_can_be_added_without_touching_the_scorer():
    """The point of the table. A deployment adds a row; nothing else changes."""
    m = EffortModel()
    m.register(
        EffortSignal(
            "tool_thrash",
            "Called the same tool repeatedly without progress",
            weight=1.0,
            attribution="session",
            measure=lambda o, n: o.extras.get("tool_repeats", 0) / 3,
        )
    )
    scored = m.score(obs(extras={"tool_repeats": 6}))
    assert scored["extra_effort"] == 2.0
    assert scored["contributions"][0]["signal"] == "tool_thrash"


def test_registering_a_known_name_replaces_it():
    m = EffortModel()
    before = len(m.signals)
    m.register(
        EffortSignal("retries", "changed", 5.0, "call", lambda o, n: 1.0)
    )
    assert len(m.signals) == before
    # Capped at MAX_SIGNAL_EFFORT: no single row of the table may claim more
    # than three calls' worth of effort, however it is weighted.
    assert m.score(obs())["extra_effort"] == MAX_SIGNAL_EFFORT


def test_weights_and_enablement_are_tunable_at_runtime():
    m = EffortModel()
    m.set_weight("retries", 2.0)
    assert m.score(obs(attempts=2))["extra_effort"] == 2.0

    m.set_enabled("retries", False)
    assert m.score(obs(attempts=2))["extra_effort"] == 0.0


def test_an_unknown_attribution_is_rejected():
    with pytest.raises(ValueError, match="attribution"):
        EffortModel().register(
            EffortSignal("x", "d", 1.0, "vibes", lambda o, n: 1.0)
        )


def test_a_broken_signal_cannot_break_routing():
    """A bad row in the table must degrade to silence, not to a 500."""
    m = EffortModel()

    def explode(o, n):
        raise RuntimeError("boom")

    m.register(EffortSignal("bad", "d", 1.0, "call", explode))
    scored = m.score(obs(attempts=2))
    assert scored["extra_effort"] == 1.0, "the other signals still score"


# -- caps ------------------------------------------------------------------
def test_total_effort_is_capped():
    """One catastrophic task must not exile a model on its own."""
    m = EffortModel()
    scored = m.score(
        obs(attempts=9, empty=True, turns_to_goal=50, user_rejected=True,
            manual_escalation=True, edit_distance=1.0),
        Norms(turns=1.0),
    )
    assert scored["extra_effort"] == MAX_TOTAL_EFFORT
    assert scored["capped"] is True


def test_the_score_is_always_itemised():
    """An effort penalty that cannot be itemised is indistinguishable from a
    grudge, and nobody can act on it."""
    scored = EffortModel().score(obs(attempts=2, empty=True))
    assert scored["contributions"]
    for c in scored["contributions"]:
        assert c["signal"] and c["describe"] and c["effort"] > 0
        assert c["attribution"] in ("call", "session")
    # Ordered by what actually drove the penalty.
    efforts = [c["effort"] for c in scored["contributions"]]
    assert efforts == sorted(efforts, reverse=True)


# -- composition with the retry model --------------------------------------
def test_no_effort_evidence_leaves_the_old_multiplier_untouched():
    """Turning this on must change nothing until there is evidence."""
    rep = Reputation(min_samples=2)
    for _ in range(4):
        rep.record("gpt-5", "code_review", True)
    assert rep.multiplier("gpt-5", "code_review") == 1.0

    for _ in range(4):
        rep.record("gpt-5-nano", "code_review", False)
    assert rep.multiplier("gpt-5-nano", "code_review") == pytest.approx(4.0)


def test_effort_and_retries_compound():
    """Retries repeat the whole task; effort is what each attempt costs beyond
    one clean call. They multiply because they are independent."""
    rep = Reputation(min_samples=2, max_penalty=100.0)
    for _ in range(4):
        rep.record("gpt-5", "code_review", True)      # success rate 1.0
        rep.record_effort(obs(attempts=2))            # one extra call each time

    assert rep.mean_effort("gpt-5", "code_review") == pytest.approx(1.0)
    assert rep.multiplier("gpt-5", "code_review") == pytest.approx(2.0)


def test_effort_alone_can_penalise_a_model_that_never_fails():
    """The gap the binary verdict misses: every answer 'worked', each one took
    four turns and an escalation to get there."""
    rep = Reputation(min_samples=2, max_penalty=100.0)
    for _ in range(4):
        rep.record("gpt-5-nano", "code_review", True)
        rep.record_effort(obs(model_key="gpt-5-nano", manual_escalation=True))

    assert rep.success_rate("gpt-5-nano", "code_review") == 1.0
    assert rep.multiplier("gpt-5-nano", "code_review") > 1.0


def test_thin_effort_evidence_is_ignored():
    rep = Reputation(min_samples=5)
    rep.record_effort(obs(attempts=4))
    assert rep.mean_effort("gpt-5", "code_review") == 0.0


def test_norms_are_learned_from_traffic_and_applied_after_the_fact():
    """A sample judged against a norm it has already moved looks less
    exceptional than it is."""
    rep = Reputation(min_samples=2)
    for _ in range(5):
        rep.record_effort(obs(completion_tokens=500, visible_tokens=500))

    norms = rep.norms_for("code_review")
    assert norms.completion_tokens == pytest.approx(500)

    scored = rep.record_effort(obs(completion_tokens=5000, visible_tokens=5000))
    assert any(c["signal"] == "token_overrun" for c in scored["contributions"])


def test_snapshot_reports_effort_beside_the_success_rate():
    rep = Reputation(min_samples=2)
    for _ in range(3):
        rep.record("gpt-5", "code_review", True)
        rep.record_effort(obs(attempts=2))

    row = rep.snapshot()[0]
    assert row["mean_extra_effort"] == pytest.approx(1.0)
    assert row["effort_samples"] == 3
    assert row["multiplier"] > 1.0
