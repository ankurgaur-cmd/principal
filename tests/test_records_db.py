"""The SQLite record store and the analytics computed over it.

The record is only worth writing if it can be read back and aggregated. These
tests hold the store to both halves: a lossless round-trip of the record, and
analytics that agree with arithmetic done by hand on the same rows.
"""

from __future__ import annotations

import time

import pytest

from aigateway.observability import RecordSink, build_sink
from aigateway.observability.db import SqliteRecordSink
from aigateway.observability.record import RequestRecord


@pytest.fixture
def sink(tmp_path) -> SqliteRecordSink:
    s = SqliteRecordSink(tmp_path / "records.db")
    yield s
    s.close()


def _record(i: int, **over) -> RequestRecord:
    base = dict(
        trace_id=f"t-{i}",
        tenant="acme",
        agent="pytest",
        session_id="s1",
        resolved_intent="classify",
        chosen_model="gpt-5-nano",
        provider="openai",
        tier="light",
        cache_state="warm_read",
        pilot_role="warm",
        routing_ok=True,
        prompt_tokens=1000,
        completion_tokens=100,
        cache_read_tokens=800,
        estimated_cost_usd=0.001,
        actual_cost_usd=0.001,
        cache_savings_usd=0.0005,
        upstream_ms=900,
        latency_total_ms=1000,
        outcome="ok",
        considered=[{"model": "gpt-5-nano", "cost_usd": 0.001}],
        fallback_chain=[],
    )
    base.update(over)
    return RequestRecord(**base)


# -- round trip --------------------------------------------------------------
def test_records_round_trip_losslessly(sink):
    rec = _record(1, quality_failures=["truncated"], hosts_contacted=["api.openai.com"])
    sink.write(rec)

    (row,) = sink.read_all()
    assert row["trace_id"] == "t-1"
    assert row["considered"] == [{"model": "gpt-5-nano", "cost_usd": 0.001}]
    assert row["quality_failures"] == ["truncated"]
    assert row["hosts_contacted"] == ["api.openai.com"]
    assert row["routing_ok"] is True
    assert row["actual_cost_usd"] == pytest.approx(0.001)


def test_read_all_matches_the_replay_harness_expectations(sink):
    """replay.harness consumes read_all(); the shape it needs must survive."""
    from aigateway.replay.harness import replay

    for i in range(3):
        sink.write(_record(i, prefix_tokens_est=5000, volatile_tokens_est=50))
    results = replay(sink.read_all())
    as_recorded = next(r for r in results if r.name == "as_recorded")
    assert as_recorded.requests == 3


def test_a_write_failure_never_raises(tmp_path):
    s = SqliteRecordSink(tmp_path / "r.db")
    s.close()  # writes against a closed connection must be swallowed, not raised
    s.write(_record(1))


def test_build_sink_switches_on_extension(tmp_path):
    assert isinstance(build_sink(tmp_path / "r.db"), SqliteRecordSink)
    assert isinstance(build_sink(str(tmp_path / "r.jsonl")), RecordSink)


# -- analytics ---------------------------------------------------------------
@pytest.fixture
def seeded(sink) -> SqliteRecordSink:
    now = time.time()
    rows = [
        # three ok on the cheap model, one of them a routing miss
        _record(1, timestamp=now - 60),
        _record(2, timestamp=now - 120, routing_ok=False, quality_failures=["empty"]),
        _record(3, timestamp=now - 180),
        # one expensive request on another model, cold write, estimator 100% high
        _record(
            4,
            timestamp=now - 240,
            chosen_model="claude-opus-5",
            provider="anthropic",
            resolved_intent="code_review",
            cache_state="cold_write",
            pilot_role="pilot",
            cache_read_tokens=0,
            cache_write_tokens=1000,
            cache_savings_usd=0.0,
            estimated_cost_usd=0.02,
            actual_cost_usd=0.01,
        ),
        # one error — no cost, must not pollute averages
        _record(
            5,
            timestamp=now - 300,
            outcome="error",
            error="boom",
            actual_cost_usd=0.0,
            estimated_cost_usd=0.0,
            cache_read_tokens=0,
            cache_savings_usd=0.0,
        ),
        # outside the window — must be invisible at hours=1
        _record(6, timestamp=now - 7200, actual_cost_usd=99.0),
    ]
    for r in rows:
        # timestamp is a dataclass default; override deliberately
        sink.write(r)
    return sink


def test_overview_matches_hand_arithmetic(seeded):
    o = seeded.analytics(hours=1.0)["overview"]
    assert o["requests"] == 5  # the 2h-old record is out of window
    assert o["ok"] == 4
    assert o["errors"] == 1
    assert o["spend_usd"] == pytest.approx(0.001 * 3 + 0.01)
    assert o["cache_savings_usd"] == pytest.approx(0.0005 * 3)
    assert o["routing_misses"] == 1
    assert o["routing_miss_rate"] == pytest.approx(0.25)
    # Totals span *all* transactions in the window, failures included — tokens
    # were consumed whether or not the request ended well.
    assert o["prompt_tokens"] == 1000 * 5
    assert o["completion_tokens"] == 100 * 5
    assert o["total_tokens"] == 1100 * 5


def test_estimate_drift_is_signed_and_skips_unpriced_rows(seeded):
    o = seeded.analytics(hours=1.0)["overview"]
    # Three exact estimates (0) and one +100% over-estimate; the error row has
    # no actual cost and must be excluded rather than dragging the mean to 0.
    assert o["estimate_error_mean"] == pytest.approx(0.25)
    assert o["estimate_error_mean_abs"] == pytest.approx(0.25)


def test_by_model_orders_by_spend_and_tracks_cache_delivery(seeded):
    models = seeded.analytics(hours=1.0)["by_model"]
    assert [m["model"] for m in models] == ["claude-opus-5", "gpt-5-nano"]

    nano = models[1]
    assert nano["expected_warm"] == 3
    assert nano["delivered_warm"] == 3
    assert nano["routing_misses"] == 1
    assert nano["p50_upstream_ms"] == 900

    opus = models[0]
    assert opus["cache_write_tokens"] == 1000
    assert opus["expected_warm"] == 0  # cold writes are not evidence against it


def test_pilot_roles_and_intents_are_aggregated(seeded):
    payload = seeded.analytics(hours=1.0)
    roles = {r["role"]: r["requests"] for r in payload["pilot_roles"]}
    assert roles == {"warm": 4, "pilot": 1}

    intents = {r["intent"]: r for r in payload["by_intent"]}
    assert intents["classify"]["requests"] == 4
    assert intents["code_review"]["spend_usd"] == pytest.approx(0.01)


def test_routing_misses_only_lists_offending_pairs(seeded):
    misses = seeded.analytics(hours=1.0)["routing_misses"]
    assert len(misses) == 1
    assert misses[0]["model"] == "gpt-5-nano"
    assert misses[0]["intent"] == "classify"
    assert misses[0]["misses"] == 1


def test_window_excludes_old_records(seeded):
    wide = seeded.analytics(hours=3.0)["overview"]
    assert wide["requests"] == 6
    assert wide["spend_usd"] == pytest.approx(0.001 * 3 + 0.01 + 99.0)


# -- transaction log ----------------------------------------------------------
def test_transactions_filter_and_paginate(seeded):
    newest_first = seeded.transactions(limit=10)
    assert [r["trace_id"] for r in newest_first][:3] == ["t-1", "t-2", "t-3"]

    # Cursor pagination: page 2 starts strictly before page 1's last row.
    page1 = seeded.transactions(limit=2)
    page2 = seeded.transactions(limit=2, before=page1[-1]["timestamp"])
    assert {r["trace_id"] for r in page1}.isdisjoint(r["trace_id"] for r in page2)

    only_opus = seeded.transactions(model="claude-opus-5")
    assert [r["trace_id"] for r in only_opus] == ["t-4"]

    errors = seeded.transactions(outcome="error")
    assert [r["trace_id"] for r in errors] == ["t-5"]

    windowed = seeded.transactions(since=__import__("time").time() - 3600)
    assert "t-6" not in {r["trace_id"] for r in windowed}


def test_transactions_rows_are_fully_decoded(seeded):
    (row,) = seeded.transactions(model="claude-opus-5")
    assert isinstance(row["considered"], list)
    assert isinstance(row["fallback_chain"], list)
    assert row["routing_ok"] is True


def test_facets_offer_only_values_present_in_the_window(seeded):
    f = seeded.facets(since=__import__("time").time() - 3600)
    assert set(f["models"]) == {"gpt-5-nano", "claude-opus-5"}
    assert "error" in f["outcomes"] and "ok" in f["outcomes"]
    assert f["tenants"] == ["acme"]


# -- read-only SQL explorer ---------------------------------------------------
def test_explorer_answers_arbitrary_selects(seeded):
    result = seeded.run_readonly(
        "SELECT chosen_model, COUNT(*) AS n FROM records GROUP BY 1 ORDER BY n DESC"
    )
    assert result["columns"] == ["chosen_model", "n"]
    assert ["gpt-5-nano", 5] in result["rows"]
    assert result["truncated"] is False


def test_explorer_refuses_writes_at_two_layers(seeded):
    # Layer 1: statement shape.
    refused = seeded.run_readonly("DELETE FROM records")
    assert "read-only" in refused["error"]

    # Layer 2: a write smuggled past the prefix hits the query_only engine.
    smuggled = seeded.run_readonly(
        "WITH x AS (SELECT 1) INSERT INTO records (trace_id) VALUES ('evil')"
    )
    assert "error" in smuggled
    assert all(r["trace_id"] != "evil" for r in seeded.read_all())


def test_explorer_reports_truncation_instead_of_lying(seeded):
    result = seeded.run_readonly("SELECT trace_id FROM records", limit=2)
    assert result["row_count"] == 2
    assert result["truncated"] is True


def test_explorer_surfaces_sql_errors_readably(seeded):
    result = seeded.run_readonly("SELECT no_such_column FROM records")
    assert "no_such_column" in result["error"]


def test_schema_names_the_table_and_row_count(seeded):
    schema = seeded.schema()
    assert schema["records"]["rows"] == 6
    names = {c["name"] for c in schema["records"]["columns"]}
    assert {"trace_id", "actual_cost_usd", "routing_reason"} <= names
