"""SQLite-backed request records and the analytics queries over them.

Why a database and not the JSONL: the record is only worth writing if it gets
*read*, and the questions worth asking of it are aggregations — spend by model,
estimate error drift, cache savings over time, routing miss rate by intent.
Those are one SQL statement each against a table and a growing re-parse against
a file. SQLite specifically because it is zero-dependency, a single file, and
every analysis tool already opens it:

    sqlite3 var/records.db 'SELECT chosen_model, SUM(actual_cost_usd) ...'
    duckdb -c "SELECT ... FROM sqlite_scan('var/records.db', 'records')"

The write path stays deliberately boring: one INSERT per request, WAL mode so
readers never block the serving path, and a failure to record must never fail
the request it records.

Growth is not a problem the way the JSONL's was: the table is indexed by time,
queries read the window they need, and a year of heavy traffic is still a file
measured in hundreds of MB — no rotation machinery required.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

from .record import RequestRecord

log = logging.getLogger(__name__)

# Fields serialised as JSON text rather than flattened into columns. They are
# variable-shape detail for drill-down, not grouping keys.
_JSON_FIELDS = ("considered", "quality_failures", "hosts_contacted", "fallback_chain")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    trace_id            TEXT PRIMARY KEY,
    timestamp           REAL NOT NULL,
    tenant              TEXT,
    agent               TEXT,
    session_id          TEXT,
    declared_intent     TEXT,
    resolved_intent     TEXT,
    intent_confidence   REAL,
    intent_source       TEXT,
    chosen_model        TEXT,
    provider            TEXT,
    tier                TEXT,
    effort              TEXT,
    routing_reason      TEXT,
    considered          TEXT,
    cache_state         TEXT,
    cache_plan          TEXT,
    pilot_role          TEXT,
    quality_verdict     TEXT,
    quality_failures    TEXT,
    routing_ok          INTEGER,
    hosts_contacted     TEXT,
    hop_count           INTEGER,
    upstream_ms         INTEGER,
    gateway_overhead_ms INTEGER,
    prefix_tokens_est   INTEGER,
    volatile_tokens_est INTEGER,
    prompt_tokens       INTEGER,
    completion_tokens   INTEGER,
    cache_read_tokens   INTEGER,
    cache_write_tokens  INTEGER,
    estimated_cost_usd  REAL,
    actual_cost_usd     REAL,
    cache_savings_usd   REAL,
    estimate_error      REAL,
    extra_effort        REAL,
    latency_ttft_ms     INTEGER,
    latency_total_ms    INTEGER,
    fallback_chain      TEXT,
    degraded            INTEGER,
    outcome             TEXT,
    error               TEXT
);
CREATE INDEX IF NOT EXISTS idx_records_time   ON records (timestamp);
CREATE INDEX IF NOT EXISTS idx_records_model  ON records (chosen_model, timestamp);
CREATE INDEX IF NOT EXISTS idx_records_tenant ON records (tenant, timestamp);
CREATE INDEX IF NOT EXISTS idx_records_intent ON records (resolved_intent, timestamp);
"""


class SqliteRecordSink:
    """Drop-in for ``RecordSink``: same ``write``/``read_all`` surface.

    A single connection guarded by a lock. Writes are sub-millisecond against
    upstream calls of tens of seconds, so a mutex — not a queue — is the right
    amount of machinery.
    """

    def __init__(self, path: str, fleet=None):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fleet = fleet
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL so dashboard reads never block the serving path's writes.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    @property
    def path(self) -> Path:
        return self._path

    # -- write path ----------------------------------------------------------
    def write(self, record: RequestRecord) -> None:
        if self._fleet is not None:
            try:
                self._fleet.record(record)
            except Exception:
                pass  # telemetry must never fail a request

        from dataclasses import asdict

        payload = asdict(record)
        payload["estimate_error"] = round(record.estimate_error, 6)
        for f in _JSON_FIELDS:
            payload[f] = json.dumps(payload.get(f) or [], default=str)
        payload["routing_ok"] = int(bool(payload.get("routing_ok")))
        payload["degraded"] = int(bool(payload.get("degraded")))

        columns = ", ".join(payload)
        placeholders = ", ".join(f":{k}" for k in payload)
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    f"INSERT OR REPLACE INTO records ({columns}) VALUES ({placeholders})",
                    payload,
                )
        except sqlite3.Error as exc:  # a lost record must never fail a request
            log.warning("record write failed: %s", exc)

    # -- read paths ----------------------------------------------------------
    def read_all(self) -> list[dict]:
        """Every record, JSON fields decoded — the shape replay.harness expects."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM records ORDER BY timestamp"
            ).fetchall()
        return [_decode(dict(r)) for r in rows]

    def query(self, sql: str, params: tuple | dict = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def transactions(
        self,
        limit: int = 50,
        before: float | None = None,
        since: float | None = None,
        tenant: str | None = None,
        model: str | None = None,
        intent: str | None = None,
        outcome: str | None = None,
        session: str | None = None,
    ) -> list[dict]:
        """One page of the transaction log, newest first, fully decoded.

        Cursor pagination by timestamp (``before``) rather than OFFSET: the
        table grows at the head, so an offset-paged reader walking backwards
        would see rows shift under it and duplicate or skip records.
        """
        clauses, params = [], []
        for clause, value in (
            ("timestamp < ?", before),
            ("timestamp >= ?", since),
            ("tenant = ?", tenant),
            ("chosen_model = ?", model),
            ("resolved_intent = ?", intent),
            ("outcome = ?", outcome),
            ("session_id = ?", session),
        ):
            if value is not None:
                clauses.append(clause)
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.query(
            f"SELECT * FROM records {where} ORDER BY timestamp DESC LIMIT ?",  # noqa: S608 - clauses are literals
            (*params, max(1, min(limit, 500))),
        )
        return [_decode(r) for r in rows]

    # -- explorer ------------------------------------------------------------
    def schema(self) -> dict:
        """What is in this database — tables, columns, and how big they are."""
        with self._lock:
            tables = [
                r["name"]
                for r in self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name"
                ).fetchall()
            ]
            out = {}
            for table in tables:
                cols = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
                count = self._conn.execute(
                    f"SELECT COUNT(*) AS n FROM {table}"  # noqa: S608 - names from sqlite_master
                ).fetchone()
                out[table] = {
                    "columns": [{"name": c["name"], "type": c["type"]} for c in cols],
                    "rows": count["n"],
                }
        return out

    def run_readonly(self, sql: str, limit: int = 200) -> dict:
        """Run one read-only statement against the records database.

        Defence in depth, because this is reachable from a browser page:
        the statement must *read* (SELECT/WITH/EXPLAIN), and it runs on a
        separate ``mode=ro`` connection with ``query_only`` set — so even a
        statement that smuggles a write past the prefix check hits a wall at
        the engine. Results are capped, and the cap is reported rather than
        silently applied.
        """
        head = sql.strip().rstrip(";").strip()
        if not head:
            return {"error": "empty statement"}
        if head.split(None, 1)[0].lower() not in ("select", "with", "explain"):
            return {
                "error": "read-only console: only SELECT / WITH / EXPLAIN run here. "
                "The database file is yours — open it with sqlite3 for writes."
            }
        try:
            conn = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True, timeout=5)
        except sqlite3.Error as exc:
            return {"error": str(exc)}
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            cursor = conn.execute(head)
            columns = [c[0] for c in cursor.description or []]
            rows = cursor.fetchmany(max(1, min(limit, 2000)))
            truncated = cursor.fetchone() is not None
            return {
                "columns": columns,
                "rows": [list(r) for r in rows],
                "truncated": truncated,
                "row_count": len(rows),
            }
        except sqlite3.Error as exc:
            return {"error": str(exc)}
        finally:
            conn.close()

    def facets(self, since: float | None = None) -> dict:
        """Distinct filter values in the window — what the dropdowns offer."""

        def distinct(column: str, extra: str = "") -> list[str]:
            clauses, params = [], []
            if since is not None:
                clauses.append("timestamp >= ?")
                params.append(since)
            if extra:
                clauses.append(extra)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            return [
                r["v"]
                for r in self.query(
                    f"SELECT DISTINCT {column} AS v FROM records {where} ORDER BY 1",  # noqa: S608
                    tuple(params),
                )
                if r["v"]
            ]

        return {
            "models": distinct("chosen_model", "chosen_model != ''"),
            "intents": distinct("resolved_intent"),
            "tenants": distinct("tenant"),
            "outcomes": distinct("outcome"),
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- analytics ------------------------------------------------------------
    def analytics(self, hours: float = 24.0) -> dict:
        """The dashboard payload: every aggregate the review said to watch.

        One method rather than one endpoint per panel, because the panels are
        read together and the queries share a time window.
        """
        since = time.time() - hours * 3600
        p = (since,)

        overview = self.query(
            """
            SELECT COUNT(*)                                        AS requests,
                   SUM(outcome = 'ok')                             AS ok,
                   SUM(outcome NOT IN ('ok', 'refusal'))           AS errors,
                   SUM(outcome = 'refusal')                        AS refusals,
                   COALESCE(SUM(actual_cost_usd), 0)               AS spend_usd,
                   COALESCE(SUM(cache_savings_usd), 0)             AS cache_savings_usd,
                   COALESCE(SUM(cache_read_tokens), 0)             AS cache_read_tokens,
                   COALESCE(SUM(cache_write_tokens), 0)            AS cache_write_tokens,
                   COALESCE(SUM(prompt_tokens), 0)                 AS prompt_tokens,
                   COALESCE(SUM(completion_tokens), 0)             AS completion_tokens,
                   COALESCE(SUM(prompt_tokens + completion_tokens), 0)
                                                                   AS total_tokens,
                   SUM(outcome = 'ok' AND routing_ok = 0)          AS routing_misses,
                   SUM(degraded)                                   AS degraded,
                   SUM(outcome = 'ok' AND fallback_chain != '[]')  AS fallbacks,
                   AVG(CASE WHEN outcome = 'ok' AND actual_cost_usd > 0
                            THEN estimate_error END)               AS estimate_error_mean,
                   AVG(CASE WHEN outcome = 'ok' AND actual_cost_usd > 0
                            THEN ABS(estimate_error) END)          AS estimate_error_mean_abs
            FROM records WHERE timestamp >= ?
            """,
            p,
        )[0]

        by_model = self.query(
            """
            SELECT chosen_model AS model, provider, tier,
                   COUNT(*)                              AS requests,
                   SUM(outcome = 'ok')                   AS ok,
                   COALESCE(SUM(actual_cost_usd), 0)     AS spend_usd,
                   COALESCE(SUM(cache_savings_usd), 0)   AS cache_savings_usd,
                   COALESCE(SUM(prompt_tokens), 0)       AS prompt_tokens,
                   COALESCE(SUM(completion_tokens), 0)   AS completion_tokens,
                   COALESCE(SUM(cache_read_tokens), 0)   AS cache_read_tokens,
                   COALESCE(SUM(cache_write_tokens), 0)  AS cache_write_tokens,
                   SUM(outcome = 'ok' AND routing_ok = 0) AS routing_misses,
                   SUM(outcome = 'ok' AND cache_state = 'warm_read')
                                                         AS expected_warm,
                   SUM(outcome = 'ok' AND cache_state = 'warm_read'
                       AND cache_read_tokens > 0)        AS delivered_warm,
                   AVG(CASE WHEN outcome = 'ok' AND actual_cost_usd > 0
                            THEN estimate_error END)     AS estimate_error_mean
            FROM records WHERE timestamp >= ? AND chosen_model != ''
            GROUP BY chosen_model ORDER BY spend_usd DESC
            """,
            p,
        )

        # Percentiles have no SQLite builtin; the window's latency column is
        # small, so pull it and take them in Python rather than an OFFSET dance.
        for row in by_model:
            lat = [
                r["upstream_ms"]
                for r in self.query(
                    "SELECT upstream_ms FROM records "
                    "WHERE timestamp >= ? AND chosen_model = ? AND outcome = 'ok' "
                    "AND upstream_ms > 0",
                    (since, row["model"]),
                )
            ]
            row["p50_upstream_ms"] = _pct(lat, 0.50)
            row["p95_upstream_ms"] = _pct(lat, 0.95)

        by_intent = self.query(
            """
            SELECT resolved_intent AS intent,
                   COUNT(*)                               AS requests,
                   COALESCE(SUM(actual_cost_usd), 0)      AS spend_usd,
                   SUM(outcome = 'ok' AND routing_ok = 0) AS routing_misses,
                   COALESCE(AVG(CASE WHEN outcome = 'ok'
                            THEN completion_tokens END), 0) AS avg_completion_tokens
            FROM records WHERE timestamp >= ?
            GROUP BY resolved_intent ORDER BY requests DESC
            """,
            p,
        )

        pilot_roles = self.query(
            """
            SELECT pilot_role AS role, COUNT(*) AS requests
            FROM records WHERE timestamp >= ? AND pilot_role != ''
            GROUP BY pilot_role ORDER BY requests DESC
            """,
            p,
        )

        misses = self.query(
            """
            SELECT chosen_model AS model, resolved_intent AS intent,
                   COUNT(*) AS requests,
                   SUM(routing_ok = 0) AS misses
            FROM records
            WHERE timestamp >= ? AND outcome = 'ok'
            GROUP BY chosen_model, resolved_intent
            HAVING misses > 0 ORDER BY misses DESC LIMIT 20
            """,
            p,
        )

        timeseries = self.query(
            """
            SELECT strftime('%Y-%m-%dT%H:00', timestamp, 'unixepoch') AS hour,
                   COUNT(*)                            AS requests,
                   COALESCE(SUM(actual_cost_usd), 0)   AS spend_usd,
                   COALESCE(SUM(cache_savings_usd), 0) AS cache_savings_usd,
                   SUM(outcome NOT IN ('ok', 'refusal')) AS errors
            FROM records WHERE timestamp >= ?
            GROUP BY hour ORDER BY hour
            """,
            p,
        )

        return {
            "window_hours": hours,
            "generated_at": time.time(),
            "db_path": str(self._path),
            "overview": {
                **{k: overview[k] or 0 for k in overview.keys()},
                "routing_miss_rate": _rate(overview["routing_misses"], overview["ok"]),
                "fallback_rate": _rate(overview["fallbacks"], overview["ok"]),
            },
            "by_model": by_model,
            "by_intent": by_intent,
            "pilot_roles": pilot_roles,
            "routing_misses": misses,
            "timeseries": timeseries,
        }


def _decode(row: dict) -> dict:
    for f in _JSON_FIELDS:
        try:
            row[f] = json.loads(row[f]) if row.get(f) else []
        except (TypeError, json.JSONDecodeError):
            row[f] = []
    row["routing_ok"] = bool(row.get("routing_ok"))
    row["degraded"] = bool(row.get("degraded"))
    return row


def _pct(values: list[int], p: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * p), len(ordered) - 1)]


def _rate(part, whole) -> float:
    return round((part or 0) / whole, 4) if whole else 0.0
