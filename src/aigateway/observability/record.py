"""The request record: one JSONL line per request.

This is the most important file in the project and the easiest to skip. You
cannot optimise cost you cannot see, and "route by intent to save money" is a
vibe until you can answer, from data: what did the router choose, why, what did
it cost, and what would the alternative have cost.

The cache fields are non-negotiable. ``cache_read_tokens`` pinned at zero across
repeated identical-prefix requests is the single clearest symptom of a broken
caching design, and it is invisible in a standard OpenAI-shaped response.

Records feed ``replay.harness``, which re-scores historical traffic against
alternative routing policies offline. That loop is what turns the router from a
set of guesses into something you can defend.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RequestRecord:
    trace_id: str
    timestamp: float = field(default_factory=time.time)

    tenant: str = ""
    agent: str = ""
    session_id: str | None = None

    declared_intent: str | None = None
    resolved_intent: str = "unknown"
    intent_confidence: float = 0.0
    intent_source: str = ""

    chosen_model: str = ""
    provider: str = ""
    tier: str = ""
    effort: str = ""
    routing_reason: str = ""
    considered: list[dict] = field(default_factory=list)

    cache_state: str = ""
    cache_plan: str = ""
    pilot_role: str = ""
    hosts_contacted: list[str] = field(default_factory=list)
    hop_count: int = 0
    upstream_ms: int = 0
    gateway_overhead_ms: int = 0
    prefix_tokens_est: int = 0
    volatile_tokens_est: int = 0

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    cache_savings_usd: float = 0.0

    latency_ttft_ms: int = 0
    latency_total_ms: int = 0

    fallback_chain: list[str] = field(default_factory=list)
    degraded: bool = False
    outcome: str = "ok"  # ok | refusal | error | rate_limited | budget_exceeded
    error: str | None = None

    @property
    def estimate_error(self) -> float:
        """Signed relative error of the routing cost estimate.

        Track this. An estimator nobody checks is how budget enforcement
        silently stops working.
        """
        if self.actual_cost_usd == 0:
            return 0.0
        return (self.estimated_cost_usd - self.actual_cost_usd) / self.actual_cost_usd


class RecordSink:
    """Append-only JSONL sink.

    Fine for a prototype. In production, point this at your OTel pipeline —
    the field set maps cleanly onto span attributes, and OTel is where LLM
    gateway telemetry belongs.
    """

    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: RequestRecord) -> None:
        payload = asdict(record)
        payload["estimate_error"] = round(record.estimate_error, 4)
        line = json.dumps(payload, default=str)
        # Line-buffered append; a lost record must never fail a request.
        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    def read_all(self) -> list[dict]:
        if not os.path.exists(self._path):
            return []
        with open(self._path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
