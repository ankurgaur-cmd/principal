"""Offline routing replay.

"Route by intent to save money" is a claim, not a result. This harness turns it
into one: take recorded traffic, re-score it under alternative routing policies,
and compare total cost — without calling a single model.

It reuses the recorded token counts (prefix, volatile, completion), so what it
varies is *policy*, not workload. That makes the comparison honest: the
counterfactual cost of routing everything to the frontier model, or of turning
off cache-aware routing, is computed against the traffic you actually served.

    python -m aigateway.replay.harness var/records.db

Caveat worth stating plainly: replay assumes output length is independent of
the model chosen. It usually is not — a weaker model may need more turns to
finish the same job. Treat the numbers as an upper bound on savings and pair
them with a quality metric before acting on them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from ..catalog import CATALOG, ModelSpec, Tier, get_model
from ..routing.policy import policy_for


@dataclass
class PolicyVariant:
    name: str
    description: str
    choose: Callable[[dict, dict], ModelSpec | None]
    cache_aware: bool = True
    sticky: bool = True


@dataclass
class Result:
    name: str
    total_usd: float = 0.0
    requests: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model_mix: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def per_request_usd(self) -> float:
        return self.total_usd / self.requests if self.requests else 0.0


def _cost(
    model: ModelSpec,
    prefix: int,
    volatile: int,
    output: int,
    warm: bool,
    cache_ttl: str = "5m",
) -> tuple[float, int, int]:
    price_in = model.price_in_per_mtok / 1_000_000
    price_out = model.price_out_per_mtok / 1_000_000
    cacheable = prefix >= model.min_cacheable_tokens

    if not cacheable:
        return (prefix + volatile) * price_in + output * price_out, 0, 0
    if warm:
        cost = (
            prefix * price_in * model.cache_read_multiplier
            + volatile * price_in
            + output * price_out
        )
        return cost, prefix, 0
    cost = (
        prefix * price_in * model.cache_write_multiplier(cache_ttl)
        + volatile * price_in
        + output * price_out
    )
    return cost, 0, prefix


def _cheapest_at_tier(tier: Tier, providers: set[str] | None = None) -> ModelSpec:
    pool = [
        m
        for m in CATALOG.values()
        if m.tier >= tier and (providers is None or m.provider in providers)
    ]
    return min(pool, key=lambda m: m.price_in_per_mtok + m.price_out_per_mtok)


# -- built-in variants -------------------------------------------------------
def _as_recorded(record: dict, _state: dict) -> ModelSpec | None:
    return get_model(record.get("chosen_model", ""))


def _always_frontier(_record: dict, _state: dict) -> ModelSpec | None:
    return max(CATALOG.values(), key=lambda m: (m.tier, m.price_out_per_mtok))


def _always_cheapest(_record: dict, _state: dict) -> ModelSpec | None:
    return min(CATALOG.values(), key=lambda m: m.price_in_per_mtok + m.price_out_per_mtok)


def _policy_floor(record: dict, _state: dict) -> ModelSpec | None:
    policy = policy_for(record.get("resolved_intent", "unknown"))
    return _cheapest_at_tier(policy.min_tier)


DEFAULT_VARIANTS = [
    PolicyVariant("as_recorded", "what the gateway actually did", _as_recorded),
    PolicyVariant(
        "always_frontier",
        "baseline: every request to the largest model",
        _always_frontier,
    ),
    PolicyVariant(
        "always_cheapest",
        "floor: every request to the smallest model (ignores capability)",
        _always_cheapest,
    ),
    PolicyVariant(
        "policy_floor_sticky",
        "cheapest model at the intent's tier floor, session-sticky",
        _policy_floor,
    ),
    PolicyVariant(
        "policy_floor_no_stickiness",
        "same, but re-decided every request — shows what stickiness is worth",
        _policy_floor,
        sticky=False,
    ),
]


def replay(records: list[dict], variants: list[PolicyVariant] | None = None) -> list[Result]:
    variants = variants or DEFAULT_VARIANTS
    results: list[Result] = []

    for variant in variants:
        result = Result(variant.name)
        # (session -> model) and the set of warm (model, session) prefixes.
        session_model: dict[str, str] = {}
        warm_prefixes: set[tuple[str, str]] = set()

        for record in records:
            if record.get("outcome") != "ok":
                continue

            model = variant.choose(record, {"session_model": session_model})
            if model is None:
                continue

            prefix = int(record.get("prefix_tokens_est", 0))
            volatile = int(record.get("volatile_tokens_est", 0))
            output = int(record.get("completion_tokens", 0))
            session = record.get("session_id") or record.get("trace_id")

            if variant.sticky and session in session_model:
                pinned = get_model(session_model[session])
                # Escalation-only: keep the warm model unless the variant wants
                # a strictly larger one.
                if pinned and pinned.tier >= model.tier:
                    model = pinned

            key = (model.key, session)
            warm = variant.cache_aware and key in warm_prefixes

            cost, read, write = _cost(model, prefix, volatile, output, warm)
            result.total_usd += cost
            result.requests += 1
            result.cache_read_tokens += read
            result.cache_write_tokens += write
            result.model_mix[model.key] += 1

            warm_prefixes.add(key)
            session_model[session] = model.key

        results.append(result)

    return results


def _format(results: list[Result]) -> str:
    if not results:
        return "no usable records (need at least one with outcome=ok)"

    baseline = next((r for r in results if r.name == "as_recorded"), results[0])
    lines = [
        f"{'variant':<30} {'requests':>9} {'total $':>12} {'$/req':>10} "
        f"{'vs recorded':>12}  model mix",
        "-" * 110,
    ]
    for r in results:
        delta = (
            f"{(r.total_usd - baseline.total_usd) / baseline.total_usd * 100:+.1f}%"
            if baseline.total_usd
            else "n/a"
        )
        mix = ", ".join(f"{k}={v}" for k, v in sorted(r.model_mix.items()))
        lines.append(
            f"{r.name:<30} {r.requests:>9} {r.total_usd:>12.4f} "
            f"{r.per_request_usd:>10.5f} {delta:>12}  {mix}"
        )

    lines.append("")
    lines.append(
        "Reminder: replay holds output length constant across models. A weaker "
        "model that needs more turns will look cheaper here than it is."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay recorded traffic against routing policies")
    parser.add_argument("records", help="path to records.db (SQLite) or records.jsonl")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    try:
        if args.records.endswith(".jsonl"):
            with open(args.records, encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh if line.strip()]
        else:
            import os

            if not os.path.exists(args.records):
                raise FileNotFoundError(args.records)
            from ..observability.db import SqliteRecordSink

            sink = SqliteRecordSink(args.records)
            try:
                records = sink.read_all()
            finally:
                sink.close()
    except FileNotFoundError:
        print(f"no records at {args.records} — send some traffic first", file=sys.stderr)
        return 1

    results = replay(records)

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "variant": r.name,
                        "requests": r.requests,
                        "total_usd": round(r.total_usd, 6),
                        "per_request_usd": round(r.per_request_usd, 8),
                        "cache_read_tokens": r.cache_read_tokens,
                        "cache_write_tokens": r.cache_write_tokens,
                        "model_mix": dict(r.model_mix),
                    }
                    for r in results
                ],
                indent=2,
            )
        )
    else:
        print(_format(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
