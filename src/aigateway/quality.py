"""Response quality checks — did the routing decision actually work out?

The gateway can prove it picked a cheaper model. It cannot, on its own, tell you
the cheaper model was *good enough* — and without that signal every "saving" is
unfalsifiable. This module is the missing half.

Two tiers, deliberately separated by cost:

* **Deterministic checks (free, always on).** Truncation, empty output, invalid
  JSON against a requested schema, malformed tool arguments, a cache that was
  supposed to be warm and wasn't. These catch the failures that actually happen
  in production, and they cost nothing.
* **LLM-as-judge (paid, opt-in).** A cheap model scores the answer against the
  request. Useful, but it is another model call with its own failure modes, so
  it is off by default and never blocks a response.

A failure here is a **routing signal**, not just a log line: if the tier we
chose produced a truncated or empty answer, that is evidence the router was
wrong, and it is recorded as such so the replay harness can weigh it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# Reasoning-capable models bill hidden reasoning tokens against the same output
# budget as the visible answer. Below these ceilings the reasoning consumes the
# entire allowance and leaves nothing for a reply — the response comes back
# empty with finish_reason "length", which looks like a gateway bug and is not.
#
# The floor scales with effort, and this is the part that is easy to get wrong:
# a single flat floor was too low for high-effort work, so requests that cleared
# it still came back empty. Reasoning depth is what consumes the budget, so the
# threshold has to move with it.
REASONING_FLOOR_BY_EFFORT = {
    "low": 300,
    "medium": 700,
    "high": 1800,
    "xhigh": 3500,
    "max": 6000,
}
# Kept for the medium default and for callers that do not know the effort.
REASONING_OUTPUT_FLOOR = REASONING_FLOOR_BY_EFFORT["medium"]

# Effort levels ordered cheapest-first, for stepping down to fit a budget.
EFFORT_LADDER = ["low", "medium", "high", "xhigh", "max"]


def effort_that_fits(effort: str, max_tokens: int) -> str:
    """Highest effort whose reasoning still leaves room for an answer.

    Respects the caller's token cap rather than silently raising it — their
    budget is their decision; how much of it we spend on reasoning is ours.
    """
    if effort not in EFFORT_LADDER:
        return effort
    for level in reversed(EFFORT_LADDER[: EFFORT_LADDER.index(effort) + 1]):
        if max_tokens >= REASONING_FLOOR_BY_EFFORT[level]:
            return level
    return "low"


@dataclass
class Check:
    id: str
    level: str  # pass | warn | fail
    title: str
    detail: str


@dataclass
class QualityReport:
    checks: list[Check] = field(default_factory=list)
    judge: dict | None = None

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.level == "fail"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.level == "warn"]

    @property
    def verdict(self) -> str:
        if self.failures:
            return "fail"
        return "warn" if self.warnings else "pass"

    @property
    def routing_ok(self) -> bool:
        """Whether the routing decision is defensible given what came back."""
        return not self.failures

    def summary(self) -> dict:
        return {
            "verdict": self.verdict,
            "routing_ok": self.routing_ok,
            "checks": [c.__dict__ for c in self.checks],
            "judge": self.judge,
        }


def assess(canonical, response, decision) -> QualityReport:
    """Deterministic checks over one response. No model calls, no cost."""
    report = QualityReport()
    text = (response.text or "").strip()
    has_tools = bool(response.tool_calls)
    finish = response.finish_reason

    floor = REASONING_FLOOR_BY_EFFORT.get(decision.effort, REASONING_OUTPUT_FLOOR)

    # --- did we get anything at all? -------------------------------------
    if not text and not has_tools:
        if finish == "length":
            report.checks.append(
                Check(
                    "reasoning_starved",
                    "fail",
                    "Empty answer — the output budget was spent on reasoning",
                    f"The model used all {response.usage.completion_tokens} output "
                    f"tokens on internal reasoning and produced no visible reply. "
                    f"Raise max_tokens above ~{floor} for effort "
                    f"'{decision.effort}', or lower the effort so more of the "
                    f"budget goes to the answer.",
                )
            )
        else:
            report.checks.append(
                Check(
                    "empty_response",
                    "fail",
                    "Empty response",
                    f"No text and no tool calls (finish_reason={finish}).",
                )
            )
    elif finish == "length":
        report.checks.append(
            Check(
                "truncated",
                "fail",
                "Answer was cut off",
                "Output hit max_tokens mid-sentence. The result is incomplete "
                "regardless of which model produced it — raise max_tokens.",
            )
        )
    else:
        report.checks.append(
            Check("complete", "pass", "Answer completed normally", f"finish_reason={finish}")
        )

    # --- did it honour a requested shape? ---------------------------------
    if canonical.response_schema:
        try:
            parsed = json.loads(text) if text else None
            if parsed is None:
                raise ValueError("empty")
            required = canonical.response_schema.get("required", [])
            missing = [k for k in required if k not in parsed]
            if missing:
                report.checks.append(
                    Check(
                        "schema_incomplete",
                        "fail",
                        "Structured output is missing required fields",
                        f"Missing: {', '.join(missing)}. The chosen model may be "
                        "too small for this schema.",
                    )
                )
            else:
                report.checks.append(
                    Check("schema_valid", "pass", "Structured output matched the schema", "")
                )
        except (json.JSONDecodeError, ValueError):
            report.checks.append(
                Check(
                    "schema_invalid",
                    "fail",
                    "Structured output was not valid JSON",
                    "A JSON schema was requested but the reply could not be parsed.",
                )
            )

    # --- are the tool calls usable? ---------------------------------------
    for call in response.tool_calls:
        name = call.get("function", {}).get("name", "?")
        raw = call.get("function", {}).get("arguments", "")
        try:
            json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            report.checks.append(
                Check(
                    "tool_args_invalid",
                    "fail",
                    f"Tool call '{name}' has unparseable arguments",
                    "A weaker model producing malformed tool JSON is a classic "
                    "sign the tier was set too low.",
                )
            )
            break
    else:
        if has_tools:
            report.checks.append(
                Check(
                    "tool_args_valid",
                    "pass",
                    f"{len(response.tool_calls)} tool call(s) parsed",
                    "",
                )
            )

    # --- did caching do what the router assumed? --------------------------
    if decision.cache_state == "warm_read" and response.usage.cache_read_tokens == 0:
        report.checks.append(
            Check(
                "cache_missed",
                "warn",
                "Expected a cache hit and got none",
                "The router priced this request assuming a warm cache. Something "
                "invalidated the prefix — a timestamp, a reordered tool, or an "
                "edited system prompt.",
            )
        )
    elif response.usage.cache_read_tokens:
        report.checks.append(
            Check(
                "cache_hit",
                "pass",
                f"{response.usage.cache_read_tokens:,} tokens read from cache",
                "Billed at roughly a tenth of the normal input rate.",
            )
        )

    # --- suspiciously thin answer for an expensive intent -----------------
    heavy_intents = {"code_review", "architecture", "hard_debug", "long_horizon_agentic"}
    if decision.intent in heavy_intents and text and len(text) < 200 and not has_tools:
        report.checks.append(
            Check(
                "thin_for_intent",
                "warn",
                "Very short answer for a demanding task",
                f"{len(text)} characters for a '{decision.intent}' request. Worth "
                "checking the answer is actually substantive.",
            )
        )

    return report


JUDGE_SYSTEM = (
    "You grade an AI assistant's reply for a routing system that needs to know "
    "whether a cheaper model was good enough. Judge only whether the reply "
    "adequately answers the request. Ignore style. Be strict about factual or "
    "structural failures and lenient about brevity."
)

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "description": "1 (unusable) to 5 (fully adequate)"},
        "adequate": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["score", "adequate", "reason"],
    "additionalProperties": False,
}


async def judge(registry, model_key: str, request_text: str, answer: str) -> dict | None:
    """Optional LLM grader. Off by default — it is a real, billable call.

    Never raises: a grader failure must not affect the response it was grading.
    """
    if not answer.strip():
        return None
    try:
        provider = registry.for_model(model_key)
        return await provider.classify(
            model_key=model_key,
            system=JUDGE_SYSTEM,
            text=f"REQUEST:\n{request_text[:2000]}\n\nREPLY:\n{answer[:2000]}",
            schema=JUDGE_SCHEMA,
        )
    except Exception:
        return None
