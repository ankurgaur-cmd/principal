"""Per-(model, intent) reputation — routing on observed quality, not just price.

The router already prices a request. What it could not do was notice that the
model it keeps picking is *failing* at this particular kind of work. This closes
that loop: every response is graded by ``quality.assess``, the outcome is folded
in here, and a model that keeps producing unusable answers for an intent becomes
more expensive in the router's eyes until it stops winning.

Four design decisions carry most of the weight:

**1. Reputation is per (model, intent), never per model.**
A small model can be excellent at classification and hopeless at code review.
A single global score for a model averages those into a number that is wrong for
both. The intent is the unit of work, so it is the unit of reputation.

**2. The penalty is expected-cost, not an arbitrary weight.**
If a model succeeds a fraction ``s`` of the time, you need ``1/s`` attempts on
average to get one usable answer, so its honest cost is ``cost / s``. That makes
the adjustment comparable to price rather than a tunable fudge factor — a model
that fails a third of the time really does cost ~1.5x its sticker price.

**3. No evidence means no adjustment.**
Below ``min_samples`` observations the multiplier is exactly 1.0. Penalising a
model for one bad response would be superstition, and it would make routing
depend on the order requests happened to arrive in.

**4. Effort composes with retries; it does not compete with them.**
``1/s`` counts one kind of effort: how many attempts to get a usable answer. It
misses everything that made a "successful" task expensive — a fallback, a
truncated first draft, four turns, an escalation. ``effort.EffortModel`` scores
those in the same unit (extra ideal calls) and the two multiply:

    multiplier = (1 + mean_extra_effort) / success_rate

With no effort signals recorded the left factor is exactly 1.0, so this changes
nothing until there is evidence to change it.

**5. Exploration is mandatory, not optional.**
A model penalised out of contention never gets traffic, so it never gets new
observations, so it can never recover — its reputation is frozen at its worst
moment. A fixed fraction of requests therefore ignore the penalty entirely, which
is what lets a model that has been fixed climb back. Without this the feedback
loop is a ratchet.

Reputation is in-memory and resets on restart, matching the health monitor. The
durable history is the JSONL record; this is the hot view over it.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict, deque

from .effort import EffortModel, EffortObservation, Norms

log = logging.getLogger(__name__)


class Reputation:
    def __init__(
        self,
        *,
        window: int = 50,
        min_samples: int = 5,
        max_penalty: float = 4.0,
        exploration_rate: float = 0.05,
        rng: random.Random | None = None,
        effort: EffortModel | None = None,
    ):
        self._window = window
        self._min_samples = min_samples
        self._max_penalty = max_penalty
        self._exploration = exploration_rate
        self._rng = rng or random.Random()
        # (model_key, intent) -> recent outcomes, True = usable answer
        self._outcomes: dict[tuple[str, str], deque[bool]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        # Extra effort, in ideal-calls, alongside the binary outcome.
        self.effort = effort or EffortModel()
        self._effort: dict[tuple[str, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )
        # Per-intent norms, so effort is judged against comparable work rather
        # than an absolute threshold that would punish whichever model gets the
        # hard problems.
        self._norm_tokens: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=window))
        self._norm_latency: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._norm_turns: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=window))

    # -- observation --------------------------------------------------------
    def record(self, model_key: str, intent: str, ok: bool) -> None:
        self._outcomes[(model_key, intent)].append(ok)
        if not ok:
            log.info("quality miss recorded: %s on intent '%s'", model_key, intent)

    def norms_for(self, intent: str) -> Norms:
        """What this kind of work normally costs.

        Returns empty norms until an intent has history — signals that need a
        norm then report "no opinion" rather than treating an unknown as zero.
        """
        tokens = self._norm_tokens.get(intent) or ()
        latency = self._norm_latency.get(intent) or ()
        turns = self._norm_turns.get(intent) or ()
        return Norms(
            completion_tokens=sum(tokens) / len(tokens) if tokens else None,
            latency_ms=sum(latency) / len(latency) if latency else None,
            turns=sum(turns) / len(turns) if turns else None,
            samples=len(tokens),
        )

    def record_effort(self, obs: EffortObservation) -> dict:
        """Score one observation and fold it in. Returns the itemised breakdown.

        The norms are updated *after* scoring, for the same reason the latency
        baselines are: a sample judged against a norm it has already moved
        drags the norm towards itself and looks less exceptional than it is.
        """
        scored = self.effort.score(obs, self.norms_for(obs.intent))
        self._effort[(obs.model_key, obs.intent)].append(scored["extra_effort"])

        if obs.completion_tokens:
            self._norm_tokens[obs.intent].append(obs.completion_tokens)
        if obs.latency_ms:
            self._norm_latency[obs.intent].append(obs.latency_ms)
        if obs.turns_to_goal:
            self._norm_turns[obs.intent].append(obs.turns_to_goal)

        if scored["extra_effort"] > 0:
            log.info(
                "effort recorded: %s on '%s' cost %.2f extra calls (%s)",
                obs.model_key, obs.intent, scored["extra_effort"],
                ", ".join(c["signal"] for c in scored["contributions"]),
            )
        return scored

    def mean_effort(self, model_key: str, intent: str) -> float:
        """Mean extra effort, or 0.0 without enough evidence to say."""
        seen = self._effort.get((model_key, intent))
        if not seen or len(seen) < self._min_samples:
            return 0.0
        return sum(seen) / len(seen)

    # -- scoring ------------------------------------------------------------
    def sample_count(self, model_key: str, intent: str) -> int:
        return len(self._outcomes.get((model_key, intent), ()))

    def success_rate(self, model_key: str, intent: str) -> float | None:
        """Observed success rate, or None when there is not enough evidence."""
        outcomes = self._outcomes.get((model_key, intent))
        if not outcomes or len(outcomes) < self._min_samples:
            return None
        return sum(outcomes) / len(outcomes)

    def multiplier(self, model_key: str, intent: str) -> float:
        """Cost multiplier from observed quality and effort. 1.0 = no adjustment.

        Two independent factors that multiply because they compound: retries
        repeat the whole task, and effort is what each attempt costs beyond one
        clean call. Either alone can be 1.0, and with no evidence both are.
        """
        effort_factor = 1.0 + self.mean_effort(model_key, intent)

        rate = self.success_rate(model_key, intent)
        if rate is None:
            retry_factor = 1.0
        elif rate <= 0:
            return self._max_penalty
        else:
            retry_factor = 1.0 / rate

        # Capped so a bad patch cannot exile a model permanently.
        return min(effort_factor * retry_factor, self._max_penalty)

    def should_explore(self) -> bool:
        """Whether to ignore reputation for this request.

        Called once per routing decision, not once per candidate — exploring
        per-candidate would scramble the comparison between them.
        """
        return self._rng.random() < self._exploration

    # -- reporting ----------------------------------------------------------
    def snapshot(self) -> list[dict]:
        rows = []
        for (model_key, intent), outcomes in sorted(self._outcomes.items()):
            n = len(outcomes)
            rate = self.success_rate(model_key, intent)
            rows.append(
                {
                    "model": model_key,
                    "intent": intent,
                    "samples": n,
                    "failures": sum(1 for o in outcomes if not o),
                    "success_rate": round(rate, 3) if rate is not None else None,
                    "multiplier": round(self.multiplier(model_key, intent), 3),
                    "mean_extra_effort": round(self.mean_effort(model_key, intent), 3),
                    "effort_samples": len(self._effort.get((model_key, intent), ())),
                    # Says plainly why a model is or is not being adjusted.
                    "status": (
                        "insufficient evidence"
                        if rate is None
                        else "penalised"
                        if rate < 1.0
                        else "clean"
                    ),
                    "needs": max(0, self._min_samples - n),
                }
            )
        return rows

    def reset(self) -> None:
        self._outcomes.clear()
        self._effort.clear()
        self._norm_tokens.clear()
        self._norm_latency.clear()
        self._norm_turns.clear()
