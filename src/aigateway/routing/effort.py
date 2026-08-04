"""What did it actually cost to reach the goal?

The router prices a request. ``reputation`` then adjusts that price by how often
a model produces a usable answer, on the principle that a model succeeding a
fraction ``s`` of the time really costs ``cost / s`` because you need ``1/s``
attempts. That is an effort measure — it just only counts one kind of effort.

This module generalises it. A task that "succeeded" after four turns, two
truncated answers and an escalation to a bigger model did not cost one call. The
sticker price of the winning call is the least interesting number in that story,
and a router that optimises it is optimising the wrong thing.

Effort is expressed in **extra ideal calls**: 0.0 means the task landed first
time with nothing wasted, 1.0 means it cost one whole extra call's worth of work.
Keeping that unit is the point — it composes with the existing retry model
instead of competing with it:

    multiplier = (1 + mean_extra_effort) / success_rate

which degenerates to exactly today's ``1/s`` when no effort signal fires, so
turning this on changes nothing until there is evidence to change it.

Two hazards, both of which make routing *worse* if ignored:

**Confounding.** Hard work legitimately takes more turns and more tokens. Scoring
raw effort punishes whichever model gets handed the hard problems, which is
precisely backwards — it would route hard tasks to models that have never seen
one. Every signal is therefore normalised against the *same intent's* observed
norm, never an absolute threshold. This is the same lesson as the latency
baselines: only compare like with like.

**Attribution.** Session-level effort spans several requests that may have been
served by different models. Blaming the model that happened to answer the last
turn is arbitrary. Signals declare their attribution: ``call`` signals are
charged to the model that served that call; ``session`` signals are charged
proportionally to the models that participated, and a signal that cannot be
attributed honestly returns ``None`` rather than guessing.

The signal table is deliberately open. Each entry is data — a name, a weight, an
attribution rule and a function — so adding "the user re-asked the same question"
later means appending a row, not editing the scorer. Signals that need data the
gateway does not yet collect are registered anyway, returning ``None``: an empty
column that names what is missing is more useful than a signal that quietly is
not there.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# No signal may claim more than this much extra effort on its own. One bad
# measurement should not be able to exile a model by itself.
MAX_SIGNAL_EFFORT = 3.0

# Total effort ceiling, for the same reason at the aggregate level.
MAX_TOTAL_EFFORT = 6.0


@dataclass
class EffortObservation:
    """Everything known about one completed unit of work.

    Fields default to "nothing to report" rather than to a plausible number.
    A signal that cannot tell the difference between *no effort* and *no data*
    is a signal that invents evidence.
    """

    model_key: str
    intent: str
    usable: bool = True

    # -- what the gateway measures directly on this call ------------------
    attempts: int = 1  # calls made inside this request, including fallbacks
    completion_tokens: int = 0
    visible_tokens: int = 0  # tokens that reached the user
    truncated: bool = False
    empty: bool = False
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    # -- what only the caller knows, reported through the feedback endpoint --
    turns_to_goal: int | None = None
    user_reasked: bool | None = None
    user_rejected: bool | None = None
    manual_escalation: bool | None = None
    edit_distance: float | None = None  # 0..1, how much the human rewrote it

    # -- the open slot ----------------------------------------------------
    # Anything a deployment wants to measure that this file has not thought of.
    # A signal can read it by name without any change here.
    extras: dict[str, float] = field(default_factory=dict)


@dataclass
class Norms:
    """Per-intent expectations, so effort is judged relative to the work.

    Populated from observed traffic. Until an intent has norms, signals that
    need one return ``None`` — which is the honest answer, not zero effort.
    """

    completion_tokens: float | None = None
    turns: float | None = None
    latency_ms: float | None = None
    samples: int = 0


# `call` — charge to the model that served this call.
# `session` — spans several calls; charge proportionally to participants.
ATTRIBUTION = ("call", "session")


@dataclass(frozen=True)
class EffortSignal:
    """One way a task can cost more than one clean call.

    ``measure`` returns extra effort in ideal-calls, or ``None`` for "no
    opinion" — missing data, or a norm not yet learned. None is not zero.
    """

    name: str
    describe: str
    weight: float
    attribution: str
    measure: Callable[[EffortObservation, Norms], float | None]
    enabled: bool = True
    needs: str = ""  # what data this is still waiting on, if any

    def evaluate(self, obs: EffortObservation, norms: Norms) -> float | None:
        if not self.enabled:
            return None
        try:
            raw = self.measure(obs, norms)
        except Exception:  # a bad signal must never break routing
            log.exception("effort signal %r failed; ignoring it", self.name)
            return None
        if raw is None:
            return None
        return max(0.0, min(raw * self.weight, MAX_SIGNAL_EFFORT))


# --------------------------------------------------------------------------
# The measurable signals: things the gateway already knows on every request.
# --------------------------------------------------------------------------
def _retries(obs: EffortObservation, _: Norms) -> float | None:
    """Each fallback attempt is literally one extra call. No normalisation
    needed — this is not an estimate, it is a count."""
    return float(max(0, obs.attempts - 1))


def _wasted_call(obs: EffortObservation, _: Norms) -> float | None:
    """An empty answer is a whole call bought and thrown away.

    This is the reasoning-starvation case: the budget went on hidden reasoning
    and nothing came back. It costs exactly one call and delivers nothing.
    """
    return 1.0 if obs.empty else 0.0


def _truncation(obs: EffortObservation, _: Norms) -> float | None:
    """A cut-off answer needs a continuation. Half a call, roughly — the
    context is already warm, so the retry is cheaper than a cold one."""
    return 0.5 if (obs.truncated and not obs.empty) else 0.0


def _token_overrun(obs: EffortObservation, norms: Norms) -> float | None:
    """Spending three times the usual tokens for this *intent* is real cost.

    Normalised against the intent, never an absolute budget: 6,000 tokens is
    profligate for a classification and frugal for an architecture review.
    """
    if not norms.completion_tokens or not obs.completion_tokens:
        return None
    ratio = obs.completion_tokens / norms.completion_tokens
    return max(0.0, ratio - 1.0)  # 2x the norm costs one extra call


def _invisible_work(obs: EffortObservation, _: Norms) -> float | None:
    """Tokens billed that the user never saw.

    Reasoning tokens are legitimate, but a model that spends 90% of its budget
    thinking and 10% answering is charging you for work you cannot inspect.
    """
    if not obs.completion_tokens or not obs.visible_tokens:
        return None
    hidden = 1.0 - (obs.visible_tokens / obs.completion_tokens)
    return max(0.0, hidden - 0.5) * 2  # nothing below 50% hidden; 100% costs 1.0


def _latency_overrun(obs: EffortObservation, norms: Norms) -> float | None:
    """Waiting is effort. Weighted lightly: slow is not the same as wrong."""
    if not norms.latency_ms or not obs.latency_ms:
        return None
    return max(0.0, obs.latency_ms / norms.latency_ms - 1.0)


# --------------------------------------------------------------------------
# The reported signals: real effort the gateway cannot see by itself.
# Registered with `needs` set, returning None until a caller reports them.
# --------------------------------------------------------------------------
def _turns_to_goal(obs: EffortObservation, norms: Norms) -> float | None:
    """Turns above what this kind of task normally takes.

    Normalisation is doing the load-bearing work: a debugging session takes
    more turns than a translation, and charging the model for that would route
    every hard problem to whichever model has the least history.
    """
    if obs.turns_to_goal is None or not norms.turns:
        return None
    return max(0.0, (obs.turns_to_goal - norms.turns) / max(norms.turns, 1.0))


def _reask(obs: EffortObservation, _: Norms) -> float | None:
    """The user asked the same thing again — the answer did not land."""
    if obs.user_reasked is None:
        return None
    return 1.0 if obs.user_reasked else 0.0


def _rejection(obs: EffortObservation, _: Norms) -> float | None:
    """An explicit thumbs-down. The strongest signal available, and the rarest."""
    if obs.user_rejected is None:
        return None
    return 2.0 if obs.user_rejected else 0.0


def _escalation(obs: EffortObservation, _: Norms) -> float | None:
    """The user overrode the router and asked for a bigger model.

    That is the router being told, in the most direct way available, that it
    picked wrong — and the user paying for two calls to get one answer.
    """
    if obs.manual_escalation is None:
        return None
    return 1.5 if obs.manual_escalation else 0.0


def _human_edit(obs: EffortObservation, _: Norms) -> float | None:
    """How much of the answer the human rewrote before using it.

    A response edited beyond recognition was not a usable answer; it was a
    first draft the human finished. That is effort, and it is invisible to
    every other signal here.
    """
    if obs.edit_distance is None:
        return None
    return max(0.0, obs.edit_distance)


DEFAULT_SIGNALS: list[EffortSignal] = [
    EffortSignal(
        "retries", "Fallback attempts inside one request", 1.0, "call", _retries
    ),
    EffortSignal(
        "wasted_call", "Empty answer — a call bought and thrown away", 1.0, "call",
        _wasted_call,
    ),
    EffortSignal(
        "truncation", "Answer cut off, needs a continuation", 1.0, "call", _truncation
    ),
    EffortSignal(
        "token_overrun", "Tokens well above this intent's norm", 0.5, "call",
        _token_overrun,
    ),
    EffortSignal(
        "invisible_work", "Budget spent on reasoning the user never sees", 0.5, "call",
        _invisible_work,
    ),
    EffortSignal(
        "latency_overrun", "Slower than this intent's norm", 0.25, "call",
        _latency_overrun,
    ),
    EffortSignal(
        "turns_to_goal", "Turns above the norm for this intent", 1.0, "session",
        _turns_to_goal, needs="caller reports turns via POST /admin/effort",
    ),
    EffortSignal(
        "reask", "User asked the same thing again", 1.0, "session", _reask,
        needs="caller reports a re-ask via POST /admin/effort",
    ),
    EffortSignal(
        "rejection", "Explicit thumbs-down", 1.0, "session", _rejection,
        needs="caller reports rejection via POST /admin/effort",
    ),
    EffortSignal(
        "escalation", "User overrode the router for a bigger model", 1.0, "session",
        _escalation, needs="caller reports escalation via POST /admin/effort",
    ),
    EffortSignal(
        "human_edit", "How much the human rewrote the answer", 1.0, "session",
        _human_edit, needs="caller reports edit distance via POST /admin/effort",
    ),
]


class EffortModel:
    """The open table, and the score it produces.

    Signals are held as data so a deployment can add, reweight or disable one
    without touching the scoring path.
    """

    def __init__(self, signals: list[EffortSignal] | None = None):
        self._signals = list(signals if signals is not None else DEFAULT_SIGNALS)

    # -- the open table -----------------------------------------------------
    def register(self, signal: EffortSignal) -> None:
        """Add or replace a signal by name."""
        if signal.attribution not in ATTRIBUTION:
            raise ValueError(
                f"attribution must be one of {ATTRIBUTION}, got {signal.attribution!r}"
            )
        self._signals = [s for s in self._signals if s.name != signal.name]
        self._signals.append(signal)

    def set_weight(self, name: str, weight: float) -> None:
        self._replace(name, weight=weight)

    def set_enabled(self, name: str, enabled: bool) -> None:
        self._replace(name, enabled=enabled)

    def _replace(self, name: str, **changes) -> None:
        from dataclasses import replace

        for i, s in enumerate(self._signals):
            if s.name == name:
                self._signals[i] = replace(s, **changes)
                return
        raise KeyError(name)

    @property
    def signals(self) -> list[EffortSignal]:
        return list(self._signals)

    # -- scoring ------------------------------------------------------------
    def score(self, obs: EffortObservation, norms: Norms | None = None) -> dict:
        """Total extra effort, and the itemised reason for it.

        The breakdown is not decoration. An effort penalty that cannot be
        itemised is indistinguishable from a grudge, and nobody can act on it.
        """
        norms = norms or Norms()
        contributions, silent = [], []
        total = 0.0

        for signal in self._signals:
            value = signal.evaluate(obs, norms)
            if value is None:
                silent.append(
                    {
                        "signal": signal.name,
                        "describe": signal.describe,
                        "reason": signal.needs or "no data for this request",
                    }
                )
                continue
            if value > 0:
                total += value
                contributions.append(
                    {
                        "signal": signal.name,
                        "describe": signal.describe,
                        "effort": round(value, 3),
                        "attribution": signal.attribution,
                    }
                )

        total = min(total, MAX_TOTAL_EFFORT)
        contributions.sort(key=lambda c: -c["effort"])
        return {
            "extra_effort": round(total, 3),
            "contributions": contributions,
            "silent": silent,
            "capped": total >= MAX_TOTAL_EFFORT,
        }

    def table(self) -> list[dict]:
        """The signal table itself, for the console and for `/admin/effort`."""
        return [
            {
                "name": s.name,
                "describe": s.describe,
                "weight": s.weight,
                "attribution": s.attribution,
                "enabled": s.enabled,
                "status": "measured" if not s.needs else "awaiting data",
                "needs": s.needs,
            }
            for s in self._signals
        ]
