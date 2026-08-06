"""System alerts: the gateway saying it needs a human.

Most failures in this gateway are the caller's problem — a bad request, a budget
exceeded, a schema that will not parse. This module is for the other kind: the
gateway is healthy, the request is fine, and there is **nothing left to serve it
with**. Every model is switched off, or unhealthy, or has no credentials.

That situation has two audiences and they need opposite things:

* **The operator** needs the technical cause and the specific remedy, loudly and
  immediately — a red flag on the console, not a line in a log file nobody is
  reading at 3am.
* **The end user** needs to be told, in plain language, that something is wrong
  on our side and not theirs, without a wall of model names and vendor jargon.

Serving one message to both is how you get an operator who cannot diagnose and a
user who thinks they broke something. So an alert carries both: ``detail`` for
the operator and ``user_message`` for whoever is on the other end of the agent.

Alerts are **latched**: raised when the condition is first seen, and cleared
explicitly when a request succeeds. A transient outage that self-heals still
leaves a record of having happened, because "it was down for ten minutes and
came back" is exactly the thing an operator needs to know and the thing a
purely-live status view hides.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# How many recent alerts to keep. Enough to show a flapping pattern, not so many
# that the endpoint becomes a log reader.
HISTORY = 50

CRITICAL = "critical"
WARNING = "warning"


@dataclass
class Alert:
    """One thing wrong, described for two audiences."""

    code: str
    severity: str
    title: str
    detail: str  # for the operator: what and why, in technical terms
    user_message: str  # for the end user: plain language, no jargon
    remedy: str = ""  # the specific action that fixes it
    needs_support: bool = False  # true when no automatic recovery is possible
    raised_at: float = field(default_factory=time.time)
    cleared_at: float | None = None
    occurrences: int = 1

    @property
    def active(self) -> bool:
        return self.cleared_at is None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "user_message": self.user_message,
            "remedy": self.remedy,
            "needs_support": self.needs_support,
            "active": self.active,
            "raised_at": round(self.raised_at, 3),
            "cleared_at": round(self.cleared_at, 3) if self.cleared_at else None,
            "occurrences": self.occurrences,
            "age_seconds": round(time.time() - self.raised_at, 1),
        }


def no_models_available(cause: str, detail: str, remedy: str) -> Alert:
    """The headline alert: nothing in the catalog can serve traffic.

    ``cause`` decides both the wording and whether this needs a human. An
    operator who switched everything off does not need paging — they need
    reminding. A fleet where every breaker has tripped does.
    """
    # Wording rules for `user_message`, learned by writing a worse version
    # first: one short sentence, calm, and no protesting. The earlier copy said
    # "Someone has turned it off deliberately — it is not a fault, and nothing
    # you did caused it", which is three clauses of reassurance nobody asked
    # for, and insisting a user is not to blame is the fastest way to suggest
    # they might be. State the situation, say what happens next, stop.
    #
    # The operator's version keeps every detail. Only this one is trimmed.
    friendly = {
        "all_switched_off": (
            "Paused by an operator",
            "The service is paused. Please try again shortly.",
            False,
        ),
        "all_unhealthy": (
            "Every model is failing health checks",
            "We can't reach the service right now. Please try again in a few minutes.",
            True,
        ),
        "no_credentials": (
            "No provider credentials configured",
            "The service isn't set up yet.",
            True,
        ),
        "pinned_model_switched_off": (
            "A pinned model is switched off",
            "That model is unavailable right now.",
            False,
        ),
        "no_capable_model": (
            "No model meets this request's requirements",
            "We can't handle this request as it stands — try shortening it.",
            False,
        ),
    }
    title, user_message, needs_support = friendly.get(
        cause,
        (
            "No model is available",
            "The service is unavailable right now. Please try again shortly.",
            True,
        ),
    )
    return Alert(
        code=f"no_models_available:{cause}",
        severity=CRITICAL,
        title=title,
        detail=detail,
        user_message=user_message,
        remedy=remedy,
        needs_support=needs_support,
    )


class AlertCentre:
    """Raise, clear and report system alerts.

    In-memory and per-process, matching the health monitor and the fleet view.
    The durable record is the JSONL; this is the live signal the console polls.
    """

    def __init__(self) -> None:
        self._active: dict[str, Alert] = {}
        self._history: list[Alert] = []

    def raise_alert(self, alert: Alert) -> Alert:
        """Raise, or bump the count if it is already up.

        Re-raising an active alert must not reset its clock: an outage that has
        lasted twenty minutes should say twenty minutes, not restart at zero
        every time another request hits it.
        """
        existing = self._active.get(alert.code)
        if existing is not None:
            existing.occurrences += 1
            return existing

        self._active[alert.code] = alert
        self._history.append(alert)
        del self._history[:-HISTORY]
        log.error(
            "SYSTEM ALERT [%s] %s — %s (remedy: %s)",
            alert.severity, alert.title, alert.detail, alert.remedy or "n/a",
        )
        return alert

    def clear(self, prefix: str = "") -> int:
        """Clear active alerts whose code starts with `prefix`.

        Called when a request succeeds: proof that whatever we were complaining
        about is no longer true. Recovery is detected from real traffic rather
        than from a probe, for the same reason the circuit breaker is.
        """
        cleared = 0
        for code in [c for c in self._active if c.startswith(prefix)]:
            alert = self._active.pop(code)
            alert.cleared_at = time.time()
            cleared += 1
            log.info("alert cleared: %s (was up %.0fs)", code, alert.cleared_at - alert.raised_at)
        return cleared

    @property
    def active(self) -> list[Alert]:
        return sorted(self._active.values(), key=lambda a: a.raised_at)

    def snapshot(self) -> dict:
        active = self.active
        return {
            # The single boolean the console needs to decide whether to light up.
            "ok": not active,
            "needs_support": any(a.needs_support for a in active),
            "severity": (
                CRITICAL
                if any(a.severity == CRITICAL for a in active)
                else WARNING
                if active
                else None
            ),
            "active": [a.to_dict() for a in active],
            "recent": [a.to_dict() for a in reversed(self._history[-10:])],
        }
