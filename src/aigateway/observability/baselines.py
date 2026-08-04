"""Latency baselines: is *this* hop slow, compared to what?

A duration on its own tells you nothing. "1,847ms" is excellent for a heavy
model writing a cold cache and terrible for a rules classifier. The console
colours each stage against a baseline so the number becomes a judgement, and
this module is where that judgement is computed.

Three things matter and each of them is a way to get this wrong:

**Segmentation.** The baseline is per *comparable population*, never global. A
cold cache write is legitimately slower than a warm read; a heavy model is
legitimately slower than a light one. Pooling them paints every cold heavy
request red, and a dashboard that cries wolf gets ignored — which is worse than
no colour at all. Keys carry the things that legitimately change the expected
duration, so we only ever compare like with like.

**Confidence.** Mean and standard deviation over three samples are noise with a
decimal point. Below ``MIN_SAMPLES`` a stage reports the ``learning`` band and
is drawn neutral: we say we do not know yet rather than guessing in colour.

**Skew.** Latency distributions are right-skewed — a long tail of slow requests,
a hard floor at zero — so "2 sigma" does *not* mean "the slowest 2.3%" the way it
would for a normal distribution. In practice it flags rather more than that. The
bands are still computed in sigma because that is the agreed vocabulary, but
every baseline also reports p50/p95 so the skew is visible rather than implied,
and the percentile rank is reported alongside z because "slower than 87% of
recent runs" is what people actually mean when they ask if something is slow.

Baselines are learned from real traffic. Until a segment has enough samples it
falls back to a **prior** — a declared expectation, marked as such. Priors are
our own seeded estimates, not vendor commitments: the model vendors publish
*availability* SLAs, not latency SLAs, so anything here presented as a latency
target is an internal SLO and is labelled ``prior`` rather than ``sla`` unless an
operator configures one explicitly.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

# Below this many samples a segment reports `learning` and is drawn neutral.
# Mean and sigma over a handful of points are noise, and colouring them trains
# the operator to distrust the colour.
MIN_SAMPLES = 8

# Rolling window per segment. Long enough for a stable sigma, short enough that
# the baseline tracks a real regression instead of averaging it away for hours.
WINDOW = 300

# Band thresholds in standard deviations above the segment mean.
WARN_SIGMA = 1.0
CRITICAL_SIGMA = 2.0

# A deviation must be both statistically anomalous *and* materially slow.
#
# Sigma alone is not enough, and the live data made that obvious: gateway-local
# stages run in tens of microseconds with a sigma of ~0.01ms, so a perfectly
# ordinary 0.08ms route scores 3.7 sigma and lights up red. That is
# statistically correct and operationally worthless — nobody is paged over 30
# microseconds, and a dashboard that is permanently red on stages nobody cares
# about is the crying-wolf failure this module exists to avoid.
#
# So a band above `normal` also requires an absolute gap. 25ms is the rough
# threshold below which a human perceives no difference at all, which makes it
# the right floor for something whose entire job is to draw a human's eye.
MIN_MATERIAL_DEVIATION_MS = 25.0

# Declared expectations used until a segment has learned its own, in ms.
#
# These are seeded from measurements taken against this gateway, not from vendor
# documentation. Treat them as starting priors that real traffic replaces, and
# note the deliberate absence of anything claiming to be a vendor latency SLA —
# see the module docstring.
PRIORS_MS: dict[str, float] = {
    "accepted": 5,
    "canonicalised": 10,
    "classified": 60,
    "budget": 10,
    "routed": 15,
    "cache": 25,
    # The upstream call, split by what the cache is doing, because a cold write
    # and a warm read are not the same operation.
    "served:cold_write": 9000,
    "served:warm_read": 5000,
    "served:uncached": 7000,
    "quality": 5,
    "hops": 5,
}

# How far a prior is trusted to spread. Without an observed sigma there is
# nothing to band against, so assume a wide, forgiving distribution: half the
# expected duration. This errs towards not flagging, which is the right error to
# make before you have data.
PRIOR_SPREAD = 0.5

def _r(v: float | None) -> float | None:
    return round(v, 2) if v is not None else None


BAND_LEARNING = "learning"
BAND_FAST = "fast"
BAND_NORMAL = "normal"
BAND_WARN = "warn"
BAND_CRITICAL = "critical"


@dataclass
class Segment:
    """Online statistics for one comparable population.

    Welford's algorithm rather than sum-of-squares: it is numerically stable,
    and it needs no second pass, which matters because this updates on the
    serving path of every request.
    """

    key: str
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    recent: deque[float] = field(default_factory=lambda: deque(maxlen=WINDOW))
    best_ms: float | None = None

    def update(self, value_ms: float) -> None:
        self.n += 1
        delta = value_ms - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value_ms - self.mean)
        self.recent.append(value_ms)
        if self.best_ms is None or value_ms < self.best_ms:
            self.best_ms = value_ms

    @property
    def stddev(self) -> float:
        # Sample standard deviation; undefined for a single observation.
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0

    def percentile(self, p: float) -> float | None:
        if not self.recent:
            return None
        ordered = sorted(self.recent)
        return ordered[min(int(len(ordered) * p), len(ordered) - 1)]

    def rank_of(self, value_ms: float) -> float:
        """Fraction of recent samples this value is faster than (0..1).

        Reported because "slower than 87% of recent runs" is what people mean
        when they ask whether something is slow, and unlike a z-score it does
        not quietly assume the distribution is symmetric.
        """
        if not self.recent:
            return 0.0
        slower_than = sum(1 for v in self.recent if v > value_ms)
        return round(slower_than / len(self.recent), 3)

    def to_dict(self) -> dict:
        # Everything published here is derived from the *rounded* sigma, so the
        # report cannot contradict itself: a segment showing "sigma 0.00" never
        # also shows thresholds computed from a sigma it did not print.
        sigma = round(self.stddev, 2)
        return {
            "key": self.key,
            "samples": self.n,
            # Two decimals throughout: gateway-local stages are sub-millisecond,
            # and rounding to 1dp published "sigma 0.0" for segments whose
            # thresholds were computed on a real, non-zero sigma.
            "mean_ms": round(self.mean, 2),
            "stddev_ms": sigma,
            "p50_ms": _r(self.percentile(0.50)),
            "p95_ms": _r(self.percentile(0.95)),
            "best_ms": round(self.best_ms, 2) if self.best_ms is not None else None,
            # With no spread there is no sigma to multiply, so a numeric
            # threshold here would be a lie: printing "warn >= 40, critical >= 40"
            # reads as "anything at the mean is critical", which is not what
            # `judge` does — it treats *any* deviation from an identical history
            # as signal. Report the rule instead of a misleading number.
            "warn_above_ms": (
                round(self.mean + WARN_SIGMA * sigma, 2) if sigma > 0 else None
            ),
            "critical_above_ms": (
                round(self.mean + CRITICAL_SIGMA * sigma, 2) if sigma > 0 else None
            ),
            "degenerate": self.n > 1 and sigma == 0.0,
            "confident": self.n >= MIN_SAMPLES,
        }


def segment_key(stage: str, *, model: str = "", cache_state: str = "") -> str:
    """The comparable population for a stage.

    The upstream call is keyed by model *and* cache state because both change
    the expected duration for legitimate reasons. Everything else is gateway-
    local work whose cost does not depend on either, so it keys on the stage
    alone and reaches a confident baseline far sooner.
    """
    if stage != "served":
        return stage
    state = (cache_state or "uncached").split(" ")[0]
    return f"served:{model or '?'}:{state}"


class LatencyBaselines:
    """Learned expectations for every stage, and the verdict on one observation.

    In-memory and rolling, like the fleet view: the JSONL record remains the
    durable history, and this is the hot view used to colour a live request.
    It resets on restart and re-learns within a few dozen requests.
    """

    def __init__(self) -> None:
        self._segments: dict[str, Segment] = {}

    def _prior_for(self, key: str) -> float | None:
        if key in PRIORS_MS:
            return PRIORS_MS[key]
        # served:<model>:<state> falls back to the per-state prior, since a
        # brand-new model has no history but the cache state still tells us
        # roughly what to expect.
        if key.startswith("served:"):
            return PRIORS_MS.get(f"served:{key.rsplit(':', 1)[-1]}")
        return None

    def observe(self, key: str, value_ms: float) -> None:
        self._segments.setdefault(key, Segment(key)).update(value_ms)

    def judge(self, key: str, value_ms: float) -> dict:
        """Band one observation against its segment.

        Always returns a verdict — an unknown segment is reported as `learning`
        with its prior, never as an error and never as a guess dressed up as a
        measurement. Callers should render `learning` neutrally.
        """
        seg = self._segments.get(key)
        prior = self._prior_for(key)

        if seg is None or seg.n < MIN_SAMPLES:
            expected = seg.mean if seg and seg.n else prior
            return {
                "band": BAND_LEARNING,
                "z": None,
                "rank": seg.rank_of(value_ms) if seg else None,
                "expected_ms": round(expected, 2) if expected else None,
                "samples": seg.n if seg else 0,
                "needs": MIN_SAMPLES,
                "source": "observed" if seg and seg.n else ("prior" if prior else "none"),
                "note": (
                    f"learning — {seg.n if seg else 0} of {MIN_SAMPLES} samples"
                    if prior is None
                    else f"learning — {seg.n if seg else 0} of {MIN_SAMPLES} samples, "
                    f"compared against a seeded prior of {_fmt_ms(prior)}"
                ),
            }

        sigma = seg.stddev
        if sigma <= 0:
            # Every sample identical: any deviation at all is the signal.
            z = 0.0 if value_ms == seg.mean else math.copysign(3.0, value_ms - seg.mean)
        else:
            z = (value_ms - seg.mean) / sigma

        # Statistically anomalous *and* materially slow. Either alone produces a
        # useless colour: sigma alone paints microsecond jitter red, and an
        # absolute threshold alone cannot tell a slow model from a fast one.
        gap = value_ms - seg.mean
        material = abs(gap) >= MIN_MATERIAL_DEVIATION_MS

        if z > CRITICAL_SIGMA and material:
            band = BAND_CRITICAL
        elif z > WARN_SIGMA and material:
            band = BAND_WARN
        elif z < -WARN_SIGMA and material:
            band = BAND_FAST
        else:
            band = BAND_NORMAL

        return {
            "band": band,
            "z": round(z, 2),
            "rank": seg.rank_of(value_ms),
            "expected_ms": round(seg.mean, 2),
            "stddev_ms": round(sigma, 2),
            "p50_ms": _r(seg.percentile(0.50)),
            "p95_ms": _r(seg.percentile(0.95)),
            "best_ms": _r(seg.best_ms),
            "samples": seg.n,
            "source": "observed",
            "record": bool(seg.best_ms is not None and value_ms <= seg.best_ms),
            "material": material,
            "note": _explain(band, z, seg, value_ms, material),
        }

    def observe_and_judge(self, key: str, value_ms: float) -> dict:
        """Judge against the baseline *before* folding this sample into it.

        Order matters: scoring a value against a baseline it has already moved
        is self-fulfilling, and it is exactly how an anomaly hides itself.
        """
        verdict = self.judge(key, value_ms)
        self.observe(key, value_ms)
        return verdict

    def snapshot(self) -> dict:
        segments = sorted(self._segments.values(), key=lambda s: s.key)
        return {
            "min_samples": MIN_SAMPLES,
            "warn_sigma": WARN_SIGMA,
            "critical_sigma": CRITICAL_SIGMA,
            "window": WINDOW,
            "segments": [s.to_dict() for s in segments],
            "priors_ms": PRIORS_MS,
        }


def _fmt_ms(v: float) -> str:
    """Gateway stages run in microseconds, upstream calls in seconds.

    A fixed precision reads as "the 0ms baseline" for the former and as noise
    for the latter, so the precision follows the magnitude.
    """
    if v >= 1000:
        return f"{v / 1000:.2f}s"
    if v >= 10:
        return f"{v:.0f}ms"
    return f"{v:.2f}ms"


def _explain(band: str, z: float, seg: Segment, value_ms: float, material: bool = True) -> str:
    rank = seg.rank_of(value_ms)
    if band == BAND_NORMAL and not material and abs(z) > WARN_SIGMA:
        # Say why it is not coloured, or the number and the colour look like
        # they disagree.
        return (
            f"{z:.1f} sigma from the {_fmt_ms(seg.mean)} baseline, but only "
            f"{_fmt_ms(abs(value_ms - seg.mean))} in absolute terms — too small to matter"
        )
    if band == BAND_CRITICAL:
        return (
            f"{z:.1f} sigma above the {_fmt_ms(seg.mean)} baseline — slower than "
            f"{rank:.0%} of the last {len(seg.recent)} runs"
        )
    if band == BAND_WARN:
        return (
            f"{z:.1f} sigma above the {_fmt_ms(seg.mean)} baseline — slower than "
            f"{rank:.0%} of recent runs"
        )
    if band == BAND_FAST:
        return f"{abs(z):.1f} sigma faster than the {_fmt_ms(seg.mean)} baseline"
    return f"within one sigma of the {_fmt_ms(seg.mean)} baseline"
