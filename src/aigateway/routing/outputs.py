"""Learned expected-output volume, per intent.

The router's cost score needs an output forecast, and the static table it
shipped with (600 tokens for low effort, 1200 for medium, …) encodes a guess
about *effort* when the real driver is the *task*: classification answers in a
sentence at any effort, review runs long at any effort. Every served request
carries the correction — its actual completion tokens — so use it.

Median over a rolling window, deliberately: completion counts are heavy-tailed
(one truncation-length outlier should not double every future estimate), and a
window rather than a running mean lets the forecast follow a workload that
changes shape. Below ``min_samples`` the estimator abstains and the static
table stands — guessing from two observations is how estimators lose trust.

In-memory, like the latency baselines: it relearns in minutes after a restart
and drifts with recent traffic, which for a forecast is a feature.
"""

from __future__ import annotations

from collections import defaultdict, deque


class OutputEstimator:
    def __init__(self, window: int = 200, min_samples: int = 8):
        self._min = min_samples
        self._samples: dict[str, deque[int]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def record(self, intent: str, completion_tokens: int) -> None:
        if intent and completion_tokens > 0:
            self._samples[intent].append(completion_tokens)

    def expected(self, intent: str) -> int | None:
        """Median completion tokens for this intent, or None to abstain."""
        samples = self._samples.get(intent)
        if not samples or len(samples) < self._min:
            return None
        ordered = sorted(samples)
        return ordered[len(ordered) // 2]

    def snapshot(self) -> dict[str, dict]:
        return {
            intent: {
                "samples": len(s),
                "expected": self.expected(intent),
            }
            for intent, s in self._samples.items()
        }
