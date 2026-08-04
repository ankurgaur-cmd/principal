"""Latency baselines.

The console colours each pipeline stage from these verdicts, so a wrong verdict
is not a cosmetic bug — it is a dashboard that lies. The tests below are mostly
about the ways this goes wrong rather than the happy path: crying wolf on thin
data, pooling populations that are not comparable, and scoring a sample against
a baseline it has already moved.
"""

from __future__ import annotations

from aigateway.observability.baselines import (
    CRITICAL_SIGMA,
    MIN_SAMPLES,
    LatencyBaselines,
    Segment,
    segment_key,
)


def _train(b: LatencyBaselines, key: str, values: list[int]) -> None:
    for v in values:
        b.observe(key, v)


# -- the statistics --------------------------------------------------------
def test_welford_matches_the_textbook():
    seg = Segment("t")
    for v in (2, 4, 4, 4, 5, 5, 7, 9):
        seg.update(v)

    assert seg.n == 8
    assert seg.mean == 5.0
    # Sample (n-1) standard deviation of that set is exactly sqrt(32/7).
    assert round(seg.stddev, 6) == round((32 / 7) ** 0.5, 6)


def test_stddev_is_undefined_for_a_single_sample():
    seg = Segment("t")
    seg.update(100)
    assert seg.stddev == 0.0


def test_best_tracks_the_fastest_seen():
    seg = Segment("t")
    for v in (500, 120, 900):
        seg.update(v)
    assert seg.best_ms == 120


# -- confidence gating -----------------------------------------------------
def test_thin_data_reports_learning_rather_than_a_colour():
    """Mean and sigma over three samples are noise with a decimal point.
    Colouring them teaches the operator to distrust the colour."""
    b = LatencyBaselines()
    _train(b, "classified", [50] * (MIN_SAMPLES - 1))

    verdict = b.judge("classified", 9999)
    assert verdict["band"] == "learning"
    assert verdict["z"] is None
    assert verdict["samples"] == MIN_SAMPLES - 1
    assert verdict["needs"] == MIN_SAMPLES


def test_an_unknown_segment_still_returns_a_verdict():
    """Never an exception and never a guess dressed as a measurement."""
    verdict = LatencyBaselines().judge("classified", 40)
    assert verdict["band"] == "learning"
    assert verdict["samples"] == 0
    assert verdict["source"] == "prior", "falls back to the seeded expectation"
    assert verdict["expected_ms"]


def test_an_unknown_model_borrows_the_prior_for_its_cache_state():
    """A brand-new model has no history, but the cache state still says roughly
    what to expect — a cold write is slow whoever is serving it."""
    verdict = LatencyBaselines().judge("served:brand-new-model:cold_write", 9000)
    assert verdict["source"] == "prior"
    assert verdict["expected_ms"] == 9000


def test_a_segment_with_no_prior_says_so_rather_than_inventing_one():
    verdict = LatencyBaselines().judge("some_unknown_stage", 40)
    assert verdict["source"] == "none"
    assert verdict["expected_ms"] is None


# -- the bands -------------------------------------------------------------
def test_bands_follow_the_sigma_thresholds():
    b = LatencyBaselines()
    # Eight points, each 100ms from a mean of 1000: sigma = sqrt(80000/7) ~ 107.
    # Deliberately well clear of MIN_MATERIAL_DEVIATION_MS so this test measures
    # the sigma thresholds and nothing else.
    _train(b, "classified", [900] * 4 + [1100] * 4)
    seg = b._segments["classified"]
    assert seg.mean == 1000
    assert round(seg.stddev, 3) == round((80000 / 7) ** 0.5, 3)

    sigma = seg.stddev
    assert b.judge("classified", 1000)["band"] == "normal"
    assert b.judge("classified", int(1000 + 0.5 * sigma))["band"] == "normal"
    assert b.judge("classified", int(1000 + 1.5 * sigma))["band"] == "warn"
    assert b.judge("classified", int(1000 + 3 * sigma))["band"] == "critical"
    assert b.judge("classified", int(1000 - 2 * sigma))["band"] == "fast"


def test_being_faster_than_baseline_is_never_flagged():
    """Only the slow tail is a problem. A fast request is good news and must
    not be coloured as an anomaly just because it deviates."""
    b = LatencyBaselines()
    _train(b, "classified", [1000] * 4 + [2000] * 4)
    verdict = b.judge("classified", 1)
    assert verdict["band"] == "fast"
    assert verdict["z"] < -CRITICAL_SIGMA


def test_an_identical_history_treats_any_material_deviation_as_signal():
    """Zero variance would divide by zero. It also genuinely means any change is
    meaningful — but still only if the change is big enough to care about."""
    b = LatencyBaselines()
    _train(b, "budget", [100] * MIN_SAMPLES)

    assert b.judge("budget", 100)["band"] == "normal"
    assert b.judge("budget", 4000)["band"] == "critical"
    assert b.judge("budget", 1)["band"] == "fast"
    assert b.judge("budget", 110)["band"] == "normal", "10ms is not worth a colour"


def test_microsecond_jitter_is_never_coloured():
    """The failure this floor exists for. Gateway-local stages run in tens of
    microseconds with a sigma near 0.01ms, so ordinary jitter scores 3-4 sigma
    and lights the pipeline red on stages nobody is paged about. A dashboard
    that is permanently red is the crying-wolf failure, not a working alarm."""
    b = LatencyBaselines()
    _train(b, "routed", [0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06])

    verdict = b.judge("routed", 0.20)  # ~4 sigma out, and utterly irrelevant
    assert verdict["z"] > CRITICAL_SIGMA
    assert verdict["band"] == "normal"
    assert verdict["material"] is False
    # And it says why, so the number and the colour do not look like they disagree.
    assert "too small to matter" in verdict["note"]


def test_a_material_slowdown_on_a_fast_stage_still_fires():
    """The floor must not make a stage unmonitorable — a route that suddenly
    takes half a second is a real problem however fast it usually is."""
    b = LatencyBaselines()
    _train(b, "routed", [0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06])

    verdict = b.judge("routed", 500)
    assert verdict["band"] == "critical"
    assert verdict["material"] is True


# -- segmentation ----------------------------------------------------------
def test_cache_states_are_not_pooled():
    """A cold write is legitimately slower than a warm read. Pooling them paints
    every cold request red, and a dashboard that cries wolf gets ignored."""
    cold = segment_key("served", model="gpt-5", cache_state="cold_write")
    warm = segment_key("served", model="gpt-5", cache_state="warm_read")
    assert cold != warm

    b = LatencyBaselines()
    _train(b, cold, [9000] * MIN_SAMPLES)
    _train(b, warm, [3000] * MIN_SAMPLES)

    # 9,000ms is unremarkable for a cold write and alarming for a warm read.
    assert b.judge(cold, 9000)["band"] == "normal"
    assert b.judge(warm, 9000)["band"] == "critical"


def test_models_are_not_pooled():
    """A heavy model is legitimately slower than a light one."""
    assert segment_key("served", model="gpt-5", cache_state="warm_read") != segment_key(
        "served", model="gpt-5-nano", cache_state="warm_read"
    )


def test_gateway_stages_key_on_the_stage_alone():
    """Local work does not depend on model or cache state, so it should reach a
    confident baseline far sooner than the upstream call does."""
    assert segment_key("classified", model="gpt-5", cache_state="warm_read") == "classified"


def test_a_dirty_cache_state_string_still_segments():
    """`cache_state` arrives with trailing prose in some paths."""
    assert segment_key(
        "served", model="gpt-5", cache_state="uncached (disabled); provider caches"
    ) == segment_key("served", model="gpt-5", cache_state="uncached")


# -- ordering --------------------------------------------------------------
def test_a_sample_is_judged_before_it_moves_the_baseline():
    """Scoring a value against a baseline it has already shifted is how an
    anomaly hides itself — the more extreme the outlier, the more it drags the
    mean towards itself and the less anomalous it looks."""
    b = LatencyBaselines()
    _train(b, "classified", [100] * MIN_SAMPLES)

    verdict = b.observe_and_judge("classified", 5000)
    assert verdict["band"] == "critical"
    assert verdict["samples"] == MIN_SAMPLES, "judged against the history before itself"
    assert b._segments["classified"].n == MIN_SAMPLES + 1, "and still recorded"


# -- reporting -------------------------------------------------------------
def test_percentile_rank_is_reported_alongside_z():
    """Latency is right-skewed, so sigma alone overstates its own precision.
    'Slower than 87% of recent runs' is what people actually mean."""
    b = LatencyBaselines()
    _train(b, "classified", list(range(100, 100 + MIN_SAMPLES * 10, 10)))

    verdict = b.judge("classified", 105)
    assert 0.0 <= verdict["rank"] <= 1.0
    assert verdict["rank"] > 0.8, "105ms beats most of a 100..170 spread"


def test_percentiles_are_exposed_so_the_skew_is_visible():
    b = LatencyBaselines()
    _train(b, "classified", [100] * 20 + [5000])
    snap = b.snapshot()["segments"][0]

    assert snap["p50_ms"] == 100
    assert snap["p95_ms"] >= 100
    assert snap["mean_ms"] > snap["p50_ms"], "the mean is dragged by the tail"


def test_a_new_record_is_flagged():
    b = LatencyBaselines()
    _train(b, "classified", [100] * MIN_SAMPLES)
    assert b.judge("classified", 20)["record"] is True
    assert b.judge("classified", 150)["record"] is False


def test_snapshot_publishes_the_thresholds_it_judged_against():
    """A colour you cannot check is a colour you have to trust blindly."""
    b = LatencyBaselines()
    _train(b, "classified", [90, 90, 90, 90, 110, 110, 110, 110])
    snap = b.snapshot()

    assert snap["min_samples"] == MIN_SAMPLES
    assert snap["critical_sigma"] == CRITICAL_SIGMA
    seg = snap["segments"][0]
    assert seg["warn_above_ms"] == round(100 + seg["stddev_ms"], 2)
    assert seg["confident"] is True


def test_the_published_numbers_never_contradict_each_other():
    """A segment that prints "sigma 0.00" must not also print thresholds derived
    from a sigma it did not print. Sub-millisecond stages made this reachable:
    real spread, rounded away in the report, thresholds still computed on it."""
    b = LatencyBaselines()
    _train(b, "routed", [0.001, 0.002, 0.001, 0.002] * 3)
    seg = b.snapshot()["segments"][0]

    assert seg["stddev_ms"] == 0.0
    assert seg["degenerate"] is True
    assert seg["warn_above_ms"] is None
    assert seg["critical_above_ms"] is None


def test_a_zero_variance_segment_publishes_no_numeric_threshold():
    """"warn >= 40, critical >= 40" reads as "anything at the mean is critical",
    which is not what `judge` does. Report the rule, not a misleading number."""
    b = LatencyBaselines()
    _train(b, "budget", [40] * MIN_SAMPLES)
    seg = b.snapshot()["segments"][0]

    assert seg["degenerate"] is True
    assert seg["warn_above_ms"] is None
    assert seg["critical_above_ms"] is None
    assert seg["confident"] is True, "confident about the mean, just not about spread"


def test_snapshot_is_serialisable():
    import json

    b = LatencyBaselines()
    _train(b, "classified", [100] * 10)
    json.dumps(b.snapshot())
