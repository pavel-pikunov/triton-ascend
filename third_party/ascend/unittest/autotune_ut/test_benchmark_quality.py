"""CPU references and resource checks for the full-sample quality metrics."""

import time
import tracemalloc
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import theilslopes

from triton.backends.ascend import _benchmark_quality as quality


def direct_qn(values):
    i, j = np.triu_indices(len(values), 1)
    rank = (len(values) // 2 + 1) * (len(values) // 2) // 2 - 1
    return 2.2219 * float(np.partition(np.abs(values[j] - values[i]), rank)[rank])


def direct_slope(times, values):
    i, j = np.triu_indices(len(values), 1)
    return float(np.median((values[j] - values[i]) / (times[j] - times[i])))


def samples(n, kind):
    rng = np.random.default_rng(42)
    times = np.cumsum(rng.uniform(.5, 2, n)) + 1e9
    values = rng.normal(100, 2, n)
    if kind == "ties":
        values = rng.integers(95, 105, n).astype(float)
    elif kind == "constant":
        values[:] = 100
    elif kind == "outliers":
        values[::17] *= 10
    elif kind == "drift":
        values += np.linspace(0, 30, n)
    elif kind == "collinear":
        times = np.arange(n, dtype=float)
        values = times * 1.235556362299253 + 100
    elif kind.startswith("threshold"):
        values[:] = 100
        limit = quality.THRESHOLDS["mean_trimmed_relative_deviation"]
        values[-2:] += (limit + (1e-7 if kind.endswith("above") else -1e-7)) * 100 * n / 2
    return times, values


@pytest.mark.parametrize("n,kind", [
    (33, "random"),
    (34, "ties"),
    (255, "constant"),
    (256, "outliers"),
    (257, "drift"),
    (258, "random"),
    (259, "ties"),
    (260, "collinear"),
    (513, "random"),
    (513, "constant"),
])
def test_pair_statistics_match_direct_and_scipy(n, kind):
    times, values = samples(n, kind)
    assert quality._qn(values) == direct_qn(values)
    assert quality._theil_sen(times, values) == direct_slope(times, values)
    assert quality._theil_sen(times, values) == theilslopes(values, times).slope
    # Certify counts and adjacent crossings used to contract the full pair set.
    i, j = np.triu_indices(n, 1)
    slopes = np.sort((values[j] - values[i]) / (times[j] - times[i]))
    for pivot in np.quantile(slopes, [0, .25, .5, .75, 1]):
        below, through, previous, following = quality._slope_counts(times, values, pivot)
        assert below == np.sum(slopes < pivot)
        assert through == np.sum(slopes <= pivot)
        assert previous == np.max(slopes[slopes < pivot], initial=-np.inf)
        assert following == np.min(slopes[slopes > pivot], initial=np.inf)


@pytest.mark.parametrize("kind",
                         ["random", "ties", "constant", "outliers", "drift", "threshold_above", "threshold_below"])
def test_quality_metrics_and_decisions_match_direct(monkeypatch, kind):
    times, values = samples(300, kind)
    actual, failures = quality.evaluate_quality(times[::-1], values[::-1], check_all=True)
    monkeypatch.setattr(quality, "_qn", direct_qn)
    monkeypatch.setattr(quality, "_theil_sen", direct_slope)

    def original_ranks(values, method):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts - 1) / 2)[inverse]

    monkeypatch.setattr(quality, "rankdata", original_ranks)
    expected, expected_failures = quality.evaluate_quality(times, values, check_all=True)
    assert actual == expected
    assert failures == expected_failures
    if kind.startswith("threshold"):
        assert (actual["mean_trimmed_relative_deviation"]
                > quality.THRESHOLDS["mean_trimmed_relative_deviation"]) == kind.endswith("above")


@pytest.mark.parametrize("inclusive", [False, True])
def test_rounded_ties_do_not_hide_crossings(inclusive):
    times = np.array([
        0.23909220398153466, 0.5490686931729879, 0.9059729500487292, 1.1285749538055665, 2.4240720539245073,
        3.7276966487007464
    ])
    values = np.array([
        0.3457955495015555, 0.7941100015311158, 1.310295395996436, 1.632241410661087, 3.5058998744389296,
        5.391313014589693
    ])
    pivot = 1.4462853398944855
    i, j = np.triu_indices(len(values), 1)
    slopes = np.sort((values[j] - values[i]) / (times[j] - times[i]))
    below, through, previous, following = quality._slope_counts(times, values, pivot)
    assert (through if inclusive else below) == np.sum(slopes <= pivot if inclusive else slopes < pivot)
    assert previous == np.max(slopes[slopes < pivot], initial=-np.inf)
    assert following == np.min(slopes[slopes > pivot], initial=np.inf)


@pytest.mark.parametrize("kind", ["random", "collinear"])
def test_full_quality_10000_samples_linear_memory(monkeypatch, kind):
    times, values = samples(10000, kind)
    # A regression to the former pair-index path fails before allocating it.
    monkeypatch.setattr(np, "triu_indices", lambda *a, **k: pytest.fail("quadratic pair indices"))
    state = np.random.get_state()
    tracemalloc.start()
    start = time.perf_counter()
    try:
        metrics, failures = quality.evaluate_quality(times, values, check_all=True)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    elapsed = time.perf_counter() - start
    assert peak < 256 * 1024**2
    assert all(np.isfinite(list(metrics.values())))
    assert bool(failures) == (kind == "collinear")
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    print(f"10000 samples: {elapsed:.3f}s, peak auxiliary memory {peak / 1024**2:.3f} MiB")


@pytest.mark.parametrize("kind", ["random", "ties", "constant", "outliers", "drift", "threshold_above"])
def test_short_circuit_preserves_quality_decision(kind):
    times, values = samples(300, kind)
    partial, failures = quality.evaluate_quality(times, values)
    complete, all_failures = quality.evaluate_quality(times, values, check_all=True)
    assert bool(failures) == bool(all_failures)
    assert all(complete[name] == value for name, value in partial.items())
    assert set(complete) == set(quality.THRESHOLDS)


def test_debug_switch_is_not_a_measurement_policy(backend, monkeypatch):
    options = backend.policy.resolve_options()
    assert options.quality_check_all is False
    monkeypatch.setenv("TRITON_NPU_BENCH_QUALITY_CHECK_ALL", "1")
    debug = backend.policy.resolve_options()
    assert debug.quality_check_all is True
    assert debug.cache_key() == options.cache_key()


def test_short_circuit_does_not_start_remaining_checks(monkeypatch):
    times, values = samples(300, "outliers")
    monkeypatch.setattr(quality, "_theil_sen", lambda *args: pytest.fail("check after failure"))
    monkeypatch.setattr(quality, "_dip", lambda *args: pytest.fail("check after failure"))
    metrics, failures = quality.evaluate_quality(times, values)
    assert failures and list(metrics) == ["mean_trimmed_relative_deviation"]


def test_gap_check_rejects_two_separated_modes():
    metrics, failures = quality.evaluate_quality(np.arange(250), np.tile([40., 60.], 125), check_all=True)
    assert metrics["dip_statistic"] == 0.
    assert metrics["central_gap_ratio"] > quality.THRESHOLDS["central_gap_ratio"]
    assert any(reason.startswith("central_gap_ratio=") for reason in failures)


def test_profile_aggregates_preserve_missing_values_ids_and_text(backend):
    import pandas as pd

    frame = pd.DataFrame({
        "Metric": [1., np.nan, 3., np.inf, -np.inf],
        "Task ID": [7] * 5,
        "Start Time(us)": [1., 2., np.nan, 4., 5.],
        "Text": [" same "] * 5,
        "Numeric text": ["1", " N/A ", "3", "null", "inf"],
        "Changing text": ["1", "x", "None", "", ""],
        "Missing": [np.nan] * 5,
    })
    assert backend.policy.aggregate_profile_rows(frame) == {
        "Metric": (2., 2, True),
        "Task ID": ("7", 5, False),
        "Start Time(us)": ("varies", 4, False),
        "Text": ("same", 5, False),
        "Numeric text": (2., 2, True),
        "Changing text": ("varies", 2, False),
        "Missing": ("unavailable", 0, False),
    }


@pytest.mark.parametrize("check_all", [False, True])
def test_debug_mode_reaches_quality_on_every_attempt(backend, tmp_path, check_all):
    policy = backend.policy
    times, values = samples(300, "outliers")
    options = replace(policy.resolve_options(), active=300, warmup=0, measure_budget_ms=None, filter_slow_configs=False,
                      max_retries=1, quality_check_all=check_all, log_level="off")
    record = policy.NpuMeasurementReport({}, 300, 0, 300, "hot",
                                         float(np.mean(values)) / 1000, .1, samples=(times, values))
    calls, reports = [], []

    def profile(*args, **kwargs):
        calls.append(1)
        return [record]

    assert policy.bench_npu(profile, [lambda: None], 5, 30, True, tmp_path, False, "kernel", options=options,
                            report_sink=reports.extend) == .1
    assert calls == [1, 1] and reports[0].quality_failures
    assert len(reports[0].quality_metrics) == (5 if check_all else 1)
