"""CPU references and resource checks for the full-sample quality metrics."""

import importlib.util
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import theilslopes


spec = importlib.util.spec_from_file_location(
    "_benchmark_quality_test", Path(__file__).resolve().parents[2] / "backend" / "_benchmark_quality.py")
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


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
        values[-2:] += (0.04 + (1e-7 if kind.endswith("above") else -1e-7)) * 100 * n / 2
    return times, values


@pytest.mark.parametrize("n", [33, 34, 255, 256, 257, 258, 259, 260, 513])
@pytest.mark.parametrize("kind", ["random", "ties", "constant", "outliers", "drift", "collinear"])
def test_pair_statistics_match_direct_and_scipy(n, kind):
    times, values = samples(n, kind)
    assert quality._qn(values) == direct_qn(values)
    assert quality._theil_sen(times, values) == direct_slope(times, values)
    assert quality._theil_sen(times, values) == theilslopes(values, times).slope
    # Exercise contraction independently of the small-sample dispatch.
    pairs = n * (n - 1) // 2
    i, j = np.triu_indices(n, 1)
    slopes = np.sort((values[j] - values[i]) / (times[j] - times[i]))
    for rank in [0, pairs // 4, pairs // 2, pairs - 1]:
        assert quality._select_slope(times, values, rank, np.random.default_rng(0)) == slopes[rank]


@pytest.mark.parametrize("kind", ["random", "ties", "constant", "outliers", "drift",
                                  "threshold_above", "threshold_below"])
def test_quality_metrics_and_decisions_match_direct(monkeypatch, kind):
    times, values = samples(300, kind)
    actual, failures = quality.evaluate_quality(times[::-1], values[::-1])
    monkeypatch.setattr(quality, "_qn", direct_qn)
    monkeypatch.setattr(quality, "_theil_sen", direct_slope)

    def original_ranks(values, method):
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts - 1) / 2)[inverse]

    monkeypatch.setattr(quality, "rankdata", original_ranks)
    expected, expected_failures = quality.evaluate_quality(times, values)
    assert actual == expected
    assert failures == expected_failures
    if kind.startswith("threshold"):
        assert (actual["mean_trimmed_relative_deviation"] > 0.04) == kind.endswith("above")


@pytest.mark.parametrize("inclusive", [False, True])
def test_rounded_ties_do_not_hide_crossings(inclusive):
    times = np.array([0.23909220398153466, 0.5490686931729879, 0.9059729500487292,
                      1.1285749538055665, 2.4240720539245073, 3.7276966487007464])
    values = np.array([0.3457955495015555, 0.7941100015311158, 1.310295395996436,
                       1.632241410661087, 3.5058998744389296, 5.391313014589693])
    pivot = 1.4462853398944855
    i, j = np.triu_indices(len(values), 1)
    slopes = np.sort((values[j] - values[i]) / (times[j] - times[i]))
    order, uncertain = quality._slope_order(times, values, pivot, inclusive)
    count, _ = quality._inversion_slopes(list(range(len(values))), order, times, values)
    assert uncertain or count == np.sum(slopes <= pivot if inclusive else slopes < pivot)
    for rank in [0, len(slopes) // 2, len(slopes) - 1]:
        assert quality._select_slope_direct(times, values, rank, pivot) == slopes[rank]


def test_full_quality_10000_samples_linear_memory(monkeypatch):
    times, values = samples(10000, "random")
    # A regression to the former pair-index path fails before allocating it.
    monkeypatch.setattr(np, "triu_indices", lambda *a, **k: pytest.fail("quadratic pair indices"))
    state = np.random.get_state()
    tracemalloc.start()
    start = time.perf_counter()
    try:
        metrics, failures = quality.evaluate_quality(times, values)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    elapsed = time.perf_counter() - start
    assert peak < 256 * 1024**2
    assert all(np.isfinite(list(metrics.values())))
    assert failures == []
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]
    print(f"10000 samples: {elapsed:.3f}s, peak auxiliary memory {peak / 1024**2:.3f} MiB")
