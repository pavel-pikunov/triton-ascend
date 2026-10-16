"""Opt-in quality sensitivity and false alarms: expected failures remain ordinary red tests.

Run this file explicitly. Each metric gets 1000 cases with a present defect and
1000 independent stationary Gaussian controls, all with 50 measured launches.
Only the target metric decides each assertion; another check cannot cover a miss.
"""

import numpy as np
import pytest
from scipy.stats import theilslopes

from triton.backends.ascend import _benchmark_quality as quality

METRICS = tuple(quality.THRESHOLDS)


def sample_case(metric, seed, defect, n=50):
    rng = np.random.default_rng(100000 + 1000 * METRICS.index(metric) + seed)
    values = 40 + rng.normal(0, .02, n)
    if defect:
        if metric == "mean_trimmed_relative_deviation":
            positions = rng.choice(n, max(1, n // 50), replace=False)
            values[positions] += 100
            assert np.count_nonzero(values > 100) == len(positions), "Missing injected spikes"
        elif metric == "central_gap_ratio":
            values[:n // 10] += 8
            rng.shuffle(values)
            assert np.count_nonzero(values > 44) == n // 10, "Missing second mode"
        elif metric == "dip_statistic":
            values[:n // 2] += 2
            rng.shuffle(values)
            assert np.count_nonzero(values > 41) == n // 2, "Missing balanced modes"
        elif metric == "normalized_drift":
            values += np.linspace(0, 2, n)
            assert np.median(values[-n // 5:]) - np.median(values[:n // 5]) > 1
        else:
            values[n // 2:] += 1
            assert np.min(values[n // 2:]) > np.max(values[:n // 2]), "Missing change point"
    # Back-to-back model launches, without host overhead. Durations and timestamps are in us.
    times = np.r_[0., np.cumsum(values[:-1])]
    return times, values


def check_target(metric, seed, defect, record_property):
    times, values = sample_case(metric, seed, defect)
    metrics, _ = quality.evaluate_quality(times, values, check_all=True)
    value, threshold = metrics[metric], quality.THRESHOLDS[metric]
    triggered = not np.isfinite(value) or value > threshold
    for name, item in dict(metric=metric, seed=seed, active=len(values), defect_present=defect, triggered=triggered,
                           value=value, threshold=threshold).items():
        record_property(name, item)
    assert triggered == defect, (f"{'Missed present defect' if defect else 'False alarm on clean Gaussian'}: "
                                 f"{metric}, seed={seed}, value={value:.9g}, threshold={threshold:g}")


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("seed", range(1000))
def test_present_defect_triggers_target_metric(metric, seed, record_property):
    check_target(metric, seed, True, record_property)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("seed", range(1000))
def test_clean_gaussian_does_not_trigger_target_metric(metric, seed, record_property):
    check_target(metric, seed, False, record_property)


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("n", [50, 100, 250])
def test_metric_matches_independent_reference(metric, n):
    times, values = sample_case(metric, 0, True, n)
    metrics, _ = quality.evaluate_quality(times, values, check_all=True)
    ordered = np.sort(values)
    i, j = np.triu_indices(n, 1)
    distances = np.sort(np.abs(values[j] - values[i]))
    scale = 2.2219 * distances[(n // 2 + 1) * (n // 2) // 2 - 1]
    if metric == "mean_trimmed_relative_deviation":
        start, count = (n - n // 2) // 2, n // 2
        expected = abs(float(np.mean(values)) / float(np.mean(ordered[start:start + count])) - 1)
    elif metric == "central_gap_ratio":
        lower, upper = int(np.ceil(.1 * n)), int(np.floor(.9 * n))
        expected = max(ordered[k] - ordered[k - 1] for k in range(lower, upper + 1)) / scale
    elif metric == "normalized_drift":
        expected = abs(theilslopes(values, times).slope) * (times[-1] - times[0]) / scale
    elif metric == "normalized_change_point":
        signed_ranks = np.sign(values[:, None] - values[None, :]).sum(axis=1)
        expected = np.max(np.abs(np.cumsum(signed_ranks))) / (n * n // 4)
    else:
        # Two equally populated narrow continuous modes approach dip=1/4 as separation grows.
        assert .2 < metrics[metric] <= .26
        return
    assert metrics[metric] == pytest.approx(expected, rel=1e-12, abs=1e-12)
