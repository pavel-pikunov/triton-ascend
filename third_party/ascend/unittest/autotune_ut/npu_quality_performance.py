"""Opt-in CPU processing budget: real CSV collector and all five quality checks."""

import statistics
import time

import numpy as np
import pandas as pd
import pytest

from npu_stability_stress import SCENARIOS, sample_values

EXTRA = ("gaussian", "rounded", "uniform", "constant", "linear", "near_linear", "bimodal", "collinear_wide")
CASES = [(name, variant) for name in SCENARIOS for variant in range(1, 6)] + [(name, 0) for name in EXTRA]


def values_for_case(n, scenario, variant, index):
    if variant:
        values = sample_values(scenario, variant, index % 3, n)
        # Invalid acquisitions are rejected before quality; time their valid retry here.
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            values = sample_values(scenario, variant, index % 3, n, attempt=2)
        return values
    rng = np.random.default_rng(42 + index)
    values = 40 + rng.normal(0, .02, n)
    if scenario == "rounded":
        values = np.round(values, 3)
    elif scenario == "uniform":
        values = 40 + rng.permutation(np.linspace(-.08, .08, n))
    elif scenario == "constant":
        values[:] = 40
    elif scenario in ("linear", "near_linear"):
        values = 40 + np.linspace(0, 2, n)
        if scenario == "near_linear":
            values += rng.normal(0, 1e-12, n)
    elif scenario == "bimodal":
        values[n // 2:] += 2
    elif scenario == "collinear_wide":
        values = np.arange(n, dtype=float) * 1.235556362299253 + 100
    return values


@pytest.mark.parametrize("scenario,variant", CASES)
@pytest.mark.parametrize("n", [250, 5000])
def test_full_csv_and_quality_budget(backend, tmp_path, record_property, n, scenario, variant):
    warmup, candidates = 50, 24
    budget = 2 if n == 5000 else 1
    durations = np.concatenate(
        [np.r_[np.full(warmup, 999.), values_for_case(n, scenario, variant, index)] for index in range(candidates)])
    frame = pd.DataFrame({
        "Name": "same", "Type": "Kernel", "Start Time(us)": 1e9 + np.arange(len(durations)) * 1000, "Duration(us)":
        durations
    })
    for index in range(28):
        frame[f"Utilization {index}(%)"] = (index + 1) / 3
    frame.to_csv(tmp_path / "kernel_details.csv", index=False)
    elapsed = []
    for _ in range(3):
        started = time.perf_counter()
        records = backend.policy._collect_prof_result(str(tmp_path), [lambda: None] * candidates, warmup, n, "same",
                                                      False, _return_samples=True, _capture_report=True,
                                                      _quantiles=(.25, .75))
        outputs = [backend.quality.evaluate_quality(*record.samples, check_all=True) for record in records]
        elapsed.append(time.perf_counter() - started)
        assert all(set(metrics) == set(backend.quality.THRESHOLDS) for metrics, _ in outputs)
    for name, value in {
            "scenario": scenario, "variant": variant, "active": n, "candidates": candidates, "all_checks": True,
            "budget_s": budget, "median_s": statistics.median(elapsed), "min_s": min(elapsed), "max_s": max(elapsed)
    }.items():
        record_property(name, value)
    assert statistics.median(elapsed) < budget, f"CPU CSV + all quality checks: {elapsed} seconds (budget={budget}s)"
