"""Regression checks for the current public NPU benchmark and comparison CSV."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def configure(monkeypatch, **values):
    settings = dict(WARMUP=0, ACTIVE=30, BATCH_RUNS=2, STABILITY_RUNS=3, STABILITY_TOLERANCE=.01, LOWER_QUANTILE=.25,
                    UPPER_QUANTILE=.75, MEASURE_BUDGET_MS=0, MAX_RETRIES=3, CACHE_MODE="hot")
    settings.update(values)
    for name, value in settings.items():
        monkeypatch.setenv("TRITON_NPU_BENCH_" + name, str(value))


def profiles(monkeypatch, testing, batches):
    counts = []

    def profile(funcs, warmup, active, buffer, directory, config):
        values = batches[len(counts)]
        counts.append((warmup, active))
        rows = [("kernel", "Vector", duration) for value in values for duration in [999.] * warmup + [value] * active]
        assert len(rows) == len(funcs) * (warmup + active)
        Path(directory).mkdir(parents=True)
        pd.DataFrame(rows, columns=["Name", "Type",
                                    "Duration(us)"]).to_csv(Path(directory) / "kernel_details.csv", index=False)

    monkeypatch.setattr(testing, "_profile_npu_batch", profile)
    return counts


@pytest.mark.parametrize("requested, expected", [(2, 2), (30, 5)])
def test_cv_keeps_counts_and_history_with_budget(backend, monkeypatch, requested, expected):
    configure(monkeypatch, ACTIVE=requested, MEASURE_BUDGET_MS=250)
    counts = profiles(monkeypatch, backend.testing, [[10000., 10001.], [10002., 12000.]])
    score = backend.testing.do_bench_npu(lambda: None, warmup=1, active=5, _active_limit=5, target_kernel_name="kernel")
    assert np.isclose(score, 10.)
    assert counts == [(0, expected), (0, expected)]


def test_vector_budget_changes_active_once_and_restarts_history(backend, monkeypatch):
    configure(monkeypatch, MEASURE_BUDGET_MS=.1)
    counts = profiles(monkeypatch, backend.testing, [[1.9, 2.2], [2., 2.001], [2.002, 2.003]])
    score = backend.testing.do_bench_npu(lambda: None, target_kernel_name="kernel")
    assert np.isclose(score, .002)
    assert counts == [(0, 30), (0, 53), (0, 53)]


def test_exhausted_batches_keep_an_unconfirmed_result(backend, monkeypatch, capsys):
    configure(monkeypatch, MAX_RETRIES=1, MEASURE_BUDGET_MS=.1)
    counts = profiles(monkeypatch, backend.testing, [[1.9, 2.2]])
    score = backend.testing.do_bench_npu(lambda: None, target_kernel_name="kernel")
    assert np.isclose(score, .0019) and counts == [(0, 30)]
    assert "minimum not confirmed for configurations [1]" in capsys.readouterr().out


def test_comparison_csv_counts_all_hundred_decisions(backend, tmp_path):
    from triton import Config

    configs = [Config({"BLOCK": 32}), Config({"BLOCK": 64})]
    comparison = backend.report.AutotuneComparison(100, tmp_path)
    for index in range(100):
        selection = comparison.prepare(configs, (128, ), True)
        scores = dict(zip(configs, (.002, .003) if index % 2 == 0 else (.003, .002)))
        comparison.record(selection, min(scores, key=scores.get), scores)
        comparison.record_timing(.1, {"compilation": .01, "measurements": .08})
    with comparison.write_csv("kernel").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["best_count"]) for row in rows] == [50, 50]
    assert all(row["total_runs"] == "100" and float(row["best_time_us"]) == 2 for row in rows)
