"""Regression checks for the current public NPU benchmark and comparison CSV."""

import csv
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("grid", [None, (1, )])
def test_benchmark_wrapper_does_not_measure_a_cached_tuner(backend, make_tuner, monkeypatch, grid):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    monkeypatch.setenv("TRITON_NPU_BENCH_SINGLE_CONFIG", "0")
    tuner = make_tuner()
    tuner.is_simt_mode = False
    monkeypatch.setattr(tuner, "generate_key_and_configs", lambda *a, **k: (128, ))
    monkeypatch.setattr(tuner, "prune_configs", lambda kwargs: tuner.configs)
    result = object()
    monkeypatch.setattr(tuner.fn, "run", lambda *a, **k: result)
    assert tuner.run() is result

    def duplicate(*args, **kwargs):
        pytest.fail("benchmark_npu repeated measurements after an autotune cache hit")

    monkeypatch.setattr(backend.testing, "do_bench_npu", duplicate)
    if grid is None:
        assert backend.testing.benchmark_npu(lambda: tuner.run()) is result
    else:
        assert backend.testing.benchmark_npu(tuner, grid) is result


@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("parts", [1, 2])
def test_current_collector_accepts_split_summary_csv(backend, monkeypatch, tmp_path, cold, parts):
    rows = []
    for index in range(2):
        for launch in range(6):
            if cold:
                rows.append(("flush", "ReduceSum", len(rows), 999.))
            rows.append(("kernel", "Vector", len(rows), 999. if launch < 2 else 10 * (index + 1) + launch - 2))
    frame = pd.DataFrame(rows, columns=["Name", "Type", "Start Time(us)", "Duration(us)"])
    full, summary = tmp_path / "full", tmp_path / "summary"
    full.mkdir()
    summary.mkdir()
    frame.to_csv(full / "kernel_details.csv", index=False)
    frame = frame.rename(columns={
        "Name": "Op Name", "Type": "OP Type", "Start Time(us)": "Task Start Time(us)", "Duration(us)":
        "Task Duration(us)"
    })
    for part in range(parts):
        frame.iloc[part::parts].to_csv(summary / f"op_summary_{part}.csv", index=False)
    calls, original_read = [], pd.read_csv

    def read_once(path, *args, **kwargs):
        calls.append(Path(path))
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read_once)
    arguments = ([None] * 2, 2, 4, "kernel", cold, (.25, .75))
    expected_means, actual_means = [], []
    expected = backend.testing._collect_prof_result(full, *arguments, expected_means)
    actual = backend.testing._collect_prof_result(summary, *arguments, actual_means)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(actual_means, expected_means)
    np.testing.assert_allclose(actual, [.0115, .0215])
    assert len(calls) == len(set(calls)) == parts + 1


@pytest.mark.parametrize("mode", [None, "full", "summary"])
@pytest.mark.parametrize("cold", [False, True])
def test_current_profiler_export_mode(backend, monkeypatch, tmp_path, mode, cold):
    import torch
    import torch_npu

    if mode is None:
        monkeypatch.delenv("TRITON_NPU_BENCH_PROFILER_MODE", raising=False)
    else:
        monkeypatch.setenv("TRITON_NPU_BENCH_PROFILER_MODE", mode)
    calls, options = [], {}
    profiler = SimpleNamespace(prof_if=SimpleNamespace(prof_path=str(tmp_path / "trace")),
                               step=lambda: calls.append("step"))

    def handler(path, **kwargs):
        options["handler"] = (path, kwargs)
        return lambda prof: calls.append("handler")

    @contextmanager
    def profile(**kwargs):
        options.update(kwargs)
        yield profiler
        kwargs["on_trace_ready"](profiler)

    monkeypatch.setattr(torch_npu.profiler, "tensorboard_trace_handler", handler)
    monkeypatch.setattr(torch_npu.profiler, "profile", profile)
    monkeypatch.setattr(torch.npu, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(backend.policy, "export_npu_summary", lambda path: calls.append(("export", path)))
    buffer = SimpleNamespace(sum=lambda: calls.append("flush")) if cold else None
    backend.testing._profile_npu_batch([lambda: calls.append("kernel")], 2, 3, buffer, str(tmp_path), "config")
    assert calls.count("kernel") == 5 and calls.count("sync") == (10 if cold else 5)
    assert calls.count("flush") == (5 if cold else 0)
    assert options["experimental_config"] == "config"
    if mode == "summary":
        assert options["handler"][1] == {"analyse_flag": False}
        assert options["schedule"](0) == torch_npu.profiler.ProfilerAction.RECORD_AND_SAVE
        assert options["schedule"](1) == torch_npu.profiler.ProfilerAction.NONE
        assert calls[-3:] == ["step", "handler", ("export", profiler.prof_if.prof_path)]

        def failed_export(path):
            raise backend.policy.ProfilerAcquisitionError("summary export failed")

        monkeypatch.setattr(backend.policy, "export_npu_summary", failed_export)
        with pytest.raises(backend.policy.ProfilerAcquisitionError, match="summary export failed"):
            backend.testing._profile_npu_batch([], 0, 1, None, str(tmp_path), "config")
    else:
        assert options["handler"][1].get("analyse_flag", True)
        assert options.get("schedule") is None
        assert calls[-1] == "handler" and "step" not in calls
        assert not any(isinstance(call, tuple) and call[0] == "export" for call in calls)


def test_current_profiler_rejects_unknown_export_mode(backend, monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_NPU_BENCH_PROFILER_MODE", "invalid")
    with pytest.raises(ValueError, match="TRITON_NPU_BENCH_PROFILER_MODE"):
        backend.testing._profile_npu_batch([], 0, 1, None, str(tmp_path), None)
