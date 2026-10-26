"""Real-device comparison of three benchmark paths through both decorators."""

import csv
import json
import math
import os

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.ascend import _autotune_report, _legacy_testing, _npu_benchmark, testing
from triton.backends.ascend.runtime import autotuner


@pytest.mark.parametrize("decorator", ["autotune", "max_autotune"])
@pytest.mark.parametrize("method,profiler_mode", [("default", None), ("npu_legacy", None), ("npu", "summary"),
                                                  ("npu", "torch_npu")])
def test_npu_vector_tuning_and_cache(monkeypatch, capsys, tmp_path, decorator, method, profiler_mode):
    for name in tuple(os.environ):
        if name.startswith(("TRITON_NPU_BENCH_", "TRITON_AUTOTUNE_")) or name == "TRITON_PRINT_AUTOTUNING":
            monkeypatch.delenv(name)
    monkeypatch.setenv("TRITON_BENCH_METHOD", method)
    if profiler_mode is not None:
        monkeypatch.setenv("TRITON_NPU_BENCH_PROFILER_MODE", profiler_mode)
    monkeypatch.setenv("TRITON_AUTOTUNE_PARALLEL_COMPILE", "0")
    monkeypatch.setenv("TRITON_NPU_BENCH_CACHE_MODE", "hot")
    monkeypatch.setenv("TRITON_NPU_BENCH_WARMUP", "0")
    monkeypatch.setenv("TRITON_NPU_BENCH_ACTIVE", "2")
    monkeypatch.setenv("TRITON_NPU_BENCH_QUALITY_CHECK", "0")
    monkeypatch.setenv("TRITON_NPU_BENCH_MAX_RETRIES", "0")
    monkeypatch.setenv("TRITON_NPU_BENCH_MEASURE_BUDGET_MS", "0")
    monkeypatch.setenv("TRITON_NPU_BENCH_FILTER_SLOW_CONFIGS", "0")
    monkeypatch.setenv("TRITON_NPU_BENCH_LOWER_QUANTILE", "0.25")
    monkeypatch.setenv("TRITON_NPU_BENCH_UPPER_QUANTILE", "0.75")
    monkeypatch.setenv("TRITON_NPU_BENCH_REPORT_BEST_CONFIG", "1")
    monkeypatch.setenv("TRITON_AUTOTUNE_REPORT_TIMING", "1")
    monkeypatch.setenv("TRITON_AUTOTUNE_RUNS", "3")
    monkeypatch.setenv("TRITON_AUTOTUNE_CSV_DIR", str(tmp_path / "comparison"))
    monkeypatch.setattr(testing.runtime.cache, "get_home_dir", lambda: str(tmp_path))
    profiles, reports, summaries = [], [], []
    original_profile = testing._profile_npu
    original_legacy_profile = _legacy_testing.do_bench_npu
    original_report = _autotune_report.print_best_config_report
    original_summary = _npu_benchmark.export_npu_summary

    def profile(*args, **kwargs):
        profiles.append(True)
        assert kwargs["_profiler_mode"] == profiler_mode
        return original_profile(*args, **kwargs)

    def summary(*args, **kwargs):
        summaries.append(True)
        return original_summary(*args, **kwargs)

    def legacy_profile(*args, **kwargs):
        profiles.append(True)
        return original_legacy_profile(*args, **kwargs)

    def report(function, configs, selected, measurement, **kwargs):
        reports.append(measurement)
        return original_report(function, configs, selected, measurement, **kwargs)

    monkeypatch.setattr(testing, "_profile_npu", profile)
    monkeypatch.setattr(_npu_benchmark, "export_npu_summary", summary)
    monkeypatch.setattr(_legacy_testing, "do_bench_npu", legacy_profile)
    monkeypatch.setattr(_autotune_report, "print_best_config_report", report)

    @triton.jit
    def add_one(x, y, N, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(x + offsets, offsets < N, other=0)
        tl.store(y + offsets, values + 1, offsets < N)

    configs = [triton.Config({"BLOCK": block}, num_stages=1) for block in (64, 128)]
    kwargs = {"kernel_type": "vector", "enable_ubuf_saving": [False]} if decorator == "max_autotune" else {}
    kernel = getattr(autotuner, decorator)(configs, ["N"], **kwargs)(add_one)
    assert len(kernel.configs) == 2
    x = torch.arange(257, dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK"]), )
    kernel[grid](x, y, x.numel())
    torch.npu.synchronize()
    torch.testing.assert_close(y, x + 1)
    assert len(profiles) == (0 if method == "default" else 3)
    assert len(summaries) == (3 if profiler_mode == "summary" else 0)
    assert len(reports) == 3
    for measurement in reports:
        assert measurement is not None
        if method == "npu":
            assert measurement.sample_count == 2
            assert measurement.cache_mode == "hot" and measurement.warmup == 0 and measurement.active == 2
            assert math.isfinite(measurement.mean_ms) and measurement.mean_ms > 0
            assert math.isfinite(measurement.score_ms) and measurement.score_ms > 0
            assert measurement.metrics["Name"][0] and measurement.metrics["Duration(us)"][0] > 0
        else:
            score = measurement.score[0] if method == "default" else measurement.score
            assert math.isfinite(score) and score > 0
            if method == "npu_legacy":
                assert measurement.cache_mode == "hot" and measurement.warmup == 5 and measurement.active == 30
    output = capsys.readouterr().out
    if method == "npu":
        assert "Incorrect schedule" not in output
    assert output.count("Selected config:") == output.count("Triton autotuning timing") == 3
    decisions = [
        json.loads(line.split(" ", 1)[1]) for line in output.splitlines() if line.startswith("AUTOTUNE_DECISION ")
    ]
    assert len(decisions) == 3 and {decision["benchmark_method"] for decision in decisions} == {method}
    assert len({decision["candidate_order_id"] for decision in decisions}) == 1
    csv_paths = list((tmp_path / "comparison").glob("*.csv"))
    assert len(csv_paths) == 1
    with csv_paths[0].open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and all(row["total_runs"] == "3" for row in rows)
    assert sum(int(row["best_count"]) for row in rows) == 3
    selected = kernel.best_config
    y.zero_()
    kernel[grid](x, y, x.numel())
    torch.npu.synchronize()
    torch.testing.assert_close(y, x + 1)
    assert kernel.best_config is selected and len(kernel.cache) == 1
    assert len(profiles) == (0 if method == "default" else 3)
    assert len(reports) == 3 and capsys.readouterr().out == ""
    assert list((tmp_path / "comparison").glob("*.csv")) == csv_paths
