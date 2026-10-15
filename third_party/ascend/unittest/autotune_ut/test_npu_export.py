"""Summary/full export controls and the shared pandas collector."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def exporter_layout(tmp_path, policy, monkeypatch):
    toolkit = tmp_path / "tool kit"
    program = toolkit / "tools/profiler/bin/msprof"
    program.parent.mkdir(parents=True)
    program.touch()
    script = toolkit / "tools/profiler/profiler_tool/analysis/msprof/msprof.py"
    script.parent.mkdir(parents=True)
    trace = tmp_path / "trace"
    cann = trace / "PROF_1"
    cann.mkdir(parents=True)
    monkeypatch.setattr(policy.shutil, "which", lambda name: str(program))
    return script, trace, cann


def test_summary_export_runs_only_the_summary_command(backend, monkeypatch, tmp_path):
    policy = backend.policy
    script, trace, cann = exporter_layout(tmp_path, policy, monkeypatch)
    script.write_text(
        '''import json
import pathlib
import sys
assert sys.argv[1:4] == ["export", "summary", "-dir"]
assert sys.argv[5:] == ["--format", "csv"]
directory = pathlib.Path(sys.argv[4])
(directory / "args.json").write_text(json.dumps(sys.argv[1:]))
(directory / "op_summary_0.csv").write_text(
    "Op Name,OP Type,Task Start Time(us),Task Duration(us)\\nk,Kernel,1,10\\nk,Kernel,2,10\\n")
''', encoding="utf-8")
    timings = {}
    policy.export_npu_summary(trace, timings)
    assert json.loads((cann / "args.json").read_text()) == ["export", "summary", "-dir", str(cann), "--format", "csv"]
    records = policy._collect_prof_result(trace, [lambda: None], 0, 2, "k", _return_samples=True)
    assert records[0].score_ms == .01 and timings["export"] > 0


@pytest.mark.parametrize("missing", ["cann", "msprof", "script"])
def test_missing_export_inputs_are_acquisition_errors(backend, monkeypatch, tmp_path, missing):
    policy = backend.policy
    script, trace, cann = exporter_layout(tmp_path, policy, monkeypatch)
    if missing == "cann":
        cann.rmdir()
    elif missing == "msprof":
        monkeypatch.setattr(policy.shutil, "which", lambda name: None)
    assert not script.exists()
    with pytest.raises(policy.ProfilerAcquisitionError):
        policy.export_npu_summary(trace)


def test_export_failure_keeps_the_tool_error(backend, monkeypatch, tmp_path):
    policy = backend.policy
    script, trace, _ = exporter_layout(tmp_path, policy, monkeypatch)
    script.touch()
    monkeypatch.setattr(policy.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=1, stderr="bad data", stdout=""))
    with pytest.raises(policy.ProfilerAcquisitionError, match="bad data"):
        policy.export_npu_summary(trace)


@pytest.mark.parametrize("cold", [False, True])
@pytest.mark.parametrize("parts", [1, 2])
def test_summary_and_kernel_details_produce_identical_measurements(backend, monkeypatch, tmp_path, cold, parts):
    policy = backend.policy
    rows = []
    for index in range(2):
        for launch in range(6):
            duration = 999 if launch < 2 else 10 * (index + 1) + launch - 2
            if cold:
                rows.append(("cache", "ReduceSum", len(rows) + 1, 999., "AI_CORE"))
            rows.append(("same", "Kernel", len(rows) + 1, duration, "AI_CORE"))
    frame = pd.DataFrame(rows, columns=["Name", "Type", "Start Time(us)", "Duration(us)", "Accelerator Core"])
    original, summary = tmp_path / "full", tmp_path / "summary"
    original.mkdir()
    summary.mkdir()
    frame.to_csv(original / "kernel_details.csv", index=False)
    frame = frame.rename(
        columns={
            "Name": "Op Name", "Type": "OP Type", "Start Time(us)": "Task Start Time(us)", "Duration(us)":
            "Task Duration(us)", "Accelerator Core": "Task Type"
        })
    for part in range(parts):
        frame.iloc[part::parts].to_csv(summary / f"op_summary_{part}.csv", index=False)
    calls, read = [], pd.read_csv

    def read_once(path, *args, **kwargs):
        calls.append(Path(path))
        return read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read_once)
    arguments = ([lambda: None] * 2, 2, 4, "same", cold)
    options = dict(_return_samples=True, _capture_report=True, _quantiles=(.25, .75))
    expected = policy._collect_prof_result(original, *arguments, **options)
    actual = policy._collect_prof_result(summary, *arguments, **options)
    assert len(calls) == parts + 1 and len(set(calls)) == parts + 1
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left.samples, right.samples)
        assert replace(left, samples=None) == replace(right, samples=None)
    assert [record.score_ms for record in actual] == [.0115, .0215]


def test_profiler_mode_defaults_and_cache_key(backend, monkeypatch):
    policy = backend.policy
    summary = policy.resolve_options()
    assert summary.profiler_mode == "summary"
    monkeypatch.setenv("TRITON_NPU_BENCH_PROFILER_MODE", "full")
    full = policy.resolve_options()
    assert full.profiler_mode == "full" and full.cache_key() != summary.cache_key()
    assert asdict(replace(full, profiler_mode="summary")) == asdict(summary)
    monkeypatch.setenv("TRITON_NPU_BENCH_PROFILER_MODE", "invalid")
    with pytest.raises(ValueError, match="PROFILER_MODE"):
        policy.resolve_options()


@pytest.mark.parametrize("mode", ["summary", "full"])
def test_mode_and_timing_reach_the_profiler(backend, monkeypatch, tmp_path, mode):
    policy = backend.policy
    clock, timings, calls = [0.], {}, []
    monkeypatch.setattr(policy.time, "perf_counter", lambda: clock[0])
    options = replace(policy.resolve_options(), profiler_mode=mode, active=2, warmup=0, max_retries=0,
                      measure_budget_ms=None, filter_slow_configs=False, log_level="off")
    measurement = policy.NpuMeasurementReport({}, 2, 0, 2, "hot", .01, .01, samples=(np.arange(2), np.full(2, 10.)))

    def profile(*args, **kwargs):
        assert kwargs["_profiler_mode"] == mode and kwargs["_timings"] is timings
        calls.append(1)
        timings.update(profiler=2., export=3., csv=5.)
        clock[0] += 10
        return [measurement]

    def quality(*args):
        clock[0] += 7
        return {}, []

    monkeypatch.setattr(backend.quality, "evaluate_quality", quality)
    assert policy.bench_npu(profile, [lambda: None], 5, 30, False, tmp_path, False, "k", options=options,
                            timings=timings) == .01
    assert calls == [1] and timings["quality"] == 7 and timings["measurements"] == 17


def test_nested_timing_does_not_double_count(backend, monkeypatch, capsys):
    policy = backend.policy
    monkeypatch.setattr(policy.time, "perf_counter", lambda: 100.)
    policy.print_timing_report(
        "kernel", 0.,
        dict(generation_pruning=10., compilation=20., slow_filter=2., calibration=3., measurements=50., profiler=30.,
             export=10., csv=5., quality=3.))
    text = capsys.readouterr().out
    assert "Total: 100.000000 s" in text and "  Other: 15.000000 s" in text
    assert "Other measurement overhead: 2.000000 s" in text
    assert text.count("CANN export/analysis:") == 1


def test_comparison_timing_reuses_the_breakdown_and_preserves_csv_totals(backend, capsys):
    report = backend.report
    durations = dict(measurements=50., profiler=30., export=10., csv=5., quality=3.)
    report.print_timing_report("kernel", 0., durations, total=60.)
    text = capsys.readouterr().out
    assert "Total: 60.000000 s" in text and "  Other: 10.000000 s" in text
    assert "CANN export/analysis: 10.000000 s" in text
    comparison = report.AutotuneComparison(1, None)
    comparison.record_timing(60., durations)
    assert comparison.timings[0]["other"] == 10.
    report.print_timing_report("legacy", 0., {"measurements": 50.}, total=60.)
    assert "CANN export/analysis" not in capsys.readouterr().out
