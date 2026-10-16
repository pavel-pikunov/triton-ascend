"""Summary/full export controls and the shared pandas collector."""

import json
from contextlib import contextmanager
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


@pytest.mark.parametrize("active,budget,warmup,expected", [
    (1, None, None, (1, 1)),
    (50, None, None, (5, 50)),
    (51, None, None, (6, 51)),
    (250, None, None, (25, 250)),
    (50, 10., None, (100, 1000)),
    (50, 10., 0, (0, 1000)),
    (50, 10., 7, (7, 1000)),
])
def test_automatic_warmup_uses_the_final_budget_count(backend, monkeypatch, tmp_path, active, budget, warmup, expected):
    policy = backend.policy

    @contextmanager
    def timer(*args):
        yield lambda fn, count: .01

    calls = []

    def profile(funcs, warmup, active, *args, **kwargs):
        calls.append((warmup, active))
        return [policy.NpuMeasurementReport({}, active, warmup, active, "hot", .01, .01)]

    monkeypatch.setattr(policy, "event_timer", timer)
    options = replace(policy.resolve_options(), active=active, warmup=warmup, measure_budget_ms=budget,
                      filter_slow_configs=False, quality_check=False, max_retries=0, log_level="off")
    assert policy.bench_npu(profile, [lambda: None], 5, 30, False, tmp_path, False, "k", options=options) == .01
    assert calls == [expected]


def test_automatic_warmup_environment_and_cache_key(backend, monkeypatch):
    automatic = backend.policy.resolve_options()
    assert automatic.warmup is None
    monkeypatch.setenv("TRITON_NPU_BENCH_WARMUP", "0")
    explicit = backend.policy.resolve_options()
    assert explicit.warmup == 0 and explicit.cache_key() != automatic.cache_key()


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("reports", [False, True])
@pytest.mark.parametrize("cache", ["hot", "cold"])
def test_fixed_kernel_benchmark_reuses_npu_policies(backend, jit_kernel, monkeypatch, capsys, wrapped, reports, cache):
    policy, testing = backend.policy, backend.testing
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu_legacy")
    for name, value in dict(CACHE_MODE=cache, ACTIVE=8, MEASURE_BUDGET_MS=.4, MAX_RETRIES=2,
                            REPORT_BEST_CONFIG=int(reports)).items():
        monkeypatch.setenv(policy.ENV_PREFIX + name, str(value))
    monkeypatch.setenv("TRITON_AUTOTUNE_REPORT_TIMING", str(int(reports)))
    compiled = SimpleNamespace(src=SimpleNamespace(name="fixed_kernel", constants={"BLOCK": 32}),
                               packed_metadata={"kernel_name": "compiled_target"},
                               metadata=SimpleNamespace(_asdict=lambda: dict(num_warps=4, num_stages=2)))
    result = (compiled, {"BLOCK": 32}) if wrapped else compiled
    launches, measurements = [], []
    data = object()

    def launch(*args, **kwargs):
        assert args == (data, ) and kwargs == dict(BLOCK=32, grid=(1, ), warmup=False)
        launches.append(1)
        return result

    def measure(profile, fn, warmup, active, cold, directory, keep, target, **kwargs):
        options = kwargs["options"]
        assert profile is testing._profile_npu and target == "compiled_target"
        assert not options.filter_slow_configs and options.quality_check
        assert options.active == active == 8 and options.measure_budget_ms == .4 and options.max_retries == 2
        assert warmup is None and cold == (cache == "cold")
        assert (options.lower_quantile, options.upper_quantile) == (.25, .75)
        assert kwargs["config_numbers"] == [1]
        assert (kwargs["timings"] is not None) == reports
        for _ in range(3):
            fn()
        measurements.append(1)
        if reports:
            record = policy.NpuMeasurementReport({}, 8, 1, 8, cache, .021, .02, quality_checked=True)
            kwargs["report_sink"]([record])
        else:
            assert kwargs["report_sink"] is None
        return .02

    monkeypatch.setattr(jit_kernel, "run", launch)
    monkeypatch.setattr(policy, "bench_npu", measure)
    assert testing.benchmark_npu(jit_kernel, (1, ), data, BLOCK=32) is result
    assert launches == [1] * 4 and measurements == [1]
    text = capsys.readouterr().out
    assert ("Triton autotuning timing for fixed_kernel" in text) == reports
    assert ("Triton autotuning result for fixed_kernel" in text) == reports
    if reports:
        assert "Selection score: 20 us" in text and "Quality checks: passed" in text
        assert "Selected config: 1/1" in text and "compiler_metadata" in text


@pytest.mark.parametrize("wrapped", [False, True])
def test_kernel_benchmark_delegates_existing_autotuner(backend, make_tuner, monkeypatch, capsys, wrapped):
    from triton.runtime import Heuristics

    tuner = make_tuner()
    calls, result = [], object()

    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        print("Existing autotuner report")
        return result

    monkeypatch.setattr(tuner, "run", launch)
    monkeypatch.setattr(backend.policy, "resolve_options", lambda: pytest.fail("unexpected standalone benchmark"))
    kernel = Heuristics(tuner, tuner.arg_names, {}) if wrapped else tuner
    assert backend.testing.benchmark_npu(kernel, (1, ), "input", BLOCK=32) is result
    assert calls == [(("input", ), dict(grid=(1, ), warmup=False, BLOCK=32))]
    assert capsys.readouterr().out == "Existing autotuner report\n"


@pytest.mark.parametrize("stage", ["compilation", "measurements"])
def test_fixed_kernel_benchmark_propagates_errors(backend, jit_kernel, monkeypatch, capsys, stage):

    def fail(*args, **kwargs):
        raise RuntimeError(stage)

    compiled = SimpleNamespace(src=SimpleNamespace(name="fixed_kernel"),
                               packed_metadata={"kernel_name": "compiled_target"})
    monkeypatch.setattr(jit_kernel, "run", fail if stage == "compilation" else lambda *a, **k: compiled)
    monkeypatch.setattr(backend.policy, "bench_npu",
                        fail if stage == "measurements" else lambda *a, **k: pytest.fail("bench after compile failure"))
    with pytest.raises(RuntimeError, match=stage):
        backend.testing.benchmark_npu(jit_kernel, (1, ), "input", BLOCK=32)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("reports", [False, True])
def test_callable_benchmark_uses_quality_retries(backend, monkeypatch, capsys, tmp_path, reports):
    policy, testing = backend.policy, backend.testing
    for name, value in dict(ACTIVE=8, WARMUP=2, MEASURE_BUDGET_MS=0, MAX_RETRIES=1,
                            REPORT_BEST_CONFIG=int(reports)).items():
        monkeypatch.setenv(policy.ENV_PREFIX + name, str(value))
    monkeypatch.setenv("TRITON_AUTOTUNE_REPORT_TIMING", str(int(reports)))
    monkeypatch.setattr(policy.runtime.cache, "get_home_dir", lambda: str(tmp_path))
    calls, attempts = [], []
    result = object()

    def operation():
        calls.append(1)
        return result

    def profile(funcs, warmup, active, cold, directory, keep, names, **kwargs):
        assert kwargs["_aggregate_calls"] and names == [None]
        assert warmup == 2 and active == 8
        for _ in range(warmup + active):
            funcs[0]()
        attempts.append(1)
        duration = .02 if len(attempts) == 1 else .021
        count = warmup + active
        pd.DataFrame({
            "Name": ["first", "second"] * count, "Type": "Kernel", "Start Time(us)": np.arange(2 * count) * 100 + 1,
            "Duration(us)": duration * 500
        }).to_csv(Path(directory) / "kernel_details.csv", index=False)
        return policy._collect_prof_result(directory, funcs, warmup, active, names,
                                           _return_samples=kwargs["_return_samples"],
                                           _capture_report=kwargs["_capture_report"], _quantiles=kwargs["_quantiles"],
                                           _aggregate_calls=kwargs["_aggregate_calls"])

    monkeypatch.setattr(testing, "_profile_npu", profile)
    monkeypatch.setattr(backend.quality, "evaluate_quality", lambda *a: ({}, ["unstable"]
                                                                         if len(attempts) == 1 else []))
    assert testing.benchmark_npu(operation) is result
    assert len(attempts) == 2 and len(calls) == 21
    text = capsys.readouterr().out
    assert ("Triton autotuning timing for operation" in text) == reports
    assert ("Selection score: 20 us" in text) == reports


@pytest.mark.parametrize("reports", [False, True])
def test_callable_does_not_repeat_autotuner_measurements(backend, make_tuner, monkeypatch, capsys, reports):
    from importlib import import_module
    comparison = import_module(backend.testing.__package__ + "._autotune_comparison")
    tuner = make_tuner()
    tuner._autotune_selection = None
    monkeypatch.setenv("TRITON_NPU_BENCH_REPORT_BEST_CONFIG", str(int(reports)))
    monkeypatch.setenv("TRITON_AUTOTUNE_REPORT_TIMING", str(int(reports)))
    monkeypatch.setattr(backend.policy, "bench_npu", lambda *a, **k: pytest.fail("duplicate benchmark"))
    calls, result = [], object()

    def operation():
        calls.append(1)
        comparison.record_scored_candidates(tuner, {object(): .02})
        if reports:
            print("Existing autotuner report")
        return result

    assert backend.testing.benchmark_npu(operation) is result
    assert calls == [1]
    assert capsys.readouterr().out == ("Existing autotuner report\n" if reports else "")


def test_callable_exception_restores_measurement_scope(backend):
    from importlib import import_module
    comparison = import_module(backend.testing.__package__ + "._autotune_comparison")
    previous = comparison._perf_measured.get()

    def fail():
        comparison._perf_measured.set(not previous)
        raise RuntimeError("operation failed")

    with pytest.raises(RuntimeError, match="operation failed"):
        backend.testing.benchmark_npu(fail)
    assert comparison._perf_measured.get() == previous


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("names", [None, [None]])
def test_operation_collector_sums_kernels_per_call(backend, tmp_path, width, names):
    count = 6
    durations = np.arange(count * width, dtype=float).reshape(count, width) + 1
    pd.DataFrame({
        "Name": [f"k{i}" for i in range(width)] * count, "Type": "Kernel", "Start Time(us)":
        np.arange(count * width) + 1, "Duration(us)": durations.ravel()
    }).to_csv(tmp_path / "kernel_details.csv", index=False)
    record, = backend.policy._collect_prof_result(tmp_path, [lambda: None], 2, 4, names, _return_samples=True,
                                                  _capture_report=True, _quantiles=(.25, .75), _aggregate_calls=True)
    totals = durations.sum(axis=1)[2:]
    assert record.score_ms == np.mean(np.sort(totals)[1:3]) / 1000
    assert record.mean_ms == np.mean(totals) / 1000
    np.testing.assert_array_equal(record.samples[1], totals)
    np.testing.assert_array_equal(record.samples[0], (np.arange(count * width) + 1).reshape(count, width)[2:, 0])
    assert record.sample_count == 4 and record.metrics["Duration(us)"][0] == np.mean(totals)


@pytest.mark.parametrize("names", [["a", "b", "a"], ["a", "b", "b", "a"]])
def test_operation_collector_rejects_changing_sequence(backend, tmp_path, names):
    pd.DataFrame({"Name": names, "Type": "Kernel", "Start Time(us)": np.arange(len(names)) + 1, "Duration(us)":
                  10}).to_csv(tmp_path / "kernel_details.csv", index=False)
    with pytest.raises(backend.policy.ProfilerAcquisitionError, match="Operation kernel"):
        backend.policy._collect_prof_result(tmp_path, [lambda: None], 0, 2, _aggregate_calls=True)
