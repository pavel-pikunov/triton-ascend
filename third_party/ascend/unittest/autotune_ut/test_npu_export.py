"""Summary/full export controls and the shared pandas collector."""

import json
from contextlib import contextmanager, nullcontext
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
                      measure_budget_ms=None, filter_slow_configs=False, stability_runs=1, log_level="off")
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
@pytest.mark.parametrize("export", [False, True])
def test_callable_benchmark_uses_quality_retries(backend, monkeypatch, capsys, tmp_path, reports, export):
    policy, testing = backend.policy, backend.testing
    for name, value in dict(ACTIVE=8, WARMUP=2, MEASURE_BUDGET_MS=0, MAX_RETRIES=1, STABILITY_RUNS=1,
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
    captured = []
    with policy.capture_npu_reports(lambda name, record: captured.append((name, record))) if export else nullcontext():
        assert testing.benchmark_npu(operation) is result
    assert len(captured) == int(export)
    if export:
        name, record = captured[0]
        assert name == "operation" and record.attempt == 1
        assert [name for name, _ in record.kernel_rows] == ["first", "second"]
        assert all(len(rows) == 8 for _, rows in record.kernel_rows)
        assert all(row["Duration(us)"] == 10 for _, rows in record.kernel_rows for row in rows)
    assert policy._perf_report_sink.get() is None
    assert len(attempts) == 2 and len(calls) == 21
    text = capsys.readouterr().out
    assert ("Triton autotuning timing for operation" in text) == reports
    assert ("NPU operation kernel breakdown for operation" in text) == reports
    assert "Selection score:" not in text
    if reports:
        assert "1. first" in text and "2. second" in text
        assert text.count("score=10 us; mean=10 us; min=10 us; max=10 us") == 2
        assert "selected attempt=1; operation quality: failed" in text


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
    assert len(record.kernel_timings) == width
    for index, (name, score, mean, low, high) in enumerate(record.kernel_timings):
        values = durations[2:, index]
        assert name == f"k{index}"
        assert score == np.mean(np.sort(values)[1:3])
        assert (mean, low, high) == (np.mean(values), min(values), max(values))


@pytest.mark.parametrize("names", [["a", "b", "a"], ["a", "b", "b", "a"]])
def test_operation_collector_rejects_changing_sequence(backend, tmp_path, names):
    pd.DataFrame({"Name": names, "Type": "Kernel", "Start Time(us)": np.arange(len(names)) + 1, "Duration(us)":
                  10}).to_csv(tmp_path / "kernel_details.csv", index=False)
    with pytest.raises(backend.policy.ProfilerAcquisitionError, match="Operation kernel"):
        backend.policy._collect_prof_result(tmp_path, [lambda: None], 0, 2, _aggregate_calls=True)


def test_operation_breakdown_keeps_repeated_kernel_positions(backend, tmp_path):
    pd.DataFrame({
        "Name": ["same"] * 6, "Type": "Kernel", "Start Time(us)": np.arange(6) + 1, "Duration(us)":
        [1000, 2000, 10, 40, 20, 60]
    }).to_csv(tmp_path / "kernel_details.csv", index=False)
    record, = backend.policy._collect_prof_result(tmp_path, [lambda: None], 1, 2, _aggregate_calls=True,
                                                  _capture_report=True)
    assert record.kernel_timings == (("same", 15., 15., 10., 20.), ("same", 50., 50., 40., 60.))


def test_operation_breakdown_is_optional(backend, tmp_path):
    pd.DataFrame({"Name": ["k"] * 2, "Type": "Kernel", "Start Time(us)": [1, 2], "Duration(us)":
                  [10, 20]}).to_csv(tmp_path / "kernel_details.csv", index=False)
    record, = backend.policy._collect_prof_result(tmp_path, [lambda: None], 0, 2, _aggregate_calls=True)
    assert record.kernel_timings == ()


@pytest.mark.parametrize("aggregate", [False, True])
def test_csv_capture_keeps_only_measured_rows(backend, tmp_path, aggregate):
    pd.DataFrame({
        "Name": ["k"] * 6, "Type": "Kernel", "Start Time(us)": np.arange(6) + 1, "Duration(us)":
        [1000, 2000, 10, 40, 20, 60]
    }).to_csv(tmp_path / "kernel_details.csv", index=False)
    with backend.policy.capture_npu_reports(lambda *a: None):
        record, = backend.policy._collect_prof_result(tmp_path, [lambda: None], 1 if aggregate else 2,
                                                      2 if aggregate else 4, _aggregate_calls=aggregate,
                                                      _capture_report=True)
    groups = record.kernel_rows
    assert len(groups) == (2 if aggregate else 1)
    assert [row["Duration(us)"]
            for _, rows in groups
            for row in rows] == ([10, 20, 40, 60] if aggregate else [10, 40, 20, 60])
    assert backend.policy._perf_report_sink.get() is None


def test_csv_capture_restores_nested_scopes(backend):
    outer, inner = lambda *a: None, lambda *a: None
    with backend.policy.capture_npu_reports(outer):
        with pytest.raises(RuntimeError):
            with backend.policy.capture_npu_reports(inner):
                assert backend.policy._perf_report_sink.get() is inner
                raise RuntimeError("export failed")
        assert backend.policy._perf_report_sink.get() is outer
    assert backend.policy._perf_report_sink.get() is None


@pytest.mark.parametrize("measured", [False, True])
def test_csv_capture_winner_works_without_console_reports(backend, make_tuner, measured, capsys):
    from importlib import import_module
    comparison = import_module(backend.testing.__package__ + "._autotune_comparison")
    tuner = make_tuner()
    tuner.report_best_config = False
    tuner.base_fn = tuner.fn
    tuner._autotune_selection = None
    selected = tuner.configs[0]
    record = backend.policy.NpuMeasurementReport({}, 2, 0, 2, "hot", .01, .01)
    captured = []
    with backend.policy.capture_npu_reports(lambda name, result: captured.append(result)):
        reports = comparison.capture_reports(tuner, True)
        assert reports == {}
        assert comparison.capture_reports(tuner, False) is None
        if measured:
            reports[selected] = record
        comparison.print_winner(tuner, selected, reports)
    assert reports == {} and captured == ([record] if measured else [])
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("exhausted", [False, True])
def test_batch_quality_status_survives_a_passing_winner(backend, monkeypatch, tmp_path, exhausted):
    policy = backend.policy
    calls, retained = [], []
    fast, slow = lambda: None, lambda: None

    def profile(funcs, warmup, active, *args, **kwargs):
        calls.append(list(funcs))
        return [
            policy.NpuMeasurementReport({}, active, warmup, active, "hot", .01 if fn is fast else .02,
                                        .01 if fn is fast else .02,
                                        samples=(np.arange(active), np.full(active, 10 if fn is fast else 20)))
            for fn in funcs
        ]

    monkeypatch.setattr(
        backend.quality, "evaluate_quality", lambda times, durations:
        ({}, ["unstable"] if durations[0] == 20 and (exhausted or len(calls) == 1) else []))
    options = replace(policy.resolve_options(), active=4, warmup=0, measure_budget_ms=None, filter_slow_configs=False,
                      max_retries=1, stability_runs=1, log_level="off")
    costs = policy.bench_npu(profile, [fast, slow], 0, 4, False, tmp_path, False, None, options=options,
                             report_sink=retained.extend)
    assert costs == [.01, .02] and calls == [[fast, slow], [slow]]
    assert not retained[0].quality_failures
    assert all(record.all_configs_passed == (not exhausted) for record in retained)
    # A later pass does not replace the earlier equally fast rejected attempt.
    assert retained[1].quality_failures == ("unstable", )


@pytest.fixture
def stability_benchmark(backend, monkeypatch, tmp_path):

    def run(series, *, scores=None, quality_failures=(), acquisition_failures=(), **settings):
        policy = backend.policy
        funcs = [lambda: None for _ in series]
        positions, batches, retained = [0] * len(funcs), [], []

        def profile(selected, warmup, active, *args, **kwargs):
            indices = [funcs.index(fn) for fn in selected]
            batches.append(indices)
            if len(batches) in acquisition_failures:
                raise OSError("missing CSV")
            records = []
            for index in indices:
                position = positions[index]
                positions[index] += 1
                mean = series[index][position]
                score = mean if scores is None else scores[index][position]
                records.append(
                    policy.NpuMeasurementReport(
                        {}, active, warmup, active, "hot", mean, score,
                        samples=(np.array([index, position]),
                                 np.full(active, mean)) if kwargs.get("_return_samples") else None))
            return records

        monkeypatch.setattr(
            backend.quality, "evaluate_quality", lambda times, durations, **kwargs:
            ({"intrinsic": 0.}, ["intrinsic failure"] if tuple(times) in quality_failures else []))
        options = replace(
            policy.resolve_options(),
            **dict(dict(active=2, warmup=0, measure_budget_ms=None, filter_slow_configs=False, log_level="off"),
                   **settings))
        costs = policy.bench_npu(profile, funcs, 0, 2, False, tmp_path, False, "same", options=options,
                                 report_sink=retained.extend)
        assert all(record.samples is None for record in retained)
        return costs, retained, batches

    return run


def test_stability_compares_each_candidate_and_retries_only_pending(stability_benchmark):
    costs, records, batches = stability_benchmark([[10., 10.3], [100., 110., 111.], [1000., 1001., 1002.]],
                                                  quality_failures={(2, 1)})
    assert batches == [[0, 1, 2], [0, 1, 2], [1, 2]]
    assert costs == [10., 100., 1000.]
    assert [bool(record.quality_failures) for record in records] == [False, True, False]
    assert records[0].quality_metrics["stability_relative_spread"] == pytest.approx(.03)
    assert all(record.all_configs_passed for record in records)


@pytest.mark.parametrize("runs", [2, 3])
def test_stability_requires_n_measurements_without_extra_retries(stability_benchmark, runs):
    cost, records, batches = stability_benchmark([[10.] * runs], stability_runs=runs, max_retries=0)
    assert cost == 10. and batches == [[0]] * runs
    assert not records[0].quality_failures and records[0].attempt == 1


def test_stability_slides_the_full_window_and_keeps_an_unconfirmed_minimum(stability_benchmark, capsys):
    cost, records, batches = stability_benchmark([[10., 11., 11.1, 11.2]], stability_runs=3, max_retries=1)
    assert cost == 10. and batches == [[0]] * 4
    assert records[0].quality_failures == ("stability not confirmed", )
    assert records[0].all_configs_passed
    text = capsys.readouterr().out
    assert "selected attempt quality: failed" in text
    assert "Not all configurations" not in text


def test_stability_confirmation_does_not_clear_the_selected_attempt_quality_failure(stability_benchmark):
    cost, records, _ = stability_benchmark([[10., 10.1]], quality_failures={(0, 0)})
    assert cost == 10. and records[0].quality_failures == ("intrinsic failure", )
    assert records[0].all_configs_passed


def test_stability_keeps_confirmation_after_the_selected_attempt_leaves_the_window(stability_benchmark):
    cost, records, batches = stability_benchmark([[10., 10.1, 11., 11.1]], quality_failures={(0, 1)})
    assert cost == 10. and batches == [[0]] * 4
    assert not records[0].quality_failures and records[0].all_configs_passed


def test_stability_selects_by_score_but_compares_untrimmed_means(stability_benchmark):
    cost, records, batches = stability_benchmark([[10., 11., 11.1]], scores=[[9., 8., 7.]])
    assert cost == 7. and batches == [[0]] * 3
    assert records[0].attempt == 3 and not records[0].quality_failures


def test_stability_acquisition_failure_does_not_confirm_an_old_minimum(stability_benchmark):
    cost, records, batches = stability_benchmark([[10., 11., 11.1]], acquisition_failures={2}, max_retries=2)
    assert cost == 10. and batches == [[0]] * 4
    assert records[0].quality_failures == ("stability not confirmed", )
    assert records[0].all_configs_passed


def test_stability_acquisition_failure_before_the_selected_attempt(stability_benchmark):
    cost, records, batches = stability_benchmark([[10., 10.1]], acquisition_failures={1}, max_retries=1)
    assert cost == 10. and batches == [[0]] * 3
    assert records[0].attempt == 2 and not records[0].quality_failures


def test_stability_reads_one_shared_csv_per_attempt(backend, monkeypatch, tmp_path, write_profile):
    policy = backend.policy
    reads, batches, retained = [], [], []
    read_csv = pd.read_csv

    def read(*args, **kwargs):
        reads.append(args[0])
        return read_csv(*args, **kwargs)

    def profile(funcs, warmup, active, cold, directory, keep, names, **kwargs):
        batches.append(len(funcs))
        write_profile(directory, [("same", 10.), ("same", 100.)], warmup, active)
        return policy._collect_prof_result(directory, funcs, warmup, active, names,
                                           _return_samples=kwargs["_return_samples"],
                                           _capture_report=kwargs["_capture_report"], _quantiles=kwargs["_quantiles"])

    monkeypatch.setattr(pd, "read_csv", read)
    options = replace(policy.resolve_options(), active=8, warmup=1, measure_budget_ms=None, filter_slow_configs=False,
                      max_retries=0, log_level="off")
    costs = policy.bench_npu(profile, [lambda: None, lambda: None], 1, 8, False, tmp_path, False, "same",
                             options=options, report_sink=retained.extend)
    assert costs == [.01, .1] and batches == [2, 2] and len(reads) == 2
    assert all(not record.quality_failures and record.all_configs_passed for record in retained)


def test_stability_exhaustion_reports_pending_configs_and_winner_status(stability_benchmark, capsys):
    costs, records, batches = stability_benchmark([[10., 10.1], [100., 110., 120.]], max_retries=1)
    assert costs == [10., 100.] and batches == [[0, 1], [0, 1], [1]]
    assert not records[0].quality_failures
    assert not any(record.all_configs_passed for record in records)
    text = capsys.readouterr().out
    assert "Not all configurations passed quality checks" in text
    assert "benchmark candidate #1" in text and "selected attempt quality: pass" in text


@pytest.mark.parametrize("settings", [{"stability_runs": 1}, {"quality_check": False}])
def test_stability_can_preserve_single_attempt_behavior(stability_benchmark, settings):
    cost, records, batches = stability_benchmark([[10.]], max_retries=0, **settings)
    assert cost == 10. and batches == [[0]]
    assert not records[0].quality_failures
    assert "stability_relative_spread" not in records[0].quality_metrics


def test_stability_single_run_keeps_the_old_quality_retry_and_minimum(stability_benchmark, capsys):
    cost, records, batches = stability_benchmark([[10., 11.]], stability_runs=1, max_retries=1,
                                                 quality_failures={(0, 0)})
    assert cost == 10. and batches == [[0], [0]]
    assert records[0].quality_failures == ("intrinsic failure", )
    assert records[0].all_configs_passed and capsys.readouterr().out == ""


@pytest.mark.parametrize("tolerance,attempts", [(0., 3), (.03, 3), (.05, 2)])
def test_stability_tolerance_controls_confirmation(stability_benchmark, tolerance, attempts):
    cost, records, batches = stability_benchmark([[10., 10.5, 10.5]], stability_tolerance=tolerance)
    assert cost == 10. and len(batches) == attempts
    assert records[0].all_configs_passed


def test_stability_environment_defaults_and_cache_keys(backend, monkeypatch):
    policy = backend.policy
    options = policy.resolve_options()
    assert (options.stability_runs, options.stability_tolerance) == (2, .03)
    assert options.cache_key() != replace(options, stability_tolerance=.04).cache_key()
    monkeypatch.setenv("TRITON_NPU_BENCH_STABILITY_RUNS", "1")
    monkeypatch.setenv("TRITON_NPU_BENCH_STABILITY_TOLERANCE", "0.05")
    single = policy.resolve_options()
    assert (single.stability_runs, single.stability_tolerance) == (1, .05)
    assert single.cache_key() == replace(single, stability_tolerance=.5).cache_key()
    disabled = replace(options, quality_check=False)
    assert disabled.cache_key() == replace(disabled, stability_runs=5, stability_tolerance=.5).cache_key()


@pytest.mark.parametrize("name,value", [("RUNS", "0"), ("RUNS", "2.5"), ("TOLERANCE", "-0.01"), ("TOLERANCE", "nan"),
                                        ("TOLERANCE", "inf")])
def test_stability_rejects_invalid_environment(backend, monkeypatch, name, value):
    monkeypatch.setenv("TRITON_NPU_BENCH_STABILITY_" + name, value)
    with pytest.raises(ValueError, match="STABILITY_" + name):
        backend.policy.resolve_options()
