"""Selected-configuration reports from actual autotuning measurements."""

import warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import torch
import torch_npu
from triton import Config
from triton.backends.ascend.runtime import autotuner

COLUMNS = ("Name", "Type", "Start Time(us)", "Duration(us)", "Unexpected metric(%)", "Task ID")


@pytest.mark.parametrize("decorator,flag", [("autotune", True), ("max_autotune", False)])
def test_public_decorators_forward_options(make_tuner, jit_kernel, decorator, flag):
    make_tuner()
    kwargs = {"kernel_type": "vector"} if decorator == "max_autotune" else {}
    options = {"active": 2}
    tuner = getattr(autotuner, decorator)([Config({"BLOCK": 64}, num_stages=1)], [], npu_bench_options=options,
                                          report_best_config=flag, report_timing=flag, **kwargs)(jit_kernel)
    assert isinstance(tuner, autotuner.AutoTilingTuner)
    assert tuner.fn is jit_kernel and tuner.report_best_config is flag and tuner.report_timing is flag
    assert tuner.npu_bench_options == options and tuner.npu_bench_options is not options


@pytest.mark.parametrize("argument,environment,expected", [
    (None, None, False),
    (None, "1", True),
    (None, "false", False),
    (False, "true", False),
    (True, "invalid but overridden", True),
])
def test_report_option_priority(backend, monkeypatch, argument, environment, expected):
    if environment is not None:
        monkeypatch.setenv(backend.report.REPORT_ENV, environment)
    assert backend.report.resolve_report_best_config(argument) is expected


def test_invalid_report_option(backend, monkeypatch):
    with pytest.raises(ValueError, match="report_best_config"):
        backend.report.resolve_report_best_config(1)
    monkeypatch.setenv(backend.report.REPORT_ENV, "invalid")
    with pytest.raises(ValueError, match=backend.report.REPORT_ENV):
        backend.report.resolve_report_best_config()


@pytest.mark.parametrize("shared_name", [False, True])
def test_csv_report_aggregates_existing_row_selection(backend, tmp_path, shared_name, write_profile):
    write_profile(tmp_path, [("a", "10.0000"), ("a" if shared_name else "b", "20.5000")], 1, 2)
    reports = []
    target = "a" if shared_name else ["a", "b"]
    result = backend.testing._collect_prof_result(str(tmp_path), [lambda: None] * 2, 1, 2, target_kernel_name=target,
                                                  clear_l2_cache=True, _report_sink=reports.extend)
    assert result == [0.01, 0.0205]
    assert [report.mean_ms for report in reports] == result
    assert all(report.sample_count == 2 and report.active == 2 and report.warmup == 1 for report in reports)
    assert [report.metrics["Task ID"] for report in reports] == [("varies", 2, False)] * 2
    assert all(report.metrics["Unexpected metric(%)"] == (1.23, 2, True) for report in reports)
    assert reports[0].metrics["Duration(us)"] == (10, 2, True)
    assert all(not hasattr(report, "rows") for report in reports)


def test_report_formats_full_config_and_profiler_aggregates(backend, tmp_path, capsys, write_profile):
    write_profile(tmp_path, [("winner", "10.0000")], 1, 75)
    reports = []
    backend.profiler._read_profile_samples(tmp_path, ["winner"], 1, 75, True, _report_sink=reports.extend)
    configs = [Config({}), Config({}), Config({})]
    selected = configs[2]
    selected.kwargs = {"BLOCK": 128}
    selected.num_warps, selected.num_stages, selected.maxnreg = 4, 2, None
    selected.ubtune_cfg = {"multibuffer": True}
    backend.report.print_best_config_report("example", configs, selected, reports[0])
    output = capsys.readouterr().out
    assert "Selected config: 3/3" in output
    assert "'BLOCK': 128" in output and "'num_stages': 2" in output and "'maxnreg': None" in output
    assert "ubtune_cfg:" in output and "'multibuffer': True" in output
    assert "Name: value=winner" in output and "Mean duration: 10 us (0.01 ms)" in output
    assert "cache=cold, warmup=1, active=75, selected attempt=1" in output
    assert "Profiler aggregates: 75 measured launches" in output
    assert "Unexpected metric(%): mean=1.23" in output
    assert "Task ID: value=varies" in output and "Start Time(us): value=varies" in output
    assert "001.2300" not in output and ",".join(COLUMNS) not in output
    assert len(output.splitlines()) < 40


@pytest.mark.parametrize("report_enabled", [False, True])
def test_budget_batch_reports_selected_attempt_without_changing_measurements(backend, monkeypatch, tmp_path,
                                                                             report_enabled, write_profile):
    calls = []
    collected = []
    monkeypatch.setattr(backend.quality, "evaluate_quality", lambda times, durations: ({}, ["bad"]
                                                                                       if durations[0] == 20 else []))

    def profile(funcs, warmup, active, cache, directory, keep, names, **kwargs):
        counts = kwargs.get("_active_counts", [active] * len(funcs))
        calls.append((list(names), warmup, counts))
        # Calibration, initial measurements, then a successful retry of b.
        durations = {"a": 10, "b": 1000} if warmup == 0 else {"a": 10, "b": 20 if len(calls) < 3 else 15}
        directory_path = Path(directory)
        write_profile(directory_path, [(name, str(durations[name])) for name in names], warmup, active, counts)
        return backend.testing._collect_prof_result(
            directory_path, funcs, warmup, active, names, cache, _return_samples=True, _active_counts=counts,
            **({"_report_sink": kwargs["_report_sink"]} if "_report_sink" in kwargs else {}))

    monkeypatch.setattr(backend.testing, "_profile_npu", profile)
    result = backend.testing.do_bench_npu([lambda: None] * 2, target_kernel_name=["a",
                                                                                  "b"], prof_dir=tmp_path / "temporary",
                                          npu_bench_options=dict(quality_check=True, max_retries=1,
                                                                 measure_budget_ms=5),
                                          **({"_report_sink": collected.extend} if report_enabled else {}))
    assert result == [0.01, 0.015]
    assert calls == [(["a", "b"], 0, [10, 10]), (["a", "b"], 5, [500, 30]), (["b"], 5, [30])]
    if report_enabled:
        assert [report.attempt for report in collected] == [1, 2]
        assert [report.active for report in collected] == [500, 30]
        assert [report.mean_ms for report in collected] == result
        assert collected[1].metrics["Duration(us)"] == (15, 30, True)
        assert collected[1].quality_failures == ()
    else:
        assert collected == []
    assert list((tmp_path / "temporary").iterdir()) == []


@pytest.mark.parametrize("report_enabled", [False, True])
def test_run_reports_original_number_once_and_keeps_selection_and_cache(backend, capsys, report_enabled, configure_run):
    configs = [Config({}), Config({}), Config({})]
    tuner = configure_run(configs[2], configs, report_enabled)
    calls = []
    row = {name: "value" for name in COLUMNS}
    row.update(Name="winner", **{"Duration(us)": "10", "Start Time(us)": "1"})
    report = backend.report.make_profile_report(COLUMNS, [row], 5, 30, True, .01)

    def batch(*args, configs, **kwargs):
        calls.append(1)
        if "_report_sink" in kwargs:
            kwargs["_report_sink"]({configs[0]: report})
        return {configs[0]: .01, configs[1]: .02}

    tuner._batch_bench = batch
    assert tuner.run(torch.empty(1)) == "result"
    output = capsys.readouterr().out
    assert ("Selected config: 3/3" in output) is report_enabled
    assert tuner.best_config is configs[2] and tuner.cache == {("torch.float32", ): configs[2]}
    assert tuner.run(torch.empty(1)) == "result"
    assert capsys.readouterr().out == "" and calls == [1]


def test_selection_without_profiler_data_does_not_add_measurements(capsys, configure_run):
    config = Config({})
    tuner = configure_run(config, [config], True)
    tuner.prune_configs = lambda kwargs: [config]
    tuner._batch_bench = lambda *args, **kwargs: pytest.fail("extra measurement")
    assert tuner.run(torch.empty(1)) == "result"
    assert "Measurements unavailable" in capsys.readouterr().out
    assert tuner.run(torch.empty(1)) == "result"
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("failure", ["observer", "formatting", "warning"])
def test_diagnostic_failure_does_not_change_measurements_or_retry(backend, monkeypatch, tmp_path, write_profile,
                                                                  failure):
    write_profile(tmp_path, [("a", "10.0000")], 0, 2)

    class UnprintableError(RuntimeError):

        def __str__(self):
            raise ValueError("diagnostic formatting failed")

    if failure == "warning":

        def warn(*args, **kwargs):
            raise OSError("warning emission failed")

        monkeypatch.setattr(backend.report.warnings, "warn", warn)

    def fail(reports):
        raise UnprintableError() if failure == "formatting" else RuntimeError("report failure")

    # The optional observer fails after sample validation; even warnings-as-errors
    # must not turn it into a profiler acquisition failure.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        samples = backend.profiler._read_profile_samples(tmp_path, ["a"], 0, 2, True, _report_sink=fail)
    assert samples[0][1].tolist() == [10, 10]


def test_retry_report_keeps_best_attempt_even_when_last_attempt_is_slower(backend, monkeypatch, tmp_path, capsys,
                                                                          sample):
    durations = iter([12, 10, 14])
    reports, calls = [], []

    def measure(funcs, names, warmup, active, directory, **kwargs):
        duration = next(durations)
        calls.append(duration)
        row = dict(zip(COLUMNS, ["a", "Kernel", "1", str(duration), "001.2300", "0001"]))
        if "_report_sink" in kwargs:
            kwargs["_report_sink"](
                [backend.report.make_profile_report(COLUMNS, [row], warmup, active, False, duration / 1000)])
        return [sample(duration, count=active)]

    monkeypatch.setattr(backend.quality, "evaluate_quality", lambda *args: ({}, ["bad quality"]))
    with pytest.warns(RuntimeWarning, match="exhausted"):
        costs = backend.policy.benchmark_with_options(
            measure,
            [lambda: None],
            ["a"],
            backend.policy.resolve_options(dict(quality_check=True, max_retries=2)),
            warmup=5,
            active=30,
            prof_root=tmp_path,
            synchronize=lambda: None,
            _report_sink=reports.extend,
        )
    assert costs == [.01] and calls == [12, 10, 14]
    assert reports[0].attempt == 2 and reports[0].mean_ms == costs[0]
    assert reports[0].quality_failures == ("bad quality", )
    config = Config({})
    backend.report.print_best_config_report("example", [config], config, reports[0])
    assert "Selected measurement failed quality checks: bad quality" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("report_enabled", [False, True])
def test_real_profiler_path_captures_report_before_cleanup_without_extra_launches(backend, monkeypatch, tmp_path,
                                                                                  report_enabled, write_profile):
    calls, reports = [], []

    @contextmanager
    def profile(**kwargs):
        yield
        write_profile(kwargs["on_trace_ready"], [("a", "10.0000")], 1, 2)

    monkeypatch.setattr(torch_npu.profiler, "profile", profile)
    monkeypatch.setattr(torch_npu.profiler, "tensorboard_trace_handler", lambda path: path)
    directory = tmp_path / "trace"
    result = backend.testing.do_bench_npu(lambda: calls.append(1), warmup=1, active=2, prof_dir=directory,
                                          target_kernel_name="a",
                                          **({"_report_sink": reports.extend} if report_enabled else {}))
    assert result == .01 and len(calls) == 4
    assert not directory.exists()
    if report_enabled:
        assert len(reports) == 1 and reports[0].mean_ms == result
        assert reports[0].sample_count == 2 and reports[0].metrics["Unexpected metric(%)"] == (1.23, 2, True)
    else:
        assert reports == []


def test_print_failure_does_not_change_run_result(backend, monkeypatch, configure_run):
    config = Config({})
    tuner = configure_run(config, [config], True)
    tuner.prune_configs = lambda kwargs: [config]

    def fail(*args):
        raise OSError("output failure")

    monkeypatch.setattr(backend.report, "print_best_config_report", fail)
    with pytest.warns(RuntimeWarning, match="output failure"):
        assert tuner.run(torch.empty(1)) == "result"


@pytest.mark.parametrize("options", [None, dict(active=30, measure_budget_ms=5)])
def test_reporting_does_not_change_actual_autotune_cache_key(backend, monkeypatch, options, make_tuner):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    tuner = make_tuner(options)
    tuner.keys, tuner.arg_names = [], ["x"]

    tensor = torch.empty(1)
    tuner.report_best_config = False
    without_report = tuner.generate_key_and_configs(tensor)
    monkeypatch.setenv(backend.report.REPORT_ENV, "1")
    tuner.report_best_config = True
    assert tuner.generate_key_and_configs(tensor) == without_report


@pytest.mark.parametrize("report_enabled", [False, True])
def test_batch_report_mapping_survives_candidate_reordering(backend, monkeypatch, tmp_path, capsys, report_enabled,
                                                            configure_run, write_profile):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    configs = [Config({}), Config({}), Config({})]
    tuner = configure_run(configs[2], configs, report_enabled)
    tuner.prune_configs = lambda kwargs: list(configs)
    tuner.fn.run = lambda *args, **kwargs: SimpleNamespace(packed_metadata={"kernel_name": "kernel"})
    calls = []

    def estimate(funcs):
        tuner.cv_warmup, tuner.cv_repeat = 1, 2
        return {configs[2]: funcs[configs[2]], configs[1]: funcs[configs[1]]}

    def profile(funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target, **kwargs):
        calls.append((len(funcs), warmup, active))
        write_profile(tmp_path, [("kernel", "10.0000"), ("kernel", "20.0000")], warmup, active)
        return backend.testing._collect_prof_result(tmp_path, funcs, warmup, active, target, clear_l2_cache, **kwargs)

    tuner._prune_by_time_limit = estimate
    monkeypatch.setattr(backend.testing, "_profile_npu", profile)
    assert tuner.run(torch.empty(1)).packed_metadata == {"kernel_name": "kernel"}
    assert tuner.best_config is configs[2] and calls == [(2, 1, 2)]
    output = capsys.readouterr().out
    if report_enabled:
        assert "Selected config: 3/3" in output
        assert "Mean duration: 10 us (0.01 ms)" in output
        assert "Unexpected metric(%): mean=1.23" in output
        assert "20.0000" not in output
    else:
        assert output == ""


@pytest.mark.parametrize("shared_name", [False, True])
def test_real_batch_profiles_once_and_reads_once_per_attempt(backend, monkeypatch, tmp_path, write_profile,
                                                             shared_name):
    sessions, reads, reports = [], [], []
    current = [None]
    original_open = Path.open

    def open_csv(path, *args, **kwargs):
        if path.name == "kernel_details.csv" and (not args or args[0] == "r"):
            reads.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_csv)
    names = ["kernel", "kernel" if shared_name else "other"]

    @contextmanager
    def profile(**kwargs):
        counts = [0, 0]
        current[0] = counts
        yield
        current[0] = None
        sessions.append(counts)
        attempt = len(sessions)
        warmup = 0 if attempt == 1 else 1
        indices = [i for i, count in enumerate(counts) if count]
        durations = [10, 1000] if attempt == 1 else [10, 20] if attempt == 2 else [10, 15]
        write_profile(kwargs["on_trace_ready"], [(names[i], durations[i]) for i in indices], warmup, 2,
                      [counts[i] - warmup for i in indices])

    def kernel(index):
        if current[0] is not None:
            current[0][index] += 1

    monkeypatch.setattr(torch_npu.profiler, "profile", profile)
    monkeypatch.setattr(torch_npu.profiler, "tensorboard_trace_handler", lambda path: path)
    monkeypatch.setattr(
        backend.quality, "evaluate_quality", lambda times, durations: ({"metric": float(durations.mean())}, ["bad"]
                                                                       if durations[0] == 20 else []))
    result = backend.testing.do_bench_npu([lambda: kernel(0), lambda: kernel(1)], warmup=1, active=2,
                                          target_kernel_name=names, prof_dir=tmp_path,
                                          npu_bench_options=dict(measure_budget_ms=1, quality_check=True,
                                                                 max_retries=1), _report_sink=reports.extend)
    assert result == [.01, .015]
    assert sessions == [[10, 10], [101, 3], [0, 3]]
    assert len(reads) == len(sessions) == 3
    assert [report.active for report in reports] == [100, 2]
    assert reports[1].attempt == 2 and reports[1].quality_metrics == {"metric": 15}
    assert not any(hasattr(report, "rows") for report in reports)
    assert list(tmp_path.iterdir()) == []


def test_report_does_not_reread_legacy_csv(backend, monkeypatch, tmp_path, write_profile):
    import pandas as pd

    write_profile(tmp_path, [("kernel", 10)], 1, 2)
    reads = []
    original = pd.read_csv

    def read(*args, **kwargs):
        reads.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read)
    reports = []
    assert backend.testing._collect_prof_result(tmp_path, [lambda: None], 1, 2, "kernel",
                                                _report_sink=reports.extend) == .01
    assert reads == [1] and reports[0].metrics["Duration(us)"] == (10, 2, True)


def test_partial_metrics_identifiers_and_text(backend, capsys):
    rows = [{"Metric": "2", "Missing": "", "Task ID": "001", "Start Time(us)": "10", "Text":
             "a"}, {"Metric": "", "Missing": None, "Task ID": "002", "Start Time(us)": "20", "Text": "b"},
            {"Metric": "4", "Missing": "", "Task ID": "003", "Start Time(us)": "30", "Text": "a"}]
    report = backend.report.make_profile_report(tuple(rows[0]), rows, 1, 3, False, .01)
    assert report.metrics["Metric"] == (3, 2, True)
    assert report.metrics["Missing"] == ("unavailable", 0, False)
    assert report.metrics["Task ID"] == report.metrics["Start Time(us)"] == report.metrics["Text"] == ("varies", 3,
                                                                                                       False)
    config = Config({})
    backend.report.print_best_config_report("kernel", [config], config, report)
    output = capsys.readouterr().out
    assert "Metric: mean=3 (2/3 available)" in output
    assert "Missing: value=unavailable (0/3 available)" in output


def test_nonfinite_and_unavailable_profiler_metrics(backend):
    rows = [{"Partial": 2, "Empty": "N/A"}, {"Partial": float("inf"), "Empty": "null"},
            {"Partial": 4, "Empty": float("nan")}]
    metrics = backend.report.aggregate_profile_rows(tuple(rows[0]), rows)
    assert metrics == {"Partial": (3, 2, True), "Empty": ("unavailable", 0, False)}


@pytest.mark.parametrize("custom", [False, True])
def test_event_and_custom_scores_use_existing_measurements(backend, monkeypatch, capsys, configure_run, custom):
    configs = [Config({}), Config({})]
    score = [1, .8, 1.2] if not custom else 123
    calls = []

    def measure(fn, **kwargs):
        calls.append(kwargs)
        return score

    tuner = configure_run(configs[1], configs, True, do_bench=measure if custom else None, report_timing=True)
    tuner.do_bench = measure
    tuner.parser_mode = "vector"
    monkeypatch.setattr(tuner, "_make_kernel_call", lambda *args, **kwargs: lambda **kw: None)
    assert tuner.run(torch.empty(1)) == "result"
    output = capsys.readouterr().out
    assert len(calls) == 2 and all(call == {"quantiles": (.5, .2, .8)} for call in calls)
    assert "Returned score: 123" in output if custom else "Median: 1 ms; quantiles 20%/80%: 0.8/1.2 ms" in output
    assert output.count("Triton autotuning timing") == 1
    assert tuner.run(torch.empty(1)) == "result" and len(calls) == 2
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("environment,argument,expected", [(None, None, False), ("1", None, True),
                                                           ("false", None, False), ("true", False, False),
                                                           ("invalid", True, True)])
def test_timing_option_priority(backend, monkeypatch, environment, argument, expected):
    if environment is not None:
        monkeypatch.setenv(backend.report.TIMING_ENV, environment)
    assert backend.report.resolve_report_timing(argument) is expected


def test_invalid_timing_option(backend, monkeypatch):
    with pytest.raises(ValueError, match="report_timing"):
        backend.report.resolve_report_timing(1)
    monkeypatch.setenv(backend.report.TIMING_ENV, "invalid")
    with pytest.raises(ValueError, match=backend.report.TIMING_ENV):
        backend.report.resolve_report_timing()


@pytest.mark.parametrize("level", ["off", "brief", "detailed"])
@pytest.mark.parametrize("winner,timing", [(False, False), (True, False), (False, True), (True, True)])
def test_output_controls_are_independent_and_cache_hits_silent(backend, monkeypatch, capsys, configure_run, level,
                                                               winner, timing):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    configs = [Config({}), Config({})]
    tuner = configure_run(configs[1], configs, winner, report_timing=timing, options={"log_level": level})
    tuner._batch_bench = lambda *args, configs, **kwargs: dict(zip(configs, [1, 2]))
    assert tuner.run(torch.empty(1)) == "result"
    output = capsys.readouterr().out
    assert ("NPU autotune:" in output) is (level != "off")
    assert ("Selected config:" in output) is winner
    assert output.count("Triton autotuning timing") == int(timing)
    assert tuner.run(torch.empty(1)) == "result" and capsys.readouterr().out == ""


def test_timing_accounts_for_profiler_exit_csv_and_quality_once(backend, monkeypatch, tmp_path, write_profile):
    clock, timing = [0.0], {}
    monkeypatch.setattr(backend.report.time, "perf_counter", lambda: clock[0])
    original_read = backend.profiler._read_profile_samples

    def read(*args, **kwargs):
        clock[0] += 3
        return original_read(*args, **kwargs)

    @contextmanager
    def profile(**kwargs):
        clock[0] += 1
        yield
        clock[0] += 2
        write_profile(kwargs["on_trace_ready"], [("kernel", 10)], 0, 2)

    def quality(*args):
        clock[0] += 4
        return {}, []

    monkeypatch.setattr(backend.profiler, "_read_profile_samples", read)
    monkeypatch.setattr(torch_npu.profiler, "profile", profile)
    monkeypatch.setattr(torch_npu.profiler, "tensorboard_trace_handler", lambda path: path)
    monkeypatch.setattr(backend.quality, "evaluate_quality", quality)
    assert backend.testing.do_bench_npu(lambda: None, warmup=0, active=2, target_kernel_name="kernel",
                                        prof_dir=tmp_path, npu_bench_options={"quality_check": True},
                                        _timing_sink=lambda name, elapsed: timing.update({name: elapsed})) == .01
    assert timing == {"slow_filter": 0, "calibration": 0, "measurements": 10}
    backend.report.print_timing_report("kernel", 0, timing)


def test_final_timing_breakdown_has_no_double_counting(backend, monkeypatch, capsys):
    monkeypatch.setattr(backend.report.time, "perf_counter", lambda: 25)
    backend.report.print_timing_report(
        "kernel", 0,
        {"generation_pruning": 2, "compilation": 3, "slow_filter": 4, "calibration": 5, "measurements": 10})
    output = capsys.readouterr().out
    assert "Total: 25.000000 s" in output and "Other: 1.000000 s" in output
    assert "Measurements (including retries): 10.000000 s" in output


@pytest.mark.parametrize("level", ["brief", "detailed"])
def test_stage_logging_preserves_actual_selection(backend, monkeypatch, capsys, configure_run, level):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    configs = [Config({"BLOCK": 64}), Config({"BLOCK": 128})]
    tuner = configure_run(configs[1], configs, False, options={"log_level": level})
    tuner.parser_mode = "vector"
    calls = []

    def profile(funcs, warmup, active, *args, **kwargs):
        calls.append(len(funcs))
        return [.01, .02]

    monkeypatch.setattr(backend.testing, "_profile_npu", profile)
    assert tuner.run(torch.empty(1)) == "result"
    output = capsys.readouterr().out
    assert tuner.best_config is configs[1] and calls == [2]
    for title in ("Initial combinations: 2", "Compilation", "Pruning result", "Attempt 1: 2 candidates"):
        assert title in output
    assert ("parameters=" in output) is (level == "detailed")
    assert "Triton autotuning timing" not in output and "Selected config:" not in output
    assert tuner.run(torch.empty(1)) == "result" and calls == [2] and capsys.readouterr().out == ""


def test_reporting_and_logging_preserve_actual_cache_key(backend, monkeypatch, make_tuner):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    tuner = make_tuner({"active": 30})
    tensor = torch.empty(1)
    key = tuner.generate_key_and_configs(tensor)
    tuner.npu_bench_options = {"active": 30, "log_level": "detailed"}
    tuner.report_timing = tuner.report_best_config = True
    assert tuner.generate_key_and_configs(tensor) == key


def test_single_candidate_timing_reports_skipped_stages(capsys, configure_run):
    config = Config({})
    tuner = configure_run(config, [config], False, report_timing=True)
    tuner.prune_configs = lambda kwargs: [config]
    tuner._batch_bench = lambda *args, **kwargs: pytest.fail("extra measurement")
    assert tuner.run(torch.empty(1)) == "result"
    output = capsys.readouterr().out
    assert output.count("Triton autotuning timing") == 1
    assert "Compilation: 0.000000 s" in output and "Measurements (including retries): 0.000000 s" in output
    assert tuner.run(torch.empty(1)) == "result" and capsys.readouterr().out == ""
