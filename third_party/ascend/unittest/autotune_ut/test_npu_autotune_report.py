"""Selected-configuration reports from actual autotuning measurements."""

import csv
import io
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
                                          report_best_config=flag, **kwargs)(jit_kernel)
    assert isinstance(tuner, autotuner.AutoTilingTuner)
    assert tuner.fn is jit_kernel and tuner.report_best_config is flag
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
def test_csv_report_retains_all_values_and_uses_existing_row_selection(backend, tmp_path, shared_name, write_profile):
    write_profile(tmp_path, [("a", "10.0000"), ("a" if shared_name else "b", "20.5000")], 1, 2)
    reports = []
    target = "a" if shared_name else ["a", "b"]
    result = backend.testing._collect_prof_result(str(tmp_path), [lambda: None] * 2, 1, 2, target_kernel_name=target,
                                                  clear_l2_cache=True, _report_sink=reports.extend)
    assert result == [0.01, 0.0205]
    assert [report.mean_ms for report in reports] == result
    assert all(report.columns == COLUMNS and report.active == 2 and report.warmup == 1 for report in reports)
    assert [row["Task ID"] for row in reports[0].rows] == ["0003", "0004"]
    assert [row["Task ID"] for row in reports[1].rows] == ["0006", "0007"]
    assert all(row["Unexpected metric(%)"] == "001.2300" for report in reports for row in report.rows)
    assert reports[0].rows[0]["Duration(us)"] == "10.0000"


def test_report_formats_full_config_and_unabridged_csv(backend, tmp_path, capsys, write_profile):
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
    assert "Profiler kernel: winner" in output and "Mean duration: 10 us (0.01 ms)" in output
    assert "cache=cold, warmup=1, active=75, selected attempt=1" in output
    table = output[output.index(",".join(COLUMNS)):]
    reader = csv.DictReader(io.StringIO(table))
    assert tuple(reader.fieldnames) == COLUMNS
    rows = list(reader)
    assert len(rows) == 75 and rows[-1]["Task ID"] == "0077"
    assert all(row["Duration(us)"] == "10.0000" and row["Unexpected metric(%)"] == "001.2300" for row in rows)


@pytest.mark.parametrize("report_enabled", [False, True])
def test_budget_grouping_reports_selected_attempt_without_changing_measurements(backend, monkeypatch, tmp_path,
                                                                                report_enabled, write_profile):
    calls = []
    collected = []
    monkeypatch.setattr(backend.quality, "evaluate_quality", lambda times, durations: ({}, ["bad"]
                                                                                       if durations[0] == 20 else []))

    def profile(funcs, warmup, active, cache, directory, keep, names, **kwargs):
        calls.append((list(names), warmup, active))
        # Calibration, initial measurements, then a successful retry of b.
        durations = {"a": 10, "b": 1000} if warmup == 0 else {"a": 10, "b": 20 if len(calls) < 4 else 15}
        directory_path = Path(directory)
        write_profile(directory_path, [(name, str(durations[name])) for name in names], warmup, active)
        return backend.testing._collect_prof_result(
            directory_path, funcs, warmup, active, names, cache, _return_samples=True,
            **({"_report_sink": kwargs["_report_sink"]} if "_report_sink" in kwargs else {}))

    monkeypatch.setattr(backend.testing, "_profile_npu", profile)
    result = backend.testing.do_bench_npu([lambda: None] * 2, target_kernel_name=["a",
                                                                                  "b"], prof_dir=tmp_path / "temporary",
                                          npu_bench_options=dict(quality_check=True, max_retries=1,
                                                                 measure_budget_ms=5),
                                          **({"_report_sink": collected.extend} if report_enabled else {}))
    assert result == [0.01, 0.015]
    assert calls == [(["a", "b"], 0, 10), (["a"], 5, 500), (["b"], 5, 30), (["b"], 5, 30)]
    if report_enabled:
        assert [report.attempt for report in collected] == [1, 2]
        assert [report.active for report in collected] == [500, 30]
        assert [report.mean_ms for report in collected] == result
        assert collected[1].rows[0]["Duration(us)"] == "15"
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
    report = backend.report.NpuMeasurementReport(COLUMNS, (row, ), 5, 30, "cold", .01)

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
    assert "NPU profiler measurements unavailable" in capsys.readouterr().out
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
                [backend.report.NpuMeasurementReport(COLUMNS, (row, ), warmup, active, "hot", duration / 1000)])
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
        assert len(reports[0].rows) == 2 and reports[0].rows[0]["Unexpected metric(%)"] == "001.2300"
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
        assert "kernel,Kernel,3,10.0000,001.2300,0003" in output
        assert "20.0000" not in output
    else:
        assert output == ""
