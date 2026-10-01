"""Policies exercised through the installed NPU backend."""

import csv
import functools
import inspect
import subprocess
import sys
import warnings
from contextlib import contextmanager, nullcontext

import numpy as np
import pytest
import torch
import torch_npu
from triton import Config
from triton.runtime.driver import driver


def test_non_quality_paths_do_not_load_scipy(backend):
    subprocess.run([
        sys.executable, "-W", "error", "-c", '''
import sys
import tempfile
import numpy as np
from triton.backends.ascend import _npu_benchmark as policy, testing
testing._profile_npu = lambda *args, **kwargs: 1
assert testing.do_bench_npu(lambda: None) == 1
assert testing.do_bench_npu(lambda: None, npu_bench_options={"cache_mode": "hot"}) == 1
options = policy.resolve_options({"max_retries": 1, "measure_budget_ms": 1})
options.cache_key()
with tempfile.TemporaryDirectory() as directory:
    costs = policy.benchmark_with_options(
        lambda funcs, names, warmup, active, path: [(np.arange(active), np.full(active, 100.))],
        [lambda: None], ["kernel"], options, warmup=0, active=2,
        prof_root=directory, synchronize=lambda: None)
assert costs == [0.1]
assert "triton.backends.ascend._benchmark_quality" not in sys.modules
assert not any(name == "scipy" or name.startswith("scipy.") for name in sys.modules)
'''
    ], check=True, capture_output=True, text=True)


def test_argument_overrides_environment_per_field(backend, monkeypatch):
    policy = backend.policy
    monkeypatch.setenv(policy.ENV_PREFIX + "CACHE_MODE", "hot")
    monkeypatch.setenv(policy.ENV_PREFIX + "QUALITY_CHECK", "true")
    monkeypatch.setenv(policy.ENV_PREFIX + "MAX_RETRIES", "invalid but overridden")
    monkeypatch.setenv(policy.ENV_PREFIX + "ACTIVE", "71")
    monkeypatch.setenv(policy.ENV_PREFIX + "FILTER_SLOW_CONFIGS", "true")
    options = policy.resolve_options(
        dict(cache_mode="cold", quality_check=False, max_retries=0, filter_slow_configs=False))
    assert options.cache_mode == "cold"
    assert options.quality_check is False and options.max_retries == 0
    assert options.active == 71 and options.measure_budget_ms is None
    assert options.filter_slow_configs is False


@pytest.mark.parametrize("options", [
    dict(quality_check=1),
    dict(active=True),
    dict(warmup=-1),
    dict(max_retries=-1),
    dict(measure_budget_ms=float("nan")),
    dict(filter_slow_configs="true"),
    dict(cache_mode="L1"),
    dict(unknown=1),
    dict(quality_check=True, active=1),
    dict(slow_config_factor=0.5),
    dict(slow_config_runs=0),
    dict(slow_config_recheck_runs=0),
    dict(calibration_runs=1),
])
def test_invalid_options(backend, options):
    with pytest.raises(ValueError):
        backend.policy.resolve_options(options)


def test_only_failed_candidates_are_remeasured(backend, tmp_path, monkeypatch, run_policy, sample):
    calls = []

    def measure(funcs, names, warmup, active, directory):
        calls.append((names, warmup, active))
        return [sample(10), sample(20)] if len(calls) == 1 else [sample(15)]

    monkeypatch.setattr(backend.quality, "evaluate_quality", lambda t, d: ({}, ["bad"] if d[0] == 20 else []))
    result = run_policy(measure, dict(quality_check=True, max_retries=2))
    assert result == [0.01, 0.015]
    assert calls == [(["a", "b"], 5, 30), (["b"], 5, 30)]
    assert list(tmp_path.iterdir()) == []


def test_quality_check_does_not_enable_budget_or_retries(backend, monkeypatch, run_policy, sample):
    calls = []
    monkeypatch.setattr(backend.quality, "evaluate_quality", lambda t, d: ({}, ["bad"]))

    def measure(funcs, names, warmup, active, directory):
        calls.append((warmup, active))
        return [sample() for _ in funcs]

    with pytest.warns(RuntimeWarning, match="failed quality checks"):
        run_policy(measure, dict(quality_check=True))
    assert calls == [(5, 30)]


def test_budget_calibrates_without_enabling_quality(backend, monkeypatch, run_policy, sample):
    calls = []
    monkeypatch.setattr(backend.quality, "evaluate_quality", lambda *args: pytest.fail("quality unexpectedly enabled"))

    def measure(funcs, names, warmup, active, directory):
        calls.append((warmup, active))
        return [sample() for _ in funcs]

    assert run_policy(measure, dict(measure_budget_ms=1)) == [0.01, 0.01]
    assert calls == [(0, 10), (5, 100)]


def test_budget_groups_individual_counts_and_preserves_candidate_order(tmp_path, run_policy, sample):
    durations = {"a": 10, "b": 1000, "c": 10}
    calls, executed = [], []

    def measure(funcs, names, warmup, active, directory):
        calls.append((names, warmup, active))
        for fn in funcs:
            fn()
        return [sample(durations[name], count=active) for name in names]

    result = run_policy(measure, dict(measure_budget_ms=5),
                        funcs=[lambda name=name: executed.append(name) for name in durations], names=list(durations))
    assert result == [0.01, 1, 0.01]
    assert calls == [(["a", "b", "c"], 0, 10), (["a", "c"], 5, 500), (["b"], 5, 30)]
    assert executed == ["a", "b", "c", "a", "c", "b"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["quality", "acquisition"])
@pytest.mark.parametrize("failed_group", ["small", "large"])
def test_budget_retries_only_failed_candidates_with_their_counts(backend, tmp_path, monkeypatch, failure, failed_group,
                                                                 run_policy, sample):
    durations = {"a": 10, "b": 1000, "c": 10}
    calls = []
    failure_call = 2 if failed_group == "small" else 3
    failure_duration = 10 if failed_group == "small" else 1000
    monkeypatch.setattr(backend.quality, "evaluate_quality", lambda t, d:
                        ({}, ["bad"] if d[0] == failure_duration and len(calls) == failure_call else []))

    def measure(funcs, names, warmup, active, directory):
        calls.append((names, warmup, active))
        if failure == "acquisition" and len(calls) == failure_call:
            raise backend.policy.ProfilerAcquisitionError("missing rows")
        return [sample(durations[name], count=active) for name in names]

    warning = (pytest.warns(RuntimeWarning, match="selected config 0")
               if failure == "quality" and failed_group == "small" else nullcontext())
    with warning:
        result = run_policy(measure, dict(measure_budget_ms=5, quality_check=failure == "quality", max_retries=1),
                            funcs=[lambda: None] * 3, names=["a", "b", "c"])
    assert result == [0.01, 1, 0.01]
    expected_retry = (["a", "c"], 5, 500) if failed_group == "small" else (["b"], 5, 30)
    assert calls == [(["a", "b", "c"], 0, 10), (["a", "c"], 5, 500), (["b"], 5, 30), expected_retry]
    assert list(tmp_path.iterdir()) == []


def test_budget_calibration_retries_acquisition_failure(backend, tmp_path, run_policy, sample):
    calls = []

    def measure(funcs, names, warmup, active, directory):
        calls.append((names, warmup, active))
        if len(calls) == 1:
            raise backend.policy.ProfilerAcquisitionError("missing calibration")
        return [sample(10 if name == "a" else 1000, count=active) for name in names]

    assert run_policy(measure, dict(measure_budget_ms=5, max_retries=1)) == [0.01, 1]
    assert calls == [(["a", "b"], 0, 10), (["a", "b"], 0, 10), (["a"], 5, 500), (["b"], 5, 30)]
    assert list(tmp_path.iterdir()) == []


def test_acquisition_failure_is_retried(backend, tmp_path, run_policy, sample):
    calls = []

    def measure(*args):
        calls.append(1)
        if len(calls) == 1:
            raise backend.policy.ProfilerAcquisitionError("missing rows")
        return [sample(), sample()]

    assert run_policy(measure, dict(max_retries=1)) == [0.01, 0.01]
    assert len(calls) == 2 and list(tmp_path.iterdir()) == []


def test_kernel_errors_are_never_retried(tmp_path, run_policy):
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("kernel failure")

    def measure(funcs, *args):
        funcs[0]()

    with pytest.raises(RuntimeError, match="kernel failure"):
        run_policy(measure, dict(max_retries=4), funcs=[broken])
    assert calls == [1] and list(tmp_path.iterdir()) == []


def test_best_measurement_survives_exhausted_quality_retries(backend, monkeypatch, run_policy, sample):
    durations = iter([12, 10, 14])
    monkeypatch.setattr(backend.quality, "evaluate_quality", lambda t, d: ({}, ["bad"]))
    with pytest.warns(RuntimeWarning, match="exhausted"):
        result = run_policy(lambda *args: [sample(next(durations))], dict(quality_check=True, max_retries=2),
                            funcs=[lambda: None])
    assert result == [0.01]


def test_real_quality_metrics_accept_stable_samples_without_retries(run_policy, sample):
    calls = []

    def measure(funcs, *args):
        calls.append(1)
        return [sample(count=30) for _ in funcs]

    assert run_policy(measure, dict(quality_check=True, max_retries=3)) == [0.01, 0.01]
    assert calls == [1]


def test_profile_sample_filtering_and_warmup(backend, tmp_path):
    with (tmp_path / "kernel_details.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Name", "Type", "Start Time(us)", "Duration(us)"])
        writer.writerows([["hook", "Other", 0, 1], ["flush", "ReduceSum", 1, 1], ["a", "Kernel", 2, 99],
                          ["a", "Kernel", 3, 10], ["a", "Kernel", 4, 11], ["b", "Kernel", 5, 99],
                          ["b", "Kernel", 6, 20], ["b", "Kernel", 7, 21]])
    samples = backend.testing._read_profile_samples(tmp_path, ["a", "b"], 1, 2, True)
    assert [durations.tolist() for _, durations in samples] == [[10, 11], [20, 21]]
    with pytest.raises(backend.policy.ProfilerAcquisitionError, match="Expected"):
        backend.testing._read_profile_samples(tmp_path, ["a", "b"], 1, 3, True)


@pytest.mark.parametrize("source", ["argument", "environment"])
def test_warmup_override_is_used_in_cv_time_estimate(backend, monkeypatch, source, make_tuner):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    tuner = make_tuner(dict(warmup=0) if source == "argument" else None)
    if source == "environment":
        monkeypatch.setenv(backend.policy.ENV_PREFIX + "WARMUP", "0")
    tuner.print_autotuning = False
    tuner._rough_bench_once = lambda fn: 10000
    profiled = []

    def profile(funcs, **kwargs):
        profiled.append((len(funcs), kwargs["warmup"], kwargs["active"]))
        return [1] * len(funcs)

    monkeypatch.setattr(backend.testing, "do_bench_npu", profile)
    configs = [Config({}) for _ in range(12)]
    assert len(tuner._batch_bench(configs=configs)) == 12
    assert profiled == [(12, 0, 1)]


def test_default_cv_calculation_preserves_counts_and_time_limit_pruning(backend, monkeypatch, make_tuner):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    tuner = make_tuner()
    tuner.print_autotuning = False
    tuner._rough_bench_once = lambda fn: 10000
    profiled = []

    def profile(funcs, **kwargs):
        profiled.append((len(funcs), kwargs["warmup"], kwargs["active"]))
        return [1] * len(funcs)

    monkeypatch.setattr(backend.testing, "do_bench_npu", profile)
    assert len(tuner._batch_bench(configs=[Config({}) for _ in range(12)])) == 10
    assert profiled == [(10, 1, 1)]


def test_pre_hook_composition_order_reset_only_and_restoration(make_tuner):
    tuner = make_tuner()
    events = []
    original = lambda args, reset_only=False: events.append("reset" if reset_only else "user")
    tuner.pre_hook = original
    tuner.fn.run = lambda *args, **kwargs: events.append("kernel")
    config = Config({})
    config.pre_hook = lambda args: events.append("config")
    call = tuner._make_kernel_call(config=config)
    assert str(inspect.signature(call)) == "(warmup)"
    with pytest.raises(RuntimeError, match="failure"):
        with tuner._npu_cache_pre_hook(lambda: events.append("evict")):
            call(warmup=False)
            tuner.pre_hook({}, reset_only=True)
            raise RuntimeError("failure")
    assert events == ["config", "user", "evict", "kernel", "reset"]
    assert tuner.pre_hook is original


def test_cache_keys_track_effective_options_and_preserve_default_key(backend, monkeypatch, make_tuner):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    tuner = make_tuner()
    tuner.keys = []
    tuner.arg_names = ["x"]

    tensor = torch.empty(1)
    cold_key = tuner.generate_key_and_configs(tensor)
    assert cold_key == ("torch.float32", )
    monkeypatch.setenv(backend.policy.ENV_PREFIX + "CACHE_MODE", "hot")
    hot_key = tuner.generate_key_and_configs(tensor)
    assert hot_key != cold_key
    tuner.npu_bench_options = dict(cache_mode="cold", quality_check=True)
    assert tuner.generate_key_and_configs(tensor) not in (cold_key, hot_key)
    tuner.npu_bench_options = dict(active=30)
    explicit_count_key = tuner.generate_key_and_configs(tensor)
    tuner.npu_bench_options = dict(active=30, filter_slow_configs=True)
    filtered_key = tuner.generate_key_and_configs(tensor)
    assert filtered_key != explicit_count_key
    tuner.npu_bench_options = dict(active=30, filter_slow_configs=True, measure_budget_ms=5)
    assert tuner.generate_key_and_configs(tensor) != filtered_key


@pytest.mark.parametrize("failure_stage", ["eviction", "kernel"])
def test_failed_measurement_runs_cleanup_and_restores_hook(failure_stage, make_tuner):
    tuner = make_tuner()
    cleanup = []
    original = tuner.pre_hook
    error = RuntimeError(failure_stage)

    def fail():
        raise error

    tuner.post_hook = lambda args, exception: cleanup.append(exception)
    tuner.fn.run = lambda *args, **kwargs: fail() if failure_stage == "kernel" else None
    call = tuner._make_kernel_call(config=Config({}))
    with pytest.raises(RuntimeError, match=failure_stage):
        with tuner._npu_cache_pre_hook(fail if failure_stage == "eviction" else lambda: None):
            call(warmup=False)
    assert cleanup == [error] and tuner.pre_hook is original


@pytest.mark.parametrize("mode", ["default", "cold", "hot"])
def test_profiler_preserves_default_order_and_composes_explicit_cold(backend, monkeypatch, tmp_path, mode, make_tuner):
    events = []
    tuner = make_tuner()
    tuner.pre_hook = original = lambda args, reset_only=False: events.append("user")
    tuner.fn.run = lambda *args, **kwargs: events.append("kernel")
    config = Config({})
    config.pre_hook = lambda args: events.append("config")
    fn = functools.partial(tuner._make_kernel_call(config=config), warmup=False)

    class Buffer:

        def float(self):
            return self

        def sum(self):
            events.append("evict")

    @contextmanager
    def profile(**kwargs):
        events.append("profile enter")
        yield
        events.append("profile exit")

    monkeypatch.setattr(torch_npu.profiler, "profile", profile)
    monkeypatch.setattr(torch_npu.profiler, "tensorboard_trace_handler", lambda path: None)
    monkeypatch.setattr(driver.active, "get_empty_cache_for_benchmark", lambda: Buffer())
    monkeypatch.setattr(backend.testing, "_collect_prof_result", lambda *args, **kwargs: 2)
    kwargs = {} if mode == "default" else dict(npu_bench_options=dict(
        cache_mode=mode), _pre_hook_scope=tuner._npu_cache_pre_hook)
    assert backend.testing.do_bench_npu(fn, warmup=0, active=2, clear_l2_cache=True, prof_dir=str(tmp_path / "trace"),
                                        **kwargs) == 2
    measured = events[events.index("profile enter") + 1:events.index("profile exit")]
    iteration = (["evict", "config", "user", "kernel"] if mode == "default" else
                 ["config", "user", "evict", "kernel"] if mode == "cold" else ["config", "user", "kernel"])
    assert measured == iteration * 2
    assert tuner.pre_hook is original
    if mode == "hot":
        assert "evict" not in events


def test_slow_config_filter_rechecks_candidates_and_preserves_result_alignment(backend, monkeypatch, run_policy,
                                                                               sample):
    clock, calls = [0.0], [0, 0]
    monkeypatch.setattr(backend.policy.time, "perf_counter", lambda: clock[0])

    def candidate(index, duration):

        def call():
            calls[index] += 1
            clock[0] += duration

        return call

    measured = []

    def measure(funcs, names, *args):
        measured.append(names)
        return [sample()]

    costs = run_policy(measure, dict(filter_slow_configs=True, slow_config_runs=2, slow_config_recheck_runs=3),
                       funcs=[candidate(0, 1), candidate(1, 8)])
    assert costs == [0.01, float("inf")]
    assert measured == [["a"]] and calls == [3, 6]


def test_disabled_slow_filter_is_not_called(backend, monkeypatch, run_policy, sample):
    monkeypatch.setattr(backend.policy, "filter_slow_configs", lambda *args: pytest.fail("filter unexpectedly enabled"))
    assert run_policy(lambda *args: [sample(), sample()], dict(quality_check=True)) == [0.01, 0.01]


@pytest.mark.parametrize("explicit_counts", [False, True])
def test_slow_filter_operates_on_candidates_remaining_after_cv_estimate(backend, monkeypatch, explicit_counts,
                                                                        make_tuner, sample):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    options = dict(filter_slow_configs=True)
    if explicit_counts:
        options["measure_budget_ms"] = 5
    tuner = make_tuner(options)
    tuner.print_autotuning = False
    tuner._rough_bench_once = lambda fn: 10000
    configs = [Config({}) for _ in range(12)]
    filtered, profiled = [], []

    def filter_candidates(funcs, synchronize, options):
        filtered.append(len(funcs))
        return [1, len(funcs) - 1]

    def profile(funcs, warmup, active, *args, **kwargs):
        profiled.append((len(funcs), warmup, active))
        return [sample(1000, count=active) for _ in funcs]

    monkeypatch.setattr(backend.policy, "filter_slow_configs", filter_candidates)
    monkeypatch.setattr(backend.testing, "_profile_npu", profile)
    costs = tuner._batch_bench(configs=configs)
    assert filtered == ([12] if explicit_counts else [10])
    assert profiled == ([(2, 0, 10), (2, 5, 30)] if explicit_counts else [(2, 1, 1)])
    kept = [config for config, cost in costs.items() if np.isfinite(cost)]
    assert kept == [configs[1], configs[11 if explicit_counts else 9]]


INACTIVE_OPTIONS = dict(verbose=True, calibration_runs=27, slow_config_runs=31, slow_config_recheck_runs=53,
                        slow_config_factor=7)


def test_verbose_only_changes_profiler_output(backend, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(backend.testing, "_profile_npu", lambda *a, **k: calls.append((a, k)) or 7)
    fn = lambda: None
    assert backend.testing.do_bench_npu(fn) == 7
    with pytest.warns(RuntimeWarning) as diagnostics:
        assert backend.testing.do_bench_npu(fn, npu_bench_options=INACTIVE_OPTIONS) == 7
    assert len(diagnostics) == 2
    assert calls[0] == calls[1]
    assert "npu benchmark:" in capsys.readouterr().out


@pytest.mark.parametrize("warnings_as_errors", [False, True])
def test_eviction_and_post_hook_double_failure_preserves_eviction(warnings_as_errors, make_tuner):
    tuner = make_tuner()
    original = tuner.pre_hook
    primary, secondary = RuntimeError("eviction"), ValueError("restore")
    cleanup = []

    def fail():
        raise primary

    def post_hook(args, exception):
        cleanup.append(exception)
        raise secondary

    tuner.post_hook = post_hook
    with warnings.catch_warnings(record=True) as diagnostics:
        warnings.simplefilter("error" if warnings_as_errors else "always")
        with pytest.raises(RuntimeError) as caught:
            with tuner._npu_cache_pre_hook(fail):
                tuner._make_kernel_call(config=Config({}))(warmup=False)
    assert caught.value is primary
    assert cleanup == [primary] and tuner.pre_hook is original
    if not warnings_as_errors:
        assert "restore" in str(diagnostics[0].message)


@pytest.mark.parametrize("stage", ["kernel", "collection", "success"])
@pytest.mark.parametrize("warnings_as_errors", [False, True])
def test_profile_and_removal_double_failure(backend, monkeypatch, tmp_path, fake_profiler, stage, warnings_as_errors):
    primary, secondary = ValueError("primary"), OSError("remove profile")
    calls = []

    def kernel():
        calls.append(True)
        if stage == "kernel" and len(calls) > 1:
            raise primary

    def collect(*args, **kwargs):
        if stage == "collection":
            raise primary
        return 7

    def remove(*args):
        raise secondary

    monkeypatch.setattr(backend.testing, "_collect_prof_result", collect)
    monkeypatch.setattr(backend.testing, "_rm_dic", remove)
    with warnings.catch_warnings(record=True) as diagnostics:
        warnings.simplefilter("error" if warnings_as_errors else "always")
        with pytest.raises(Exception) as caught:
            backend.testing._profile_npu(kernel, warmup=0, active=1, prof_dir=str(tmp_path))
    assert caught.value is (secondary if stage == "success" else primary)
    if stage != "success" and not warnings_as_errors:
        assert "remove profile" in str(diagnostics[0].message)


@pytest.mark.parametrize("failure", [False, True])
def test_policy_profile_removal_preserves_primary_or_propagates(backend, monkeypatch, failure, run_policy, sample):
    primary, secondary = ValueError("execution"), OSError("remove")

    def measure(*args):
        if failure:
            raise primary
        return [sample()]

    def remove(*args):
        raise secondary

    monkeypatch.setattr(backend.policy.shutil, "rmtree", remove)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(Exception) as caught:
            run_policy(measure, dict(quality_check=True), funcs=[lambda: None])
    assert caught.value is (primary if failure else secondary)


def test_budget_retains_all_10000_launches_and_samples(backend, monkeypatch, run_policy):
    calls, launches, checked = [], [], []
    values = np.random.default_rng(42).normal(1000, 2, 10000)

    def measure(funcs, names, warmup, active, directory):
        calls.append((warmup, active))
        for _ in range(active):
            funcs[0]()
        return [(np.arange(active, dtype=float), np.full(active, 1000.) if active == 10 else values)]

    original = backend.quality.evaluate_quality

    def quality(times, durations):
        checked.append(len(durations))
        return original(times, durations)

    monkeypatch.setattr(backend.quality, "evaluate_quality", quality)
    result = run_policy(measure, dict(measure_budget_ms=10000, quality_check=True),
                        funcs=[lambda: launches.append(True)])
    assert calls == [(0, 10), (5, 10000)]
    assert len(launches) == 10010 and checked == [10000]
    assert result == [float(np.mean(values)) / 1000]


@pytest.mark.parametrize("source,policy,count,warning_fields", [
    ("argument", {}, 1, ["calibration_runs", "slow_config_runs"]),
    ("environment", {}, 2, ["calibration_runs", "slow_config_runs"]),
    ("argument", {"active": 30}, 2, ["calibration_runs", "slow_config_runs"]),
    ("environment", {"quality_check": True}, 2, ["calibration_runs", "slow_config_runs"]),
    ("argument", {"measure_budget_ms": 5}, 2, ["slow_config_runs"]),
    ("environment", {"filter_slow_configs": True}, 2, ["calibration_runs"]),
    ("argument", {"measure_budget_ms": 5, "filter_slow_configs": True}, 2, []),
])
def test_inactive_fields_preserve_priority_route_and_cache(backend, monkeypatch, make_tuner, source, policy, count,
                                                           warning_fields):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    fields = dict(INACTIVE_OPTIONS)
    # Fields are inactive only when their corresponding policy is disabled.
    if policy.get("measure_budget_ms"):
        fields.pop("calibration_runs")
    if policy.get("filter_slow_configs"):
        for name in ("slow_config_runs", "slow_config_recheck_runs", "slow_config_factor"):
            fields.pop(name)
    baseline = backend.policy.resolve_options(policy)
    if source == "environment":
        for name, value in fields.items():
            monkeypatch.setenv(backend.policy.ENV_PREFIX + name.upper(), str(value))
        arguments = policy
    else:
        arguments = dict(fields, **policy)
    tuner = make_tuner(arguments)
    context = pytest.warns(RuntimeWarning) if warning_fields else nullcontext()
    with context as diagnostics:
        resolved = tuner._get_npu_benchmark_options()
    if warning_fields:
        assert len(diagnostics) == len(warning_fields)
        assert all(field in str(item.message) for field, item in zip(warning_fields, diagnostics))
    assert resolved.is_default == baseline.is_default
    assert resolved.cache_key() == baseline.cache_key()
    tensor = torch.empty(1)
    key = tuner.generate_key_and_configs(tensor)
    assert key == (("torch.float32", ) if baseline.is_default else ("torch.float32", baseline.cache_key()))
    prunes, profiles = [], []

    def prune(funcs):
        prunes.append(True)
        tuner.cv_warmup, tuner.cv_repeat = 8, 50
        return funcs

    def profile(funcs, **kwargs):
        profiles.append((kwargs["warmup"], kwargs["active"]))
        return [7] * len(funcs)

    tuner._prune_by_time_limit = prune
    tuner.do_bench = lambda *a, **k: 7
    monkeypatch.setattr(backend.testing, "do_bench_npu", profile)
    assert list(tuner._batch_bench(configs=[Config({}) for _ in range(count)]).values()) == [7] * count
    existing_counts = count > 1 and baseline.active is None and baseline.measure_budget_ms is None
    assert prunes == ([True] if existing_counts else [])
    assert profiles == ([] if count == 1 else [(8, 50)] if existing_counts else [(5, 30)])


def test_enabled_fields_participate_in_cache_key(backend):
    resolve = backend.policy.resolve_options
    assert resolve({"measure_budget_ms": 5}).cache_key() != resolve({"measure_budget_ms": 5, "calibration_runs":
                                                                     27}).cache_key()
    assert resolve({"filter_slow_configs":
                    True}).cache_key() != resolve({"filter_slow_configs": True, "slow_config_runs": 31}).cache_key()


@pytest.mark.parametrize("source,counts,filter_slow,count", [
    ("argument", {}, False, 2),
    ("environment", {}, True, 2),
    ("argument", {"active": 30}, False, 12),
    ("environment", {"active": 17}, True, 12),
    ("argument", {"measure_budget_ms": 5}, True, 12),
    ("environment", {"measure_budget_ms": 5}, False, 12),
    ("argument", {}, False, 1),
    ("argument", {"cache_mode": "hot"}, False, 1),
])
def test_resolved_options_preserve_counts_route_and_inactive_diagnostics(backend, monkeypatch, make_tuner, sample,
                                                                         source, counts, filter_slow, count):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    # Explicit defaults must also disable conflicting environment policies.
    monkeypatch.setenv(backend.policy.ENV_PREFIX + "QUALITY_CHECK", "true")
    monkeypatch.setenv(backend.policy.ENV_PREFIX + "FILTER_SLOW_CONFIGS", str(not filter_slow))
    arguments = dict(quality_check=False, filter_slow_configs=filter_slow)
    if source == "argument":
        arguments.update(counts)
    else:
        for name, value in counts.items():
            monkeypatch.setenv(backend.policy.ENV_PREFIX + name.upper(), str(value))
    tuner = make_tuner(arguments)
    prunes, profiles, filters = [], [], []

    def prune(funcs):
        prunes.append(True)
        tuner.cv_warmup, tuner.cv_repeat = 8, 50
        return funcs

    def filter_candidates(funcs, synchronize, options):
        filters.append(len(funcs))
        return list(range(len(funcs)))

    def profile(funcs, warmup, active, *args, **kwargs):
        profiles.append((len(funcs), warmup, active))
        if kwargs.get("_return_samples"):
            return [sample(1000, count=active) for _ in funcs]
        return [1] * len(funcs) if len(funcs) > 1 else 1

    tuner._prune_by_time_limit = prune
    tuner.do_bench = lambda *a, **k: 1
    monkeypatch.setattr(backend.policy, "filter_slow_configs", filter_candidates)
    monkeypatch.setattr(backend.quality, "evaluate_quality", lambda *a: pytest.fail("quality unexpectedly enabled"))
    monkeypatch.setattr(backend.testing, "_profile_npu", profile)
    with warnings.catch_warnings(record=True) as diagnostics:
        warnings.simplefilter("always")
        assert list(tuner._batch_bench(configs=[Config({}) for _ in range(count)]).values()) == [1] * count
    assert not diagnostics  # _ResolvedOptions must not diagnose synthesized defaults.
    existing_counts = count > 1 and "active" not in counts and "measure_budget_ms" not in counts
    assert prunes == ([True] if existing_counts else [])
    assert filters == ([count] if filter_slow else [])
    expected = ([(count, 0, 10), (count, 5, 30)] if "measure_budget_ms" in counts else
                [(count, 5, counts.get("active", 30))] if counts else [(count, 8, 50)] if count > 1 else [])
    assert profiles == expected
