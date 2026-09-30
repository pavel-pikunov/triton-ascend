"""Hardware-independent policy tests. Run with --confcutdir=autotune_ut."""

import ast
import builtins
import csv
import functools
import importlib
import inspect
import os
import sys
import types
import warnings
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


BACKEND = Path(__file__).resolve().parents[2] / "backend"
PACKAGE = "_npu_policy_test_backend"


@pytest.fixture
def backend(monkeypatch):
    # Load the real policy and profiler code without requiring compiled Triton
    # bindings or torch_npu on the machine running these policy tests.
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(BACKEND)]
    monkeypatch.setitem(sys.modules, PACKAGE, package)
    policy = importlib.import_module(f"{PACKAGE}._npu_benchmark")
    for key in tuple(os.environ):
        if key.startswith(policy.ENV_PREFIX):
            monkeypatch.delenv(key)
    testing = types.ModuleType(f"{PACKAGE}.testing")
    testing.__package__ = PACKAGE
    tree = ast.parse((BACKEND / "testing.py").read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and any(alias.name == "triton.runtime" for alias in node.names)
    )]
    exec(compile(tree, str(BACKEND / "testing.py"), "exec"), testing.__dict__)
    monkeypatch.setitem(sys.modules, testing.__name__, testing)

    source = ast.parse((BACKEND / "runtime" / "autotuner.py").read_text(encoding="utf-8"))
    original = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "AutoTilingTuner")
    names = {"_get_npu_benchmark_options", "_npu_cache_pre_hook", "_make_kernel_call", "_batch_bench",
             "_resolve_target_kernel_name", "generate_key_and_configs"}
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for method in methods:
        method.body = [node for node in method.body if not (
            isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("triton.")
        )]
    cls = ast.ClassDef(name="Tuner", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = dict(__package__=PACKAGE + ".runtime", contextmanager=contextmanager, functools=functools,
                     os=os, builtins=builtins, warnings=warnings, asdict=asdict,
                     get_byte_per_numel=lambda dtype: 4, CompileTimeAssertionFailure=type("CompileError", (Exception,), {}),
                     MLIRCompilationError=type("MLIRError", (Exception,), {}),
                     OutOfResources=type("ResourceError", (Exception,), {}))
    exec(compile(module, str(BACKEND / "runtime" / "autotuner.py"), "exec"), namespace)
    return SimpleNamespace(policy=policy, testing=testing, Tuner=namespace["Tuner"])


def test_argument_overrides_environment_per_field(backend, monkeypatch):
    policy = backend.policy
    monkeypatch.setenv(policy.ENV_PREFIX + "CACHE_MODE", "hot")
    monkeypatch.setenv(policy.ENV_PREFIX + "QUALITY_CHECK", "true")
    monkeypatch.setenv(policy.ENV_PREFIX + "MAX_RETRIES", "invalid but overridden")
    monkeypatch.setenv(policy.ENV_PREFIX + "ACTIVE", "71")
    options = policy.resolve_options(dict(cache_mode="cold", quality_check=False, max_retries=0))
    assert options.cache_mode == "cold"
    assert options.quality_check is False and options.max_retries == 0
    assert options.active == 71 and options.measure_budget_ms is None
    assert options.pruning == "existing"


@pytest.mark.parametrize("options", [dict(quality_check=1), dict(active=True), dict(warmup=-1),
                                    dict(max_retries=-1), dict(measure_budget_ms=float("nan")),
                                    dict(pruning="both"), dict(cache_mode="L1"), dict(unknown=1),
                                    dict(quality_check=True, active=1), dict(prune_factor=0.5)])
def test_invalid_options(backend, options):
    with pytest.raises(ValueError):
        backend.policy.resolve_options(options)


def sample(duration=10, count=4):
    return np.arange(count, dtype=float) + 1, np.full(count, duration, dtype=float)


def run_policy(backend, tmp_path, measure, options, funcs=None):
    funcs = funcs or [lambda: None, lambda: None]
    return backend.policy.benchmark_with_options(
        measure, funcs, ["a", "b"][:len(funcs)], backend.policy.resolve_options(options),
        warmup=5, active=30, prof_root=tmp_path, synchronize=lambda: None,
    )


def test_only_failed_candidates_are_remeasured(backend, tmp_path, monkeypatch):
    calls = []

    def measure(funcs, names, warmup, active, directory):
        calls.append((names, warmup, active))
        return [sample(10), sample(20)] if len(calls) == 1 else [sample(15)]

    monkeypatch.setattr(backend.policy, "evaluate_quality",
                        lambda t, d: ({}, ["bad"] if d[0] == 20 else []))
    result = run_policy(backend, tmp_path, measure, dict(quality_check=True, max_retries=2))
    assert result == [0.01, 0.015]
    assert calls == [(["a", "b"], 5, 30), (["b"], 5, 30)]
    assert list(tmp_path.iterdir()) == []


def test_quality_check_does_not_enable_budget_or_retries(backend, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(backend.policy, "evaluate_quality", lambda t, d: ({}, ["bad"]))

    def measure(funcs, names, warmup, active, directory):
        calls.append((warmup, active))
        return [sample() for _ in funcs]

    with pytest.warns(RuntimeWarning, match="failed quality checks"):
        run_policy(backend, tmp_path, measure, dict(quality_check=True))
    assert calls == [(5, 30)]


def test_budget_calibrates_without_enabling_quality(backend, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(backend.policy, "evaluate_quality", lambda *args: pytest.fail("quality unexpectedly enabled"))

    def measure(funcs, names, warmup, active, directory):
        calls.append((warmup, active))
        return [sample() for _ in funcs]

    assert run_policy(backend, tmp_path, measure, dict(measure_budget_ms=1)) == [0.01, 0.01]
    assert calls == [(0, 10), (5, 100)]


def test_acquisition_failure_is_retried(backend, tmp_path):
    calls = []

    def measure(*args):
        calls.append(1)
        if len(calls) == 1:
            raise backend.policy.ProfilerAcquisitionError("missing rows")
        return [sample(), sample()]

    assert run_policy(backend, tmp_path, measure, dict(max_retries=1)) == [0.01, 0.01]
    assert len(calls) == 2 and list(tmp_path.iterdir()) == []


def test_kernel_errors_are_never_retried(backend, tmp_path):
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("kernel failure")

    def measure(funcs, *args):
        funcs[0]()

    with pytest.raises(RuntimeError, match="kernel failure"):
        run_policy(backend, tmp_path, measure, dict(max_retries=4), funcs=[broken])
    assert calls == [1] and list(tmp_path.iterdir()) == []


def test_best_measurement_survives_exhausted_quality_retries(backend, tmp_path, monkeypatch):
    durations = iter([12, 10, 14])
    monkeypatch.setattr(backend.policy, "evaluate_quality", lambda t, d: ({}, ["bad"]))
    with pytest.warns(RuntimeWarning, match="exhausted"):
        result = run_policy(backend, tmp_path, lambda *args: [sample(next(durations))],
                            dict(quality_check=True, max_retries=2), funcs=[lambda: None])
    assert result == [0.01]


def test_real_quality_metrics_accept_stable_samples_without_retries(backend, tmp_path):
    calls = []

    def measure(funcs, *args):
        calls.append(1)
        return [sample(count=30) for _ in funcs]

    assert run_policy(backend, tmp_path, measure, dict(quality_check=True, max_retries=3)) == [0.01, 0.01]
    assert calls == [1]


def test_profile_sample_filtering_and_warmup(backend, tmp_path):
    with (tmp_path / "kernel_details.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Name", "Type", "Start Time(us)", "Duration(us)"])
        writer.writerows([["hook", "Other", 0, 1], ["flush", "ReduceSum", 1, 1],
                          ["a", "Kernel", 2, 99], ["a", "Kernel", 3, 10], ["a", "Kernel", 4, 11],
                          ["b", "Kernel", 5, 99], ["b", "Kernel", 6, 20], ["b", "Kernel", 7, 21]])
    samples = backend.testing._read_profile_samples(tmp_path, ["a", "b"], 1, 2, True)
    assert [durations.tolist() for _, durations in samples] == [[10, 11], [20, 21]]
    with pytest.raises(backend.policy.ProfilerAcquisitionError, match="Expected"):
        backend.testing._read_profile_samples(tmp_path, ["a", "b"], 1, 3, True)


def make_tuner(backend, options=None):
    tuner = backend.Tuner()
    tuner.npu_bench_options = options
    tuner.user_defined_do_bench = False
    tuner.compile_parallel = False
    tuner.nargs = {}
    tuner.parser_mode = "cube"
    tuner.cv_parse_result = object()
    tuner.fn = SimpleNamespace(run=lambda *args, **kwargs: SimpleNamespace(packed_metadata={"kernel_name": "kernel"}))
    tuner.pre_hook = lambda args, reset_only=False: None
    tuner.post_hook = lambda args, exception: None
    tuner.do_bench = lambda *args, **kwargs: pytest.fail("unexpected event benchmark")
    return tuner


class Config:
    pre_hook = None
    kwargs = {}

    def all_kwargs(self):
        return {}


@pytest.mark.parametrize("pruning", ["existing", "fast"])
def test_pruning_strategies_are_exclusive(backend, monkeypatch, pruning):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    tuner = make_tuner(backend, dict(pruning=pruning))
    prunes, profiled = [], []

    def existing_prune(funcs):
        prunes.append("existing")
        tuner.cv_warmup, tuner.cv_repeat = 8, 50
        return funcs

    tuner._prune_by_time_limit = existing_prune

    def profile(funcs, **kwargs):
        profiled.append(kwargs)
        return [1, 2]

    monkeypatch.setattr(backend.testing, "do_bench_npu", profile)
    assert list(tuner._batch_bench(configs=[Config(), Config()]).values()) == [1, 2]
    assert prunes == (["existing"] if pruning == "existing" else [])
    kwargs = profiled[0]
    if pruning == "fast":
        assert kwargs["npu_bench_options"]["pruning"] == "fast"
        assert kwargs["warmup"] == 5 and kwargs["active"] == 30
    else:
        assert backend.policy.resolve_options(kwargs["npu_bench_options"]).is_default
        assert kwargs["warmup"] == 8 and kwargs["active"] == 50


def test_legacy_single_candidate_uses_existing_benchmarker(backend, monkeypatch):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    tuner = make_tuner(backend)
    tuner.do_bench = lambda fn, quantiles: 7
    assert list(tuner._batch_bench(configs=[Config()]).values()) == [7]


def test_pre_hook_composition_order_reset_only_and_restoration(backend):
    tuner = make_tuner(backend)
    events = []
    original = lambda args, reset_only=False: events.append("reset" if reset_only else "user")
    tuner.pre_hook = original
    tuner.fn.run = lambda *args, **kwargs: events.append("kernel")
    config = Config()
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


def test_cache_keys_track_effective_options_and_preserve_legacy_key(backend, monkeypatch):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    tuner = make_tuner(backend)
    tuner.keys = []
    tuner.arg_names = ["x"]

    class Cached(dict):
        def __contains__(self, key):
            return True

    tuner.cache = Cached()
    tensor = SimpleNamespace(dtype="float32")
    cold_key = tuner.generate_key_and_configs(tensor)
    assert cold_key == ("float32",)
    monkeypatch.setenv(backend.policy.ENV_PREFIX + "CACHE_MODE", "hot")
    hot_key = tuner.generate_key_and_configs(tensor)
    assert hot_key != cold_key
    tuner.npu_bench_options = dict(cache_mode="cold", quality_check=True)
    assert tuner.generate_key_and_configs(tensor) not in (cold_key, hot_key)


@pytest.mark.parametrize("failure_stage", ["eviction", "kernel"])
def test_failed_measurement_runs_cleanup_and_restores_hook(backend, failure_stage):
    tuner = make_tuner(backend)
    cleanup = []
    original = tuner.pre_hook
    error = RuntimeError(failure_stage)

    def fail():
        raise error

    tuner.post_hook = lambda args, exception: cleanup.append(exception)
    tuner.fn.run = lambda *args, **kwargs: fail() if failure_stage == "kernel" else None
    call = tuner._make_kernel_call(config=Config())
    with pytest.raises(RuntimeError, match=failure_stage):
        with tuner._npu_cache_pre_hook(fail if failure_stage == "eviction" else lambda: None):
            call(warmup=False)
    assert cleanup == [error] and tuner.pre_hook is original


def test_other_methods_and_user_benchmarkers_ignore_npu_options(backend, monkeypatch):
    tuner = make_tuner(backend, dict(quality_check=True))
    monkeypatch.setenv("TRITON_BENCH_METHOD", "default")
    assert tuner._get_npu_benchmark_options() is None
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    tuner.user_defined_do_bench = True
    assert tuner._get_npu_benchmark_options() is None


@pytest.mark.parametrize("mode", ["legacy", "cold", "hot"])
def test_profiler_preserves_legacy_order_and_composes_explicit_cold(backend, monkeypatch, tmp_path, mode):
    events = []
    tuner = make_tuner(backend)
    tuner.pre_hook = original = lambda args, reset_only=False: events.append("user")
    tuner.fn.run = lambda *args, **kwargs: events.append("kernel")
    config = Config()
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

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=SimpleNamespace(
        _ExperimentalConfig=lambda **kw: None, AiCMetrics=SimpleNamespace(PipeUtilization=1),
        ProfilerLevel=SimpleNamespace(Level1=1), ProfilerActivity=SimpleNamespace(NPU=1),
        tensorboard_trace_handler=lambda path: None, profile=profile,
    )))
    backend.testing.runtime = SimpleNamespace(driver=SimpleNamespace(active=SimpleNamespace(
        get_empty_cache_for_benchmark=lambda: Buffer(),
    )))
    monkeypatch.setattr(backend.testing, "_collect_prof_result", lambda *args, **kwargs: 2)
    kwargs = {} if mode == "legacy" else dict(npu_bench_options=dict(cache_mode=mode),
                                              _pre_hook_scope=tuner._npu_cache_pre_hook)
    assert backend.testing.do_bench_npu(fn, warmup=0, active=2, clear_l2_cache=True,
                                       prof_dir=str(tmp_path / "trace"), **kwargs) == 2
    measured = events[events.index("profile enter") + 1:events.index("profile exit")]
    iteration = (["evict", "config", "user", "kernel"] if mode == "legacy" else
                 ["config", "user", "evict", "kernel"] if mode == "cold" else ["config", "user", "kernel"])
    assert measured == iteration * 2
    assert tuner.pre_hook is original
    if mode == "hot":
        assert "evict" not in events


def test_explicit_defaults_do_not_reenable_environment_policies(backend, monkeypatch):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    monkeypatch.setenv(backend.policy.ENV_PREFIX + "QUALITY_CHECK", "true")
    monkeypatch.setenv(backend.policy.ENV_PREFIX + "PRUNING", "fast")
    tuner = make_tuner(backend, dict(quality_check=False, pruning="existing"))

    def existing_prune(funcs):
        tuner.cv_warmup, tuner.cv_repeat = 5, 30
        return funcs

    tuner._prune_by_time_limit = existing_prune
    calls = []

    def profile(*args, **kwargs):
        calls.append(kwargs)
        return [1, 2]

    monkeypatch.setattr(backend.testing, "_profile_npu", profile)
    tuner._batch_bench(configs=[Config(), Config()])
    assert len(calls) == 1 and "_return_samples" not in calls[0]


def test_fast_prune_rechecks_slow_candidates_and_preserves_result_alignment(backend, monkeypatch, tmp_path):
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

    costs = run_policy(backend, tmp_path, measure,
                       dict(pruning="fast", prune_runs=2, prune_recheck_runs=3),
                       funcs=[candidate(0, 1), candidate(1, 8)])
    assert costs == [0.01, float("inf")]
    assert measured == [["a"]] and calls == [3, 6]
