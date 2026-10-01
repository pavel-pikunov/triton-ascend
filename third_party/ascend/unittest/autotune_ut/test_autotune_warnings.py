"""Config, measurement, fallback and CSV diagnostics on real tuners."""

import ast
import builtins
import warnings
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import torch
import triton
from triton import Config
from triton.backends.ascend.runtime import autotuner as generators
from triton.backends.ascend.runtime.dsl_analysis import kernel_classifier as classifier


def test_constructor_initializes_warnings_before_axis_analysis(monkeypatch, make_tuner):

    def fail(self):
        raise RuntimeError("constexpr analysis failed")

    monkeypatch.setattr(generators.AutoTilingTuner, "_get_constexpr_candidates", fail)
    with pytest.warns(RuntimeWarning, match="constexpr analysis failed"):
        tuner = make_tuner(hints={"axes": {"x": "x"}})
    assert "constexpr_axis_analysis" in tuner._warning_reasons


@pytest.mark.parametrize("kind", ["cube", "cv", "vector"])
def test_generator_warnings_preserve_configs_and_validation(kind):
    generate = getattr(generators, f"get_autotune_{kind}_config")
    baseline = generate(num_stages=[1])
    with pytest.warns(RuntimeWarning, match="multibuffer") as captured:
        actual = generate(num_stages=[1], multibuffer=[False])
    assert len(captured) == 1
    assert [vars(cfg) for cfg in actual] == [vars(cfg) for cfg in baseline]
    with pytest.warns(RuntimeWarning, match="typo"):
        assert generate(typo=[1]) == []
    # Existing invalid-value checks still return no configs.
    assert generate(num_stages=[3]) == []


def test_max_generator_checks_once_before_expansion(monkeypatch, make_tuner, jit_kernel):
    config = Config({"BLOCK": 64}, num_warps=8, num_stages=1, pre_hook=lambda args: None)
    baseline = generators.get_max_configs(config, kernel_type="vector")
    with pytest.warns(RuntimeWarning, match="typo.*unit_flag") as captured:
        actual = generators.get_max_configs(config, kernel_type="vector", typo=[1], unit_flag=[False])
    assert len(captured) == 1
    assert [vars(cfg) for cfg in actual] == [vars(cfg) for cfg in baseline]
    with pytest.warns(RuntimeWarning, match="falling back to 'mixcv'"):
        fallback = generators.get_max_configs(config, kernel_type="unknown")
    assert [vars(cfg) for cfg in fallback] == [vars(cfg) for cfg in generators.get_max_configs(config)]
    with pytest.raises(ValueError, match="Invalid value"):
        generators.get_max_configs(config, kernel_type="vector", num_stages=[3])

    # Capture expanded configs at the existing decorator boundary.
    make_tuner()  # Install the isolated benchmarker for the real decorator.
    calls = []
    original = generators._expand_max_configs

    def expand(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(generators, "_expand_max_configs", expand)
    decorator = generators.max_autotune([config, config], [], kernel_type="unknown", typo=[1])
    assert not calls
    with pytest.warns(RuntimeWarning) as captured:
        actual = decorator(jit_kernel).configs
    assert len(captured) == 2 and len(calls) == 2
    assert [vars(cfg) for cfg in actual] == [vars(cfg) for cfg in fallback] * 2
    calls.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning, match="typo"):
            generators.max_autotune([config], [], typo=[1])(jit_kernel)
    assert not calls


@pytest.mark.parametrize("policy", [{}, {"verbose": True}, {"active": 30}, {"cache_mode": "hot"}])
def test_single_pruned_config_skips_measurements_and_keeps_execution(monkeypatch, policy, configure_run):
    configs = [Config({}), Config({})]
    tuner = configure_run(configs[1], configs, False)
    tuner.prune_configs = lambda kwargs: [configs[1]]
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    tuner.npu_bench_options = policy
    tuner._batch_bench = lambda *a, **k: pytest.fail("single config must not be measured")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert tuner.run(torch.empty(1)) == "result"
    assert len(captured) == (0 if tuner._npu_benchmark_options.is_default else 1)
    if captured:
        assert "skips measurements" in str(captured[0].message)
    assert tuner.best_config is configs[1] and tuner.cache == {}


def test_extra_winner_profile_warns_and_keeps_environment_route(backend, monkeypatch, make_tuner):
    tuner = make_tuner({"active": 99, "cache_mode": "cold"})
    tuner.auto_profile_dir = "winner"
    monkeypatch.setenv(backend.policy.ENV_PREFIX + "ACTIVE", "17")
    profiles = []

    def profile(fn, **kwargs):
        profiles.append((kwargs, backend.policy.resolve_options(kwargs.get("npu_bench_options"))))

    monkeypatch.setattr(backend.testing, "do_bench_npu", profile)
    with pytest.warns(RuntimeWarning, match="additional winner _profile") as captured:
        tuner._profile(config=Config({}))
    assert len(captured) == 1 and len(profiles) == 1
    assert all(kwargs == dict(prof_dir="winner", keep_res=True) and options.active == 17 and options.cache_mode is None
               for kwargs, options in profiles)
    tuner.npu_bench_options = None
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        tuner._profile(config=Config({}))
    assert not captured


@pytest.mark.parametrize("operation",
                         ["classification", "dot_sites", "tunable", "vector", "vector_v2", "ubtuner", "constexpr_axes"])
def test_fallback_failures_warn_without_debug_and_keep_results(monkeypatch, operation, make_tuner):
    tuner = make_tuner()
    tuner.print_autotuning = False
    tuner.hints = {}
    tuner._is_auto_kernel_hint = lambda: True
    error = RuntimeError("parser broke")

    def fail(*args, **kwargs):
        raise error

    if operation == "classification":
        tuner.fn.parse = fail
        monkeypatch.setattr(generators, "resolve_kernel_type", lambda hints, parsed: "vector")
        call, expected = tuner._resolve_kernel_type, "vector"
    elif operation == "dot_sites":
        tuner.fn.parse = lambda: ast.parse("def kernel(): pass")
        tuner._build_cv_parse_ast_context = lambda *a: (None, ast.parse(""), "kernel", "test")
        monkeypatch.setattr(generators, "resolve_kernel_type", lambda *a: "vector")
        monkeypatch.setattr(generators, "analyze_dot_site_mnk", fail)
        call, expected = tuner._resolve_kernel_type, "vector"
    elif operation == "tunable":
        tuner.fn.parse = lambda: ast.parse("")
        tuner.arg_names, tuner.split_params, tuner.tiling_params, tuner.explicit_tunable_params = [], {}, {}, []
        tuner._get_constexpr_candidates = lambda: ["BLOCK"]
        monkeypatch.setattr(generators, "analyze_signature_and_missing_tunable", fail)
        call = lambda: tuner._detect_missing_tunable_params({}, ["x", "BLOCK"])
        expected = ["BLOCK"]
    elif operation == "vector":
        call = lambda: tuner._run_vector_parser_with_fallback("reduction_axes", fail, [])
        expected = []
    elif operation == "vector_v2":
        tuner.enable_vv_parser_v2, tuner.parser_mode = True, "vector"
        tuner.vv_parse_result_v2 = tuner.vv_adapter_result_v2 = object()
        tuner.fn.parse = fail
        call, expected = tuner._autoparse_axis_info_v2_for_vector, None
    elif operation == "constexpr_axes":
        tuner.arg_names = ["x", "BLOCK"]
        tuner._get_constexpr_candidates = fail
        call, expected = tuner._get_runtime_arg_names_for_hints_axes, ["x", "BLOCK"]
    else:
        tuner.enable_ubtuner = True
        tuner.ubtuner = SimpleNamespace(get_best_config=fail)
        available = {Config({}): object()}
        original = available.copy()
        call = lambda: tuner._try_ubtuner(config=Config({}), excp=RuntimeError("UB overflow"), run_fns=available)
        expected = None
    with pytest.warns(RuntimeWarning, match="RuntimeError: parser broke"):
        assert call() == expected
    if operation == "vector_v2":
        assert tuner.vv_parse_result_v2 is None and tuner.vv_adapter_result_v2 is None
    if operation == "ubtuner":
        assert available == original
    if operation == "tunable":
        monkeypatch.setattr(generators, "analyze_signature_and_missing_tunable", lambda *a, **k:
                            (_ for _ in ()).throw(ValueError("invalid hint")))
        with pytest.raises(ValueError, match="invalid hint"):
            call()


def test_classifier_internal_failure_warns_and_keeps_vector_fallback(monkeypatch):

    def fail(tree):
        raise RuntimeError("AST walk broke")

    monkeypatch.setattr(classifier.ast, "walk", fail)
    with pytest.warns(RuntimeWarning, match="AST walk broke.*vector fallback"):
        assert classifier.classify_kernel_type_from_dsl(ast.parse("")) == "vector"
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert classifier.classify_kernel_type_from_dsl(None) == "vector"
    assert not captured


@pytest.mark.parametrize("warnings_as_errors", [False, True])
def test_parallel_compile_failure_restores_mode_before_warning(monkeypatch, warnings_as_errors, make_tuner):
    tuner = make_tuner()
    tuner.compile_parallel, tuner.parser_mode = True, "vector"
    tuner.do_bench = lambda *a, **k: 7
    active_mode = triton.runtime._async_compile.active_mode

    @contextmanager
    def mode(executor):
        active_mode.set("active")
        yield
        raise RuntimeError("async exit broke")

    @contextmanager
    def executor(**kwargs):
        yield object()

    monkeypatch.setattr(triton, "AsyncCompileMode", mode)
    monkeypatch.setattr(generators, "ThreadPoolExecutor", executor)
    monkeypatch.setenv("TRITON_BENCH_METHOD", "default")
    configs = [Config({}), Config({})]
    token = active_mode.set(None)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("error" if warnings_as_errors else "always")
            if warnings_as_errors:
                with pytest.raises(RuntimeWarning, match="async exit broke"):
                    tuner._batch_bench(configs=configs)
            else:
                assert tuner._batch_bench(configs=configs) == dict.fromkeys(configs, 7)
        assert active_mode.get() is None
        if not warnings_as_errors:
            assert len(captured) == 1 and "available candidates" in str(captured[0].message)
    finally:
        active_mode.reset(token)


@pytest.mark.parametrize("count", [1, 2])
def test_missing_csv_warns_and_preserves_inf_results(backend, tmp_path, count):
    with pytest.warns(RuntimeWarning, match="kernel_details.csv.*not found"):
        result = backend.testing._collect_prof_result(str(tmp_path), [lambda: None] * count, 0, 1)
    assert result == (float("inf") if count == 1 else [float("inf")] * count)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(RuntimeWarning, match="returning inf"):
            backend.testing._collect_prof_result(str(tmp_path), [lambda: None], 0, 1)


@pytest.mark.parametrize("costs", [[float("inf"), float("inf")], [float("nan"), float("nan")],
                                   [[float("inf")] * 3, [float("inf")] * 3], [1, float("inf")]])
def test_unusable_final_scores_warn_and_preserve_selection_and_cache(costs, configure_run):
    configs = [Config({}), Config({})]
    tuner = configure_run(configs[0], configs, False)
    tuner.prune_configs = lambda kwargs: configs
    tuner._batch_bench = lambda *a, **k: dict(zip(configs, costs))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        assert tuner.run(torch.empty(1)) == "result"
        assert tuner.run(torch.empty(1)) == "result"
    assert tuner.best_config is configs[0] and tuner.cache == {("torch.float32", ): configs[0]}
    assert len(captured) == (0 if costs[0] == 1 else 1)
    if captured:
        assert "without usable measurements" in str(captured[0].message)


def test_warning_reasons_are_per_instance_and_located_at_kernel(monkeypatch, make_tuner):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "default")
    tuners = [make_tuner(dict(active=30)) for _ in range(2)]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("default")
        for tuner in tuners:
            tuner._get_npu_benchmark_options()
            tuner._get_npu_benchmark_options()
        monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
        tuners[0].npu_bench_options = dict(calibration_runs=10)
        tuners[0]._get_npu_benchmark_options()
        tuners[0]._get_npu_benchmark_options()
    assert len(captured) == 3
    assert all(item.filename == tuners[0].base_fn.__code__.co_filename
               and item.lineno == tuners[0].base_fn.__code__.co_firstlineno for item in captured)
    assert "kernel kernel" in str(captured[0].message)
    assert "calibration_runs" in str(captured[2].message)


@pytest.mark.parametrize("action", ["ignore", "error"])
def test_tuner_warning_filters_and_successful_reason_tracking(monkeypatch, action, make_tuner):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "default")
    tuner = make_tuner(dict(active=30))
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter(action)
        if action == "error":
            for _ in range(2):
                with pytest.raises(RuntimeWarning, match="active"):
                    tuner._get_npu_benchmark_options()
            assert "ignored_npu_options" not in tuner._warning_reasons
        else:
            tuner._get_npu_benchmark_options()
            assert "ignored_npu_options" in tuner._warning_reasons
    assert not captured
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        tuner._get_npu_benchmark_options()
    assert len(captured) == (1 if action == "error" else 0)


@pytest.mark.parametrize("method,custom,source", [
    (None, False, "argument"),
    ("default", False, "argument"),
    ("npu", True, "argument"),
    ("default", False, "environment"),
    ("npu", True, "environment"),
])
def test_ignored_options_warn_without_validation_or_scipy(backend, monkeypatch, make_tuner, method, custom, source):
    original_import = builtins.__import__

    def import_without_quality(name, *args, **kwargs):
        assert not name.endswith("_benchmark_quality") and not name.startswith("scipy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_quality)
    if method is None:
        monkeypatch.delenv("TRITON_BENCH_METHOD")
    else:
        monkeypatch.setenv("TRITON_BENCH_METHOD", method)
    monkeypatch.setenv(backend.report.REPORT_ENV, "1")
    monkeypatch.setenv(backend.policy.ENV_PREFIX + "NOT_A_POLICY", "1")
    tuner = make_tuner(do_bench=(lambda *a, **k: 1) if custom else None)
    with warnings.catch_warnings(record=True) as diagnostics:
        warnings.simplefilter("always")
        assert tuner._get_npu_benchmark_options() is None
    assert not diagnostics  # Reporting and unknown environment names are separate.
    if source == "argument":
        tuner.npu_bench_options = {"quality_check": True}
    else:
        monkeypatch.setenv(backend.policy.ENV_PREFIX + "ACTIVE", "invalid")
    reason = "custom do_bench" if custom else "TRITON_BENCH_METHOD"
    with pytest.warns(RuntimeWarning, match=reason) as diagnostics:
        assert tuner._get_npu_benchmark_options() is None
    assert len(diagnostics) == 1
    assert ("quality_check" if source == "argument" else "active") in str(diagnostics[0].message)
    assert "REPORT_BEST_CONFIG" not in str(diagnostics[0].message)
