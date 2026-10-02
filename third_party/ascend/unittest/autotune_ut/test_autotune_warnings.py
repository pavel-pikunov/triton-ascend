"""NPU measurement settings, profiler failures and selected-config diagnostics."""

import builtins
import warnings

import pytest

import torch
from triton import Config


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
    assert tuner.best_config is configs[1] and list(tuner.cache.values()) == [configs[1]]


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
def test_unusable_final_scores_warn_and_preserve_selection_and_cache(costs, configure_run, monkeypatch):
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
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


@pytest.mark.parametrize("method,custom", [("default", False), ("npu", True)])
def test_unusable_scores_do_not_warn_for_other_benchmarkers(monkeypatch, configure_run, method, custom):
    monkeypatch.setenv("TRITON_BENCH_METHOD", method)
    configs = [Config({}), Config({})]
    kwargs = {"do_bench": lambda *a, **k: float("inf")} if custom else {}
    tuner = configure_run(configs[0], configs, False, **kwargs)
    tuner._batch_bench = lambda *a, **k: dict.fromkeys(configs, float("inf"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert tuner.run(torch.empty(1)) == "result"
    assert tuner._npu_benchmark_options is None and tuner.best_config is configs[0]


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
