# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def _load_hyper_autotune_modules():
    package_name = "_hyper_autotune_under_test"
    package_dir = (
        Path(__file__).resolve().parents[2]
        / "backend"
        / "runtime"
        / "hyper_autotune"
    )
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_dir)]
        sys.modules[package_name] = package
    config_module = importlib.import_module(f"{package_name}.hyperparameter_config")
    tuner_module = importlib.import_module(f"{package_name}.hyperparameter_tuner")
    cache_module = importlib.import_module(f"{package_name}.hyperparameter_cache")
    return config_module, tuner_module, cache_module


_config_module, _tuner_module, _cache_module = _load_hyper_autotune_modules()
HYPER_PARAMETER_COUNT = _config_module.HYPER_PARAMETER_COUNT
HyperAutotuneConfig = _config_module.HyperAutotuneConfig
HyperparameterAutotuner = _tuner_module.HyperparameterAutotuner
HyperparameterTuningResult = _tuner_module.HyperparameterTuningResult
import_optuna = _tuner_module.import_optuna
make_compiler_flags = _tuner_module.make_compiler_flags
HyperparameterAutotuneCache = _cache_module.HyperparameterAutotuneCache


class _FakeTrial:
    def __init__(self, values):
        self._values = values
        self.params = {}
        self.value = None

    def suggest_int(self, name, low, high):
        value = self._values[len(self.params)]
        assert low <= value <= high
        self.params[name] = value
        return value


class _FakeStudy:
    def __init__(self, vectors):
        self._vectors = vectors
        self.trials = []
        self.best_trial = None
        self.optimize_kwargs = None

    def optimize(self, objective, n_trials, timeout):
        self.optimize_kwargs = {"n_trials": n_trials, "timeout": timeout}
        for vector in self._vectors[:n_trials]:
            trial = _FakeTrial(vector)
            trial.value = objective(trial)
            self.trials.append(trial)
        finite_trials = [trial for trial in self.trials if trial.value != float("inf")]
        self.best_trial = min(finite_trials, key=lambda trial: trial.value) if finite_trials else self.trials[0]


def _fake_optuna(vectors, seed_holder=None, study_holder=None):
    module = types.SimpleNamespace()

    class TPESampler:
        def __init__(self, **kwargs):
            if seed_holder is not None:
                seed_holder.update(kwargs)

    def create_study(direction, sampler):
        assert direction == "minimize"
        assert isinstance(sampler, TPESampler)
        study = _FakeStudy(vectors)
        if study_holder is not None:
            study_holder["study"] = study
        return study

    module.samplers = types.SimpleNamespace(TPESampler=TPESampler)
    module.create_study = create_study
    return module


def test_config_from_env_disabled_returns_inert_config():
    config = HyperAutotuneConfig.from_env({})

    assert config == HyperAutotuneConfig.disabled()
    assert not config.enabled
    assert config.dim == 0
    assert config.max_trials == 0
    assert config.low == ()
    assert config.high == ()


def test_config_from_env_enabled_parses_bounds_and_controls():
    config = HyperAutotuneConfig.from_env(
        {
            "TRITON_ASCEND_HYPER_AUTOTUNE": "1",
            "TRITON_ASCEND_HYPER_AUTOTUNE_DIM": str(HYPER_PARAMETER_COUNT),
            "TRITON_ASCEND_HYPER_AUTOTUNE_TRIALS": "7",
            "TRITON_ASCEND_HYPER_AUTOTUNE_TIMEOUT_SEC": "3.5",
            "TRITON_ASCEND_HYPER_AUTOTUNE_LOW": "2",
            "TRITON_ASCEND_HYPER_AUTOTUNE_HIGH": "10",
            "TRITON_ASCEND_HYPER_AUTOTUNE_SEED": "123",
            "TRITON_ASCEND_HYPER_AUTOTUNE_LOG": "true",
            "TRITON_ASCEND_HYPER_AUTOTUNE_FORCE": "yes",
        }
    )

    assert config.enabled
    assert config.dim == HYPER_PARAMETER_COUNT
    assert config.max_trials == 7
    assert config.timeout_sec == 3.5
    assert config.low == (2,) * HYPER_PARAMETER_COUNT
    assert config.high == (10,) * HYPER_PARAMETER_COUNT
    assert config.seed == 123
    assert config.log
    assert config.force


def test_config_from_env_rejects_non_32_dim_for_hyper_parameters_flag():
    with pytest.raises(ValueError, match="exactly 32"):
        HyperAutotuneConfig.from_env(
            {
                "TRITON_ASCEND_HYPER_AUTOTUNE": "1",
                "TRITON_ASCEND_HYPER_AUTOTUNE_DIM": "2",
            }
        )


def test_make_compiler_flags_returns_hyper_parameters_flag_and_32_values():
    vector = tuple(range(1, HYPER_PARAMETER_COUNT + 1))

    assert make_compiler_flags(vector) == ("--hyper-parameters",) + tuple(str(value) for value in vector)


def test_make_compiler_flags_rejects_non_32_value_vector():
    with pytest.raises(ValueError, match="exactly 32"):
        make_compiler_flags((1, 2, 3))


def test_fake_objective_tuning_uses_bounds_trials_timeout_and_seed():
    seed_holder = {}
    study_holder = {}
    config = HyperAutotuneConfig(
        enabled=True,
        dim=2,
        max_trials=3,
        timeout_sec=9.0,
        low=(1, 1),
        high=(5, 5),
        seed=99,
    )
    objective_calls = []

    def objective(vector):
        objective_calls.append(vector)
        return abs(vector[0] - 4) + abs(vector[1] - 4)

    tuner = HyperparameterAutotuner(
        config,
        objective,
        optuna_module=_fake_optuna([(1, 1), (4, 3), (4, 4)], seed_holder, study_holder),
    )

    result = tuner.tune()

    assert result == HyperparameterTuningResult(vector=(4, 4), objective=0.0)
    assert objective_calls == [(1, 1), (4, 3), (4, 4)]
    assert seed_holder == {"seed": 99}
    assert study_holder["study"].optimize_kwargs == {"n_trials": 3, "timeout": 9.0}




def test_tuning_enqueues_all_ones_default_vector_when_supported():
    class EnqueueStudy(_FakeStudy):
        def __init__(self, vectors):
            super().__init__(vectors)
            self._queued_vectors = []

        def enqueue_trial(self, params):
            self._queued_vectors.append(tuple(params[f"hyper_parameter_{index}"] for index in range(len(params))))

        def optimize(self, objective, n_trials, timeout):
            self.optimize_kwargs = {"n_trials": n_trials, "timeout": timeout}
            vectors = (self._queued_vectors + self._vectors)[:n_trials]
            for vector in vectors:
                trial = _FakeTrial(vector)
                trial.value = objective(trial)
                self.trials.append(trial)
            finite_trials = [trial for trial in self.trials if trial.value != float("inf")]
            self.best_trial = min(finite_trials, key=lambda trial: trial.value)

    module = types.SimpleNamespace()
    module.samplers = types.SimpleNamespace(TPESampler=lambda **kwargs: object())
    module.create_study = lambda direction, sampler: EnqueueStudy([(2,)])
    config = HyperAutotuneConfig(enabled=True, dim=1, max_trials=2, low=(1,), high=(3,))
    objective_calls = []

    result = HyperparameterAutotuner(
        config,
        lambda vector: objective_calls.append(vector) or float(vector[0]),
        optuna_module=module,
    ).tune()

    assert objective_calls == [(1,), (2,)]
    assert result.vector == (1,)


def test_unseeded_tuning_does_not_seed_tpe_sampler():
    seed_holder = {}
    config = HyperAutotuneConfig(enabled=True, dim=1, max_trials=1, low=(1,), high=(2,))

    result = HyperparameterAutotuner(
        config,
        lambda vector: 1.0,
        optuna_module=_fake_optuna([(1,)], seed_holder),
    ).tune()

    assert result.vector == (1,)
    assert seed_holder == {}


def test_failed_trials_return_inf_but_successful_trial_can_win():
    config = HyperAutotuneConfig(enabled=True, dim=1, max_trials=2, low=(1,), high=(2,))

    def objective(vector):
        if vector == (1,):
            raise RuntimeError("compile failed")
        return 2.5

    result = HyperparameterAutotuner(
        config,
        objective,
        optuna_module=_fake_optuna([(1,), (2,)]),
    ).tune()

    assert result == HyperparameterTuningResult(vector=(2,), objective=2.5)




def test_failed_trial_logs_exception_when_hyper_log_enabled(capsys):
    config = HyperAutotuneConfig(enabled=True, dim=1, max_trials=1, low=(1,), high=(2,), log=True)

    with pytest.raises(RuntimeError, match="all trials failed"):
        HyperparameterAutotuner(
            config,
            lambda vector: (_ for _ in ()).throw(RuntimeError("compile failed")),
            optuna_module=_fake_optuna([(1,)]),
        ).tune()

    captured = capsys.readouterr()
    assert "Triton hyper autotuning: trial failed vector=(1,)" in captured.out
    assert "RuntimeError: compile failed" in captured.out


def test_all_failed_trials_raise_clear_runtime_error():
    config = HyperAutotuneConfig(enabled=True, dim=1, max_trials=2, low=(1,), high=(2,))

    with pytest.raises(RuntimeError, match="all trials failed"):
        HyperparameterAutotuner(
            config,
            lambda vector: (_ for _ in ()).throw(RuntimeError("compile failed")),
            optuna_module=_fake_optuna([(1,), (2,)]),
        ).tune()


def test_lazy_optuna_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)

    with pytest.raises(RuntimeError, match="requires Optuna"):
        import_optuna()


def test_cache_keys_include_autotune_key_and_config_identity():
    cache = HyperparameterAutotuneCache()
    config = HyperAutotuneConfig(enabled=True, dim=1, max_trials=1, low=(1,), high=(4,))
    result = HyperparameterTuningResult(vector=(3,), objective=1.25)

    cache.put(("M", 128), config, "selected-config-a", result)

    assert cache.get(("M", 128), config, "selected-config-a").vector == (3,)
    assert cache.get(("M", 256), config, "selected-config-a") is None
    assert cache.get(("M", 128), config, "selected-config-b") is None


def test_disabled_integration_style_path_does_not_call_callback():
    config = HyperAutotuneConfig.disabled()
    callback_called = False

    def compile_and_benchmark(extra_flags):
        nonlocal callback_called
        callback_called = True
        return 0.0

    extra_compile_flags = ()
    if config.enabled:
        result = HyperparameterAutotuner(config, compile_and_benchmark).tune()
        extra_compile_flags = make_compiler_flags(result.vector)

    assert extra_compile_flags == ()
    assert not callback_called
