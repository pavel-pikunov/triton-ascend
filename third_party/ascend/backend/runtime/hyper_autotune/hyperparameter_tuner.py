# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from .hyperparameter_config import HyperAutotuneConfig

Objective = Callable[[Tuple[int, ...]], float]


def import_optuna():
    try:
        return importlib.import_module("optuna")
    except ImportError as exc:
        raise RuntimeError(
            "TRITON_ASCEND_HYPER_AUTOTUNE requires Optuna. "
            "Install optuna or unset TRITON_ASCEND_HYPER_AUTOTUNE."
        ) from exc


def make_compiler_flags(vector: Sequence[int]) -> Tuple[str, ...]:
    return ("--hyper-max-parallel-parameters=" + ",".join(str(int(value)) for value in vector),)


@dataclass(frozen=True)
class HyperparameterTuningResult:
    vector: Tuple[int, ...]
    objective: float


class HyperparameterAutotuner:
    def __init__(self, config: HyperAutotuneConfig, objective: Objective, optuna_module=None):
        if not config.enabled:
            raise ValueError("HyperparameterAutotuner requires an enabled config")
        self.config = config
        self.objective = objective
        self._optuna = optuna_module

    def _load_optuna(self):
        if self._optuna is None:
            self._optuna = import_optuna()
        return self._optuna

    def _suggest_vector(self, trial) -> Tuple[int, ...]:
        return tuple(
            int(trial.suggest_int(f"hyper_max_parallel_parameter_{index}", low, high))
            for index, (low, high) in enumerate(zip(self.config.low, self.config.high))
        )

    def tune(self) -> HyperparameterTuningResult:
        optuna = self._load_optuna()
        sampler_kwargs = {}
        if self.config.seed is not None:
            sampler_kwargs["seed"] = self.config.seed
        sampler = optuna.samplers.TPESampler(**sampler_kwargs)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def trial_objective(trial) -> float:
            vector = self._suggest_vector(trial)
            try:
                value = float(self.objective(vector))
            except Exception:
                return float("inf")
            if math.isnan(value):
                return float("inf")
            return value

        study.optimize(trial_objective, n_trials=self.config.max_trials, timeout=self.config.timeout_sec)

        complete_trials = [trial for trial in getattr(study, "trials", []) if math.isfinite(float(getattr(trial, "value", float("inf"))))]
        if not complete_trials:
            raise RuntimeError("Hyperparameter autotuning failed: all trials failed")

        best_trial = getattr(study, "best_trial", None)
        best_value = float(getattr(best_trial, "value", float("inf"))) if best_trial is not None else float("inf")
        if not math.isfinite(best_value):
            best_trial = min(complete_trials, key=lambda trial: float(trial.value))
            best_value = float(best_trial.value)

        best_vector = tuple(
            int(best_trial.params[f"hyper_max_parallel_parameter_{index}"])
            for index in range(self.config.dim)
        )
        return HyperparameterTuningResult(vector=best_vector, objective=best_value)
