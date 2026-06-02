#!/usr/bin/env python3
"""Manual hyperparameter autotune smoke test using a deterministic synthetic objective."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path


def _load_hyper_autotune_modules():
    package_name = "_manual_hyper_autotune"
    package_dir = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "ascend"
        / "backend"
        / "runtime"
        / "hyper_autotune"
    )
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_dir)]
        sys.modules[package_name] = package
    return (
        importlib.import_module(f"{package_name}.hyperparameter_config"),
        importlib.import_module(f"{package_name}.hyperparameter_tuner"),
    )


_config_module, _tuner_module = _load_hyper_autotune_modules()
HyperAutotuneConfig = _config_module.HyperAutotuneConfig
HyperparameterAutotuner = _tuner_module.HyperparameterAutotuner
make_compiler_flags = _tuner_module.make_compiler_flags


def synthetic_objective(vector):
    target = (4,) * len(vector)
    return float(sum((value - expected) ** 2 for value, expected in zip(vector, target)))


def main() -> None:
    os.environ.setdefault("TRITON_ASCEND_HYPER_AUTOTUNE", "1")
    os.environ.setdefault("TRITON_ASCEND_HYPER_AUTOTUNE_DIM", "2")
    os.environ.setdefault("TRITON_ASCEND_HYPER_AUTOTUNE_TRIALS", "16")
    os.environ.setdefault("TRITON_ASCEND_HYPER_AUTOTUNE_LOW", "1")
    os.environ.setdefault("TRITON_ASCEND_HYPER_AUTOTUNE_HIGH", "8")
    os.environ.setdefault("TRITON_ASCEND_HYPER_AUTOTUNE_SEED", "0")

    config = HyperAutotuneConfig.from_env()
    result = HyperparameterAutotuner(config, synthetic_objective).tune()
    print(f"best_vector={result.vector}")
    print(f"objective={result.objective}")
    print(f"flags={make_compiler_flags(result.vector)}")


if __name__ == "__main__":
    main()
