#!/usr/bin/env python3
"""Real-Optuna synthetic environment for Ascend hyper-autotune.

This is an environment simulation, not an optimizer simulation: it uses the real
``HyperparameterAutotuner`` and therefore requires the optional ``optuna`` Python
package.  It does not require Ascend hardware, CANN, or a built Triton runtime.

Run from the repository root:

    python3 -m pip install optuna
    python3 tools/hyper_autotune_simulation_benchmark.py --trials 80 --repeats 5

The script validates:

* functionality: the real optimizer proposes vectors through
  ``HyperparameterAutotuner`` and ``make_compiler_flags``;
* correctness: the synthetic kernel output is compared with a deterministic
  reference for both baseline and tuned vectors;
* performance: the tuned vector must be faster than a deliberately poor
  baseline in a repeatable synthetic scheduling model;
* cache behavior: the best vector/objective round-trips through the in-memory
  hyper-autotune cache.
"""

from __future__ import annotations

import argparse
import importlib
import math
import statistics
import sys
import time
import types
from pathlib import Path
from typing import List, Sequence, Tuple


def _load_hyper_autotune_modules():
    """Load the new package without importing the full Triton/Ascend backend."""
    package_name = "_real_optuna_hyper_autotune_simulation"
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
        importlib.import_module(f"{package_name}.hyperparameter_cache"),
        importlib.import_module(f"{package_name}.hyperparameter_config"),
        importlib.import_module(f"{package_name}.hyperparameter_tuner"),
    )


_cache_module, _config_module, _tuner_module = _load_hyper_autotune_modules()
HyperparameterAutotuneCache = _cache_module.HyperparameterAutotuneCache
HyperAutotuneConfig = _config_module.HyperAutotuneConfig
HyperparameterAutotuner = _tuner_module.HyperparameterAutotuner
make_compiler_flags = _tuner_module.make_compiler_flags


TARGET_VECTOR = (5, 3, 2)
BASELINE_VECTOR = (1, 1, 1)


def _complex_reference(data: Sequence[float]) -> List[float]:
    """Reference for a nonlinear, deterministic synthetic workload."""
    return [
        math.sin(x) * math.cos(x * 0.25)
        + math.sqrt(x + 1.0) * 0.125
        - math.log1p(x) * 0.03125
        + (x * x - 3.0 * x + 2.0) * 0.0005
        for x in data
    ]


def _synthetic_kernel(vector: Tuple[int, ...], data: Sequence[float]) -> List[float]:
    """Compute the same result for every vector, with vector-dependent cost.

    The vector-dependent burn loop emulates compile-flag-sensitive scheduling:
    bad vectors do extra work, while the target vector has no extra overhead.
    The accumulator is multiplied by zero before touching the output, so it
    cannot change correctness while still preventing the loop from being a no-op.
    """
    result = _complex_reference(data)
    penalty = sum((actual - expected) ** 2 for actual, expected in zip(vector, TARGET_VECTOR))
    burn_iterations = penalty * 6000
    accumulator = 0.0
    for index in range(burn_iterations):
        accumulator += ((index % 17) - 8) * 1.0e-12
    if result:
        result[0] += accumulator * 0.0
    return result


def _time_kernel(vector: Tuple[int, ...], data: Sequence[float], repeats: int) -> Tuple[float, float]:
    reference = _complex_reference(data)
    timings = []
    max_abs_error = 0.0
    for _ in range(repeats):
        started = time.perf_counter()
        actual = _synthetic_kernel(vector, data)
        timings.append((time.perf_counter() - started) * 1000.0)
        max_abs_error = max(max_abs_error, max(abs(lhs - rhs) for lhs, rhs in zip(actual, reference)))
    return statistics.median(timings), max_abs_error


def _config_from_args(args) -> HyperAutotuneConfig:
    return HyperAutotuneConfig.from_env(
        {
            "TRITON_ASCEND_HYPER_AUTOTUNE": "1",
            "TRITON_ASCEND_HYPER_AUTOTUNE_DIM": "3",
            "TRITON_ASCEND_HYPER_AUTOTUNE_TRIALS": str(args.trials),
            "TRITON_ASCEND_HYPER_AUTOTUNE_TIMEOUT_SEC": str(args.timeout_sec),
            "TRITON_ASCEND_HYPER_AUTOTUNE_LOW": "1,1,1",
            "TRITON_ASCEND_HYPER_AUTOTUNE_HIGH": "7,5,3",
            "TRITON_ASCEND_HYPER_AUTOTUNE_SEED": str(args.seed),
        }
    )


def run_benchmark(args) -> None:
    config = _config_from_args(args)
    data = [index / 64.0 for index in range(args.size)]

    baseline_ms, baseline_error = _time_kernel(BASELINE_VECTOR, data, args.repeats)
    evaluated_vectors = set()

    def objective(vector: Tuple[int, ...]) -> float:
        flags = make_compiler_flags(vector)
        if flags != ("--hyper-max-parallel-parameters=" + ",".join(str(value) for value in vector),):
            raise RuntimeError(f"unexpected compiler flags for {vector}: {flags}")
        elapsed_ms, max_abs_error = _time_kernel(vector, data, repeats=1)
        evaluated_vectors.add(vector)
        return elapsed_ms + max_abs_error * 1.0e9

    tuner = HyperparameterAutotuner(config, objective)
    tuning_result = tuner.tune()
    tuned_ms, tuned_error = _time_kernel(tuning_result.vector, data, args.repeats)
    speedup = baseline_ms / tuned_ms if tuned_ms > 0 else float("inf")

    cache = HyperparameterAutotuneCache()
    cache.put("synthetic-autotune-key", config, "synthetic-selected-config", tuning_result)
    cached = cache.get("synthetic-autotune-key", config, "synthetic-selected-config")

    print("Real-Optuna hyper-autotune simulation benchmark")
    print(f"  config={config}")
    print(f"  evaluated_vectors={len(evaluated_vectors)}")
    print(f"  baseline_vector={BASELINE_VECTOR}")
    print(f"  baseline_flags={make_compiler_flags(BASELINE_VECTOR)}")
    print(f"  baseline_median_ms={baseline_ms:.3f}")
    print(f"  baseline_max_abs_error={baseline_error:.3e}")
    print(f"  tuned_vector={tuning_result.vector}")
    print(f"  tuned_flags={make_compiler_flags(tuning_result.vector)}")
    print(f"  tuned_objective={tuning_result.objective:.3f}")
    print(f"  tuned_median_ms={tuned_ms:.3f}")
    print(f"  tuned_max_abs_error={tuned_error:.3e}")
    print(f"  speedup={speedup:.2f}x")
    print(f"  cache_vector={cached.vector if cached is not None else None}")

    if cached is None or cached.vector != tuning_result.vector:
        raise AssertionError("cache did not return the tuned vector")
    if baseline_error > args.tolerance or tuned_error > args.tolerance:
        raise AssertionError(
            "synthetic kernel output differs from the reference: "
            f"baseline_error={baseline_error}, tuned_error={tuned_error}"
        )
    if speedup < args.min_speedup:
        raise AssertionError(f"expected at least {args.min_speedup:.2f}x speedup, got {speedup:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=80, help="Optuna trials for the real tuner")
    parser.add_argument("--timeout-sec", type=float, default=30.0, help="Optuna timeout in seconds")
    parser.add_argument("--seed", type=int, default=0, help="TPESampler seed")
    parser.add_argument("--repeats", type=int, default=5, help="timing repeats for baseline and final tuned runs")
    parser.add_argument("--size", type=int, default=768, help="synthetic input length")
    parser.add_argument("--tolerance", type=float, default=1.0e-12, help="maximum allowed absolute error")
    parser.add_argument("--min-speedup", type=float, default=1.25, help="minimum tuned-vs-baseline speedup")
    args = parser.parse_args()

    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.timeout_sec <= 0:
        raise ValueError("--timeout-sec must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.size <= 0:
        raise ValueError("--size must be positive")
    run_benchmark(args)


if __name__ == "__main__":
    main()
