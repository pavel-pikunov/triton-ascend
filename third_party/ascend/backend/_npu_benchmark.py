"""Optional policies around NPU profiling; the profiler remains in testing.py."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ._benchmark_quality import THRESHOLDS, evaluate_quality


ENV_PREFIX = "TRITON_NPU_BENCH_"
COUNT_MINIMUMS = {
    "warmup": 0,
    "active": 1,
    "max_retries": 0,
    "calibration_runs": 2,
    "slow_config_runs": 1,
    "slow_config_recheck_runs": 1,
}


class ProfilerAcquisitionError(RuntimeError):
    """The profiler did not produce a complete, usable set of device samples."""


@dataclass(frozen=True)
class NpuBenchmarkOptions:
    # None inherits the caller's existing settings, including CV-derived counts.
    cache_mode: str | None = None
    warmup: int | None = None
    active: int | None = None
    quality_check: bool = False
    max_retries: int = 0
    measure_budget_ms: float | None = None
    calibration_runs: int = 10
    filter_slow_configs: bool = False
    slow_config_runs: int = 20
    slow_config_recheck_runs: int = 40
    slow_config_factor: float = 5.0
    verbose: bool | None = None

    @property
    def is_default(self):
        return self == NpuBenchmarkOptions()

    @property
    def needs_samples(self):
        return self.quality_check or self.max_retries > 0 or self.measure_budget_ms is not None

    def cache_key(self):
        policy = tuple(sorted(asdict(self).items()))
        thresholds = tuple(sorted(THRESHOLDS.items())) if self.quality_check else ()
        return ("npu-benchmark", 2, policy, thresholds)


def resolve_options(arguments=None):
    """Resolve each field independently: argument > environment > default."""
    if arguments is not None and not isinstance(arguments, Mapping):
        raise TypeError("npu_bench_options must be a mapping")
    arguments = dict(arguments or {})
    values = asdict(NpuBenchmarkOptions())
    unknown = arguments.keys() - values.keys()
    if unknown:
        raise ValueError(f"Unknown NPU benchmark options: {', '.join(sorted(unknown))}")

    float_fields = {"measure_budget_ms", "slow_config_factor"}
    for name in values:
        if arguments.get(name) is not None:
            values[name] = arguments[name]
            continue
        value = os.getenv(ENV_PREFIX + name.upper())
        if value is None:
            continue
        try:
            if name in COUNT_MINIMUMS:
                value = int(value)
            elif name in float_fields:
                value = float(value)
            elif name in {"quality_check", "filter_slow_configs", "verbose"}:
                normalized = value.strip().lower()
                if normalized not in {"0", "1", "false", "true"}:
                    raise ValueError("expected 0, 1, false or true")
                value = normalized in {"1", "true"}
            else:
                value = value.strip().lower()
        except ValueError as exc:
            raise ValueError(f"Invalid {ENV_PREFIX + name.upper()}: {value!r}") from exc
        values[name] = value

    if values["cache_mode"] not in (None, "hot", "cold"):
        raise ValueError("cache_mode must be 'hot' or 'cold'")
    for name in ("quality_check", "filter_slow_configs", "verbose"):
        if values[name] is not None and not isinstance(values[name], bool):
            raise ValueError(f"{name} must be a boolean")
    for name, minimum in COUNT_MINIMUMS.items():
        value = values[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if values["quality_check"] and values["active"] is not None and values["active"] < 2:
        raise ValueError("quality_check requires active >= 2")
    for name in float_fields:
        value = values[name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if values["slow_config_factor"] < 1:
        raise ValueError("slow_config_factor must be >= 1 to retain the fastest candidate")
    return NpuBenchmarkOptions(**values)


def filter_slow_configs(funcs, synchronize, options):
    """Conservatively reject slow candidates using the best rough host time."""
    if len(funcs) < 2:
        return list(range(len(funcs)))
    for fn in funcs:
        fn()
        synchronize()

    def measure(fn, count):
        best = float("inf")
        for _ in range(count):
            start = time.perf_counter()
            fn()
            synchronize()
            best = min(best, time.perf_counter() - start)
        return best

    first = [measure(fn, options.slow_config_runs) for fn in funcs]
    threshold = min(first) * options.slow_config_factor
    return [i for i, value in enumerate(first)
            if value <= threshold or measure(funcs[i], options.slow_config_recheck_runs) <= threshold]


def benchmark_with_options(measure, funcs, names, options, *, warmup, active, prof_root, synchronize,
                           verbose=False, keep_res=False):
    """Wrap the existing profiler with optional filtering, calibration and retries.

    measure returns device timestamps and durations (microseconds) per callable.
    Profiling errors can be retried; callable execution errors always propagate.
    """
    if options.quality_check and active < 2:
        raise ValueError("quality_check requires active >= 2")
    verbose = verbose if options.verbose is None else options.verbose

    def log(message):
        if verbose:
            print(f"npu benchmark: {message}")

    indices = (filter_slow_configs(funcs, synchronize, options)
               if options.filter_slow_configs else list(range(len(funcs))))
    active_counts = [active] * len(funcs)
    costs = [float("inf")] * len(funcs)
    best_failures = [[] for _ in funcs]
    root = Path(prof_root)
    root.mkdir(parents=True, exist_ok=True)

    def collect(selected, warmup_count, active_count):
        execution_failed = False

        def track(fn):
            def call():
                nonlocal execution_failed
                try:
                    result = fn()
                    synchronize()
                    return result
                except Exception:
                    execution_failed = True
                    raise
            return call

        directory = tempfile.mkdtemp(prefix="npu_bench_", dir=root)
        try:
            return measure([track(funcs[i]) for i in selected], [names[i] for i in selected],
                           warmup_count, active_count, directory)
        except (RuntimeError, OSError) as exc:
            if execution_failed:
                raise
            raise ProfilerAcquisitionError(str(exc)) from exc
        finally:
            if not keep_res:
                shutil.rmtree(directory, ignore_errors=True)

    log(f"configs={len(funcs)}, kept={indices}, warmup={warmup}, active={active}")
    if options.measure_budget_ms is not None:
        for attempt in range(options.max_retries + 1):
            try:
                calibration = collect(indices, 0, options.calibration_runs)
                break
            except ProfilerAcquisitionError as exc:
                log(f"calibration attempt={attempt + 1}: {exc}")
                if attempt == options.max_retries:
                    raise RuntimeError("NPU benchmark: no usable calibration samples") from exc
        for index, (_, durations) in zip(indices, calibration):
            required = options.measure_budget_ms * 1000 / float(np.mean(durations))
            if not math.isfinite(required):
                raise ValueError("Unrepresentable NPU benchmark sample count")
            active_counts[index] = max(active, math.ceil(required))
        log(f"budget_ms={options.measure_budget_ms}, active_per_config={active_counts}")

    pending = indices
    last_acquisition_error = None
    for attempt in range(options.max_retries + 1):
        groups = {}
        for index in pending:
            groups.setdefault(active_counts[index], []).append(index)
        retry = []
        for active_count, selected in groups.items():
            try:
                samples = collect(selected, warmup, active_count)
            except ProfilerAcquisitionError as exc:
                last_acquisition_error = exc
                log(f"attempt={attempt + 1}, configs={selected}, active={active_count}: {exc}")
                retry.extend(selected)
                continue
            for index, (times, durations) in zip(selected, samples):
                # Keep the existing NPU arithmetic mean for configuration ranking.
                cost = float(np.mean(durations)) / 1000
                metrics, failures = evaluate_quality(times, durations) if options.quality_check else ({}, [])
                if cost < costs[index]:
                    costs[index], best_failures[index] = cost, failures
                log(f"config={index}, attempt={attempt + 1}, active={active_count}, cost_ms={cost:.8g}, "
                    f"metrics={metrics}, failures={failures}")
                if failures:
                    retry.append(index)
        pending = retry
        if not pending:
            break

    if not any(math.isfinite(value) for value in costs):
        raise RuntimeError("NPU benchmark: no usable profiler measurements") from last_acquisition_error
    best = min(range(len(costs)), key=costs.__getitem__)
    reasons = []
    if pending:
        reasons.append(f"quality/acquisition retries exhausted for configs {pending}")
    if best_failures[best]:
        reasons.append(f"selected config {best} uses a measurement that failed quality checks: "
                       + "; ".join(best_failures[best]))
    if reasons:
        warnings.warn("NPU benchmark: " + "; ".join(reasons), RuntimeWarning, stacklevel=3)
    return costs
