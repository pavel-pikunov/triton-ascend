"""Optional policies around NPU profiling; the profiler remains in testing.py."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from ._autotune_report import log_stage, report_safely, timing_stage, yellow_warning

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


class _ResolvedOptions(dict):
    """Internal snapshot whose inactive fields have already been diagnosed."""


def _supplied_option_fields(arguments=None):
    fields = {
        name
        for name in NpuBenchmarkOptions.__dataclass_fields__
        if os.getenv(ENV_PREFIX + name.upper()) is not None
    }
    if isinstance(arguments, Mapping):
        fields.update(name for name, value in arguments.items() if value is not None)
    return fields


def _inactive_option_messages(options, arguments=None):
    fields = _supplied_option_fields(arguments)
    if options.measure_budget_ms is None and "calibration_runs" in fields:
        yield ("inactive_calibration", "calibration_runs is ignored because measure_budget_ms is unset. "
               "Set measure_budget_ms to enable calibration.")
    slow_fields = sorted(
        fields & {"slow_config_runs", "slow_config_recheck_runs", "slow_config_factor", "slow_config_recheck_delay_s"})
    if not options.filter_slow_configs and slow_fields:
        yield ("inactive_slow_filter", f"{', '.join(slow_fields)} are ignored because filter_slow_configs=False. "
               "Set filter_slow_configs=True to use these fields.")


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
    slow_config_recheck_delay_s: float = 0.5
    log_level: str = "off"

    @property
    def is_default(self):
        return self._effective_policy() == NpuBenchmarkOptions()._effective_policy()

    def _effective_policy(self):
        policy = asdict(self)
        del policy["log_level"]
        if self.measure_budget_ms is None:
            del policy["calibration_runs"]
        if not self.filter_slow_configs:
            for name in ("slow_config_runs", "slow_config_recheck_runs", "slow_config_factor",
                         "slow_config_recheck_delay_s"):
                del policy[name]
        return tuple(sorted(policy.items()))

    @property
    def needs_samples(self):
        return self.quality_check or self.max_retries > 0 or self.measure_budget_ms is not None

    def cache_key(self):
        policy = self._effective_policy()
        thresholds = ()
        if self.quality_check:
            from ._benchmark_quality import THRESHOLDS
            thresholds = tuple(sorted(THRESHOLDS.items()))
        return ("npu-benchmark", 2, policy, thresholds)


def _warn_secondary(message, error):
    """Report cleanup failures without replacing an exception in flight."""
    try:
        warnings.warn(f"{message}: {error}", RuntimeWarning, stacklevel=3)
    except Exception:
        # Includes warnings-as-errors and failures in diagnostic formatting.
        pass


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
            elif name in float_fields or name == "slow_config_recheck_delay_s":
                value = float(value)
            elif name in {"quality_check", "filter_slow_configs"}:
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
    if values["log_level"] not in ("off", "brief", "detailed"):
        raise ValueError("log_level must be 'off', 'brief' or 'detailed'")
    for name in ("quality_check", "filter_slow_configs"):
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
    delay = values["slow_config_recheck_delay_s"]
    if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay < 0:
        raise ValueError("slow_config_recheck_delay_s must be finite and nonnegative")
    return NpuBenchmarkOptions(**values)


def filter_slow_configs(funcs, synchronize, options):
    """Recheck slow candidates in a later round using the best rough host time."""
    if len(funcs) < 2:
        log_stage(options.log_level, "Slow filter", f"{len(funcs)}/{len(funcs)} candidates retained; no second round")
        return list(range(len(funcs)))
    log_stage(options.log_level, "Slow filter", f"First round: {len(funcs)} candidates")
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
    suspects = [i for i, value in enumerate(first) if value > threshold]
    log_stage(options.log_level, "Slow filter", f"Second round needed: {len(suspects)}/{len(funcs)} candidates")
    log_stage(
        options.log_level, "Slow filter settings", f"rough_times_s={first}, threshold_s={threshold}, "
        f"suspects={suspects}, pause_s={options.slow_config_recheck_delay_s}", detailed=True)
    if not suspects:
        log_stage(options.log_level, "Slow filter", f"{len(funcs)}/{len(funcs)} candidates retained")
        return list(range(len(funcs)))
    if options.slow_config_recheck_delay_s:
        time.sleep(options.slow_config_recheck_delay_s)
    rechecked = {i: measure(funcs[i], options.slow_config_recheck_runs) for i in suspects}
    kept = [i for i, value in enumerate(first) if value <= threshold or rechecked[i] <= threshold]
    log_stage(options.log_level, "Slow filter", f"{len(kept)}/{len(funcs)} candidates retained")
    return kept


def benchmark_with_options(measure, funcs, names, options, *, warmup, active, prof_root, synchronize, keep_res=False,
                           _report_sink=None, _timing_sink=None):
    """Wrap the existing profiler with optional filtering, calibration and retries.

    measure returns device timestamps and durations (microseconds) per callable.
    Profiling errors can be retried; callable execution errors always propagate.
    """
    from ._benchmark_quality import central_50_mean

    if options.quality_check and active < 2:
        raise ValueError("quality_check requires active >= 2")
    if options.quality_check:
        from ._benchmark_quality import evaluate_quality
    with timing_stage(_timing_sink, "slow_filter"):
        indices = (filter_slow_configs(funcs, synchronize, options) if options.filter_slow_configs else list(
            range(len(funcs))))
    active_counts = [active] * len(funcs)
    costs = [float("inf")] * len(funcs)
    best_failures = [[] for _ in funcs]
    best_reports = [None] * len(funcs) if _report_sink is not None else None
    root = Path(prof_root)
    root.mkdir(parents=True, exist_ok=True)

    def collect(selected, warmup_count, active_count, reports=None, counts=None):
        execution_failed = False
        failed = False

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
            diagnostics = {"_report_sink": reports.extend} if reports is not None else {}
            if counts is not None:
                diagnostics["_active_counts"] = counts
            return measure([track(funcs[i]) for i in selected], [names[i] for i in selected], warmup_count,
                           active_count, directory, **diagnostics)
        except (RuntimeError, OSError) as exc:
            failed = True
            if execution_failed:
                raise
            raise ProfilerAcquisitionError(str(exc)) from exc
        except BaseException:
            failed = True
            raise
        finally:
            if not keep_res:
                try:
                    shutil.rmtree(directory)
                except Exception as exc:
                    if not failed:
                        raise
                    _warn_secondary("NPU profile cleanup failed", exc)

    with timing_stage(_timing_sink, "calibration"):
        if options.measure_budget_ms is not None:
            active_counts = calibrate_counts(collect, indices, options, active_counts, active)

    pending = indices
    last_acquisition_error = None
    with timing_stage(_timing_sink, "measurements"):
        for attempt in range(options.max_retries + 1):
            selected = pending
            counts = [active_counts[index] for index in selected]
            log_stage(options.log_level, "Profiler measurement", f"Attempt {attempt + 1}: {len(selected)} candidates")
            log_stage(options.log_level, "Measurement settings", f"configs={selected}, warmup={warmup}, "
                      f"active_per_config={counts}", detailed=True)
            reports = [] if _report_sink is not None else None
            try:
                samples = collect(selected, warmup, active, reports,
                                  counts=counts if options.measure_budget_ms is not None else None)
            except ProfilerAcquisitionError as exc:
                last_acquisition_error = exc
                log_stage(options.log_level, "Profiler acquisition failed", str(exc), detailed=True)
                pending = selected
            else:
                retry = []
                for position, (index, (times, durations)) in enumerate(zip(selected, samples)):
                    cost = central_50_mean(durations) / 1000
                    mean = float(np.mean(durations)) / 1000
                    metrics, failures = evaluate_quality(times, durations) if options.quality_check else ({}, [])
                    if cost < costs[index]:
                        costs[index], best_failures[index] = cost, failures
                        if best_reports is not None:

                            def capture():
                                best_reports[index] = None
                                if len(reports) == len(selected) and reports[position] is not None:
                                    best_reports[index] = replace(reports[position], mean_ms=mean, score_ms=cost,
                                                                  attempt=attempt + 1, quality_failures=tuple(failures),
                                                                  quality_checked=options.quality_check,
                                                                  quality_metrics=metrics)

                            report_safely(capture)
                    log_stage(
                        options.log_level, "Config measurement", f"config={index}, attempt={attempt + 1}, "
                        f"active={counts[position]}, score_ms={cost:.8g}, mean_ms={mean:.8g}, "
                        f"metrics={metrics}, failures={failures}", detailed=True)
                    if failures:
                        retry.append(index)
                pending = retry
            log_stage(
                options.log_level, "Quality/acquisition result", f"{len(selected) - len(pending)}/{len(selected)} "
                f"candidates passed; retry candidates={len(pending)}")
            if not pending:
                break

    if not any(math.isfinite(value) for value in costs):
        raise RuntimeError("NPU benchmark: no usable profiler measurements") from last_acquisition_error
    best = min(range(len(costs)), key=costs.__getitem__)
    reasons = []
    if pending:
        reasons.append(f"quality/acquisition retries exhausted for configs {pending}")
    if best_failures[best]:
        reasons.append(f"selected config {best} uses a measurement that failed quality checks: " +
                       "; ".join(best_failures[best]))
    if reasons:
        with yellow_warning():
            warnings.warn("NPU benchmark: " + "; ".join(reasons), RuntimeWarning, stacklevel=3)
    if _report_sink is not None:
        report_safely(lambda: _report_sink(best_reports))
    return costs


def calibrate_counts(collect, indices, options, active_counts, active):
    log_stage(options.log_level, "Calibration", f"Starting: {len(indices)} candidates")
    for attempt in range(options.max_retries + 1):
        log_stage(options.log_level, "Calibration profiler", f"Attempt {attempt + 1}: {len(indices)} candidates")
        try:
            calibration = collect(indices, 0, options.calibration_runs)
            break
        except ProfilerAcquisitionError as exc:
            log_stage(options.log_level, "Calibration acquisition failed", str(exc), detailed=True)
            if attempt == options.max_retries:
                raise RuntimeError("NPU benchmark: no usable calibration samples") from exc
            log_stage(options.log_level, "Calibration retry", f"Retry candidates={len(indices)}")
    for index, (_, durations) in zip(indices, calibration):
        required = options.measure_budget_ms * 1000 / float(np.mean(durations))
        if not math.isfinite(required):
            raise ValueError("Unrepresentable NPU benchmark sample count")
        active_counts[index] = max(active, math.ceil(required))
    log_stage(options.log_level, "Calibration settings", f"budget_ms={options.measure_budget_ms}, "
              f"active_per_config={active_counts}", detailed=True)
    return active_counts
