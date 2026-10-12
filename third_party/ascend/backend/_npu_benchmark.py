"""Environment-controlled NPU measurements and diagnostics."""

from __future__ import annotations

import io
import math
import os
import pprint
import re
import shutil
import tempfile
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import fmean

import numpy as np
import triton.runtime as runtime

ENV_PREFIX = "TRITON_NPU_BENCH_"
TIMING_STAGES = ("generation_pruning", "compilation", "slow_filter", "calibration", "measurements")
COUNT_MINIMUMS = {"warmup": 0, "active": 1, "max_retries": 0, "calibration_runs": 2, "slow_config_rounds": 1}


class ProfilerResultMismatchError(RuntimeError):

    def __init__(self, target_kernel_name, expected_rows, actual_rows):
        self.target_kernel_name = target_kernel_name
        self.expected_rows = expected_rows
        self.actual_rows = actual_rows
        super().__init__(
            "Profiler rows filtered by target kernel name do not match the expected count. "
            f"target_kernel_name={target_kernel_name!r}, expected_rows={expected_rows}, actual_rows={actual_rows}")


class ProfilerAcquisitionError(RuntimeError):
    """The profiler did not produce usable device samples."""


@dataclass(frozen=True)
class NpuBenchmarkOptions:
    cache_mode: str | None = None
    warmup: int | None = None
    active: int | None = None
    quality_check: bool = False
    max_retries: int = 0
    measure_budget_ms: float | None = None
    calibration_runs: int = 10
    filter_slow_configs: bool = False
    slow_config_rounds: int = 2
    slow_config_factor: float = 5.0
    slow_config_recheck_delay_s: float = 0.5
    log_level: str = "off"
    report_best_config: bool = False
    report_timing: bool = False

    def effective_policy(self):
        policy = asdict(self)
        for name in ("log_level", "report_best_config", "report_timing"):
            del policy[name]
        if self.measure_budget_ms is None:
            del policy["calibration_runs"]
        if not self.filter_slow_configs:
            for name in ("slow_config_rounds", "slow_config_factor", "slow_config_recheck_delay_s"):
                del policy[name]
        elif self.slow_config_rounds == 1:
            del policy["slow_config_recheck_delay_s"]
        return tuple(sorted(policy.items()))

    @property
    def is_default(self):
        return self.effective_policy() == NpuBenchmarkOptions().effective_policy()

    def cache_key(self):
        thresholds = ()
        if self.quality_check:
            from ._benchmark_quality import THRESHOLDS
            thresholds = tuple(sorted(THRESHOLDS.items()))
        return ("npu-benchmark", 4, self.effective_policy(), thresholds)


def resolve_options():
    """Read the NPU environment; unset fields inherit existing call arguments."""
    values = asdict(NpuBenchmarkOptions())
    positive = {"measure_budget_ms", "slow_config_factor"}
    for name, default in values.items():
        environment = "TRITON_AUTOTUNE_REPORT_TIMING" if name == "report_timing" else ENV_PREFIX + name.upper()
        value = os.getenv(environment)
        if value is None:
            continue
        try:
            if isinstance(default, bool):
                normalized = value.strip().lower()
                if normalized not in {"0", "1", "false", "true"}:
                    raise ValueError("expected 0, 1, false or true")
                value = normalized in {"1", "true"}
            elif name in COUNT_MINIMUMS:
                value = int(value)
                if value < COUNT_MINIMUMS[name]:
                    raise ValueError(f"expected >= {COUNT_MINIMUMS[name]}")
            elif name in positive or name == "slow_config_recheck_delay_s":
                value = float(value)
                minimum = 1 if name == "slow_config_factor" else 0
                if not math.isfinite(value) or value < minimum or (name in positive and value == 0):
                    raise ValueError("invalid finite range")
            else:
                value = value.strip().lower()
                allowed = {"hot", "cold"} if name == "cache_mode" else {"off", "brief", "detailed"}
                if value not in allowed:
                    raise ValueError(f"expected one of {sorted(allowed)}")
        except ValueError as exc:
            raise ValueError(f"Invalid {environment}: {value!r}") from exc
        values[name] = value
    if values["quality_check"] and values["active"] is not None and values["active"] < 2:
        raise ValueError("quality_check requires active >= 2")
    return NpuBenchmarkOptions(**values)


def report_safely(action):
    """Optional diagnostics must not change measurement or execution failures."""
    try:
        return action()
    except Exception:
        return None


def log_stage(level, title, message, *, detailed=False):
    if level != "off" and (not detailed or level == "detailed"):
        report_safely(lambda: print(f"NPU autotune: {title}\n  {message}"))


@contextmanager
def timing_stage(sink, name):
    started = time.perf_counter() if sink is not None else None
    try:
        yield
    finally:
        if sink is not None:
            elapsed = time.perf_counter() - started
            report_safely(lambda: sink(name, elapsed))


@dataclass(frozen=True)
class NpuMeasurementReport:
    metrics: dict
    sample_count: int
    warmup: int
    active: int
    cache_mode: str
    mean_ms: float
    score_ms: float
    attempt: int = 1
    quality_failures: tuple = ()
    quality_checked: bool = False
    quality_metrics: dict = field(default_factory=dict)


def aggregate_profile_rows(frame):
    metrics = {}
    for column in frame.columns:
        values = [
            str(value).strip()
            for value in frame[column]
            if str(value).strip().lower() not in {"", "nan", "n/a", "null", "none"}
        ]
        absolute = re.search(r"(^|[^a-z])(id|timestamp)([^a-z]|$)|start\s*time|end\s*time", column.lower())
        try:
            numbers = [float(value) for value in values] if not absolute else []
        except ValueError:
            numbers = []
        if numbers:
            numbers = [value for value in numbers if math.isfinite(value)]
            metrics[column] = (fmean(numbers), len(numbers), True) if numbers else ("unavailable", 0, False)
        else:
            text = values[0] if len(set(values)) == 1 else "varies" if values else "unavailable"
            metrics[column] = (text, len(values), False)
    return metrics


def print_best_config_report(function_name, configs, config, measurement):
    number = next((i for i, candidate in enumerate(configs, 1) if candidate is config), None)
    stream = io.StringIO()
    stream.write(f"Triton autotuning result for {function_name}\n" + "-" * 60 + "\n")
    stream.write(f"Selected config: {number}/{len(configs)} (1-based, before pruning)\n"
                 if number is not None else "Selected config: not in the original candidate list\n")
    parameters = dict(vars(config))
    ub_config = parameters.pop("ubtune_cfg", None)
    stream.write("Full config parameters:\n" + pprint.pformat(parameters, sort_dicts=False) + "\n")
    if ub_config is not None:
        stream.write("ubtune_cfg:\n" + pprint.pformat(ub_config, sort_dicts=False) + "\n")
    if measurement is None:
        stream.write("NPU measurements unavailable for this selection; no extra benchmarking was run.\n")
    else:
        stream.write(f"Central 50% score: {measurement.score_ms * 1000:.12g} us ({measurement.score_ms:.12g} ms)\n")
        stream.write(f"Mean duration: {measurement.mean_ms * 1000:.12g} us ({measurement.mean_ms:.12g} ms)\n")
        stream.write(f"cache={measurement.cache_mode}, warmup={measurement.warmup}, active={measurement.active}, "
                     f"selected attempt={measurement.attempt}\n")
        quality = ("failed"
                   if measurement.quality_failures else "passed" if measurement.quality_checked else "not requested")
        stream.write(f"Quality checks: {quality}; metrics={measurement.quality_metrics}\n")
        if measurement.quality_failures:
            stream.write("Failure reasons: " + "; ".join(measurement.quality_failures) + "\n")
        stream.write(f"Profiler aggregates: {measurement.sample_count} measured launches (warmup excluded)\n")
        for column, (value, count, numeric) in measurement.metrics.items():
            formatted = f"{value:.12g}" if numeric else value
            available = f" ({count}/{measurement.sample_count} available)" if count != measurement.sample_count else ""
            stream.write(f"  {column}: {'mean' if numeric else 'value'}={formatted}{available}\n")
    print(stream.getvalue(), end="")


def print_timing_report(function_name, started, durations):
    total = time.perf_counter() - started
    labels = ("Generation/pruning", "Compilation", "Slow filter", "Calibration", "Measurements (including retries)")
    lines = [f"Triton autotuning timing for {function_name}", "-" * 60, f"  Total: {total:.6f} s"]
    lines.extend(f"  {label}: {durations.get(stage, 0):.6f} s" for stage, label in zip(TIMING_STAGES, labels))
    lines.append(f"  Other: {max(0, total - sum(durations.values())):.6f} s")
    print("\n".join(lines))


def _collect_prof_result(base_dir, funcs, num_warmup, num_active, target_kernel_name=None, clear_l2_cache=False,
                         _return_samples=False, _report_sink=None, _active_counts=None):
    """Original pandas collector with cumulative counts and optional sample/report extraction."""
    import pandas as pd
    from ._benchmark_quality import central_50_mean

    kernel_details_file = None
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "kernel_details.csv":
                kernel_details_file = os.path.join(root, file)
                break
    num_funcs = len(funcs)
    if kernel_details_file is None:
        if _return_samples:
            raise ProfilerAcquisitionError("NPU profiler CSV kernel_details.csv was not found")
        return float("inf") if num_funcs == 1 else [float("inf")] * num_funcs

    df = pd.read_csv(kernel_details_file)
    filter_cond = (not clear_l2_cache) | ~df["Type"].str.contains(r"^ReduceSum$", case=False, na=False)
    filter_df = df[filter_cond]
    names = ([target_kernel_name] * num_funcs
             if target_kernel_name is None or isinstance(target_kernel_name, str) else list(target_kernel_name))
    if target_kernel_name is not None:
        if isinstance(target_kernel_name, str):
            filter_df = filter_df[filter_df["Name"] == target_kernel_name]
        elif all(name is not None for name in names):
            filter_df = filter_df[filter_df["Name"].isin(names)]

    counts = [num_active] * num_funcs if _active_counts is None else list(_active_counts)
    if len(counts) != num_funcs or len(names) != num_funcs:
        raise ValueError("Provide one kernel name and active count per function")
    if num_warmup < 0 or any(count < 1 for count in counts):
        raise ValueError("Warmup must be nonnegative and active counts positive")
    expected_rows = sum(num_warmup + count for count in counts)
    if target_kernel_name is not None and len(filter_df) != expected_rows:
        raise ProfilerResultMismatchError(target_kernel_name, expected_rows, len(filter_df))

    costs, samples, reports = [], [], []
    offset = 0
    for index, count in enumerate(counts):
        measured = filter_df.iloc[offset + num_warmup:offset + num_warmup + count]
        offset += num_warmup + count
        durations = measured["Duration(us)"].to_numpy(dtype=float)
        if len(durations) != count:
            raise ProfilerAcquisitionError(f"Incomplete profiler rows for config {index}")
        if names[index] is not None and not measured["Name"].eq(names[index]).all():
            raise ProfilerAcquisitionError(f"Unexpected profiler target order for config {index}")
        cost = central_50_mean(durations) / 1000
        costs.append(cost)
        if _return_samples:
            times = measured["Start Time(us)"].to_numpy(dtype=float)
            if (not np.all(np.isfinite(times)) or not np.all(np.isfinite(durations)) or np.any(durations <= 0)
                    or np.any(np.diff(times) <= 0)):
                raise ProfilerAcquisitionError(f"Invalid profiler samples for config {index}")
            samples.append((times, durations))
        if _report_sink is not None:
            reports.append(
                report_safely(lambda: NpuMeasurementReport(aggregate_profile_rows(measured), count, num_warmup, count,
                                                           "cold" if clear_l2_cache else "hot",
                                                           float(np.mean(durations)) / 1000, cost)))
    if _report_sink is not None:
        report_safely(lambda: _report_sink(reports))
    return samples if _return_samples else costs[0] if num_funcs == 1 else costs


def filter_slow_configs(funcs, options):
    """Benchmark everyone once, then recheck only persistent suspects."""
    from triton.testing import do_bench

    indices = list(range(len(funcs)))
    if len(funcs) < 2:
        return indices, {}
    log_stage(options.log_level, "Slow filter", f"Round 1/{options.slow_config_rounds}: {len(funcs)} candidates")
    durations = {index: do_bench(fn, return_mode="mean") for index, fn in enumerate(funcs)}
    threshold = min(durations.values()) * options.slow_config_factor
    suspects = [index for index in indices if durations[index] > threshold]
    log_stage(options.log_level, "Slow filter", f"Further rounds needed: {len(suspects)}/{len(funcs)} candidates")
    for round_index in range(1, options.slow_config_rounds):
        if not suspects:
            break
        if options.slow_config_recheck_delay_s:
            time.sleep(options.slow_config_recheck_delay_s)
        log_stage(options.log_level, "Slow filter", f"Round {round_index + 1}/{options.slow_config_rounds}: "
                  f"{len(suspects)} suspects")
        for index in suspects:
            durations[index] = do_bench(funcs[index], return_mode="mean")
        suspects = [index for index in suspects if durations[index] > threshold]
    kept = [index for index in indices if index not in suspects]
    log_stage(options.log_level, "Slow filter", f"{len(kept)}/{len(funcs)} candidates retained")
    log_stage(options.log_level, "Slow filter settings", f"event_mean_ms={durations}, threshold_ms={threshold}",
              detailed=True)
    return kept, durations


@contextmanager
def event_timer(clear_l2_cache, pre_hook_scope):
    """Short event calibration when ordinary slow-filter measurements are unavailable."""
    import torch

    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    end.record()
    torch.npu.synchronize()
    buffer = runtime.driver.active.get_empty_cache_for_benchmark().float() if clear_l2_cache else None
    scoped = buffer is not None and pre_hook_scope is not None

    def prepare():
        if buffer is not None:
            buffer.sum()
            torch.npu.synchronize()
        start.record()

    def measure(fn, count):
        total_ms = 0.0
        for _ in range(count):
            if not scoped:
                prepare()
            fn()
            end.record()
            torch.npu.synchronize()
            total_ms += start.elapsed_time(end)
        return total_ms / count

    try:
        with pre_hook_scope(prepare) if scoped else nullcontext():
            yield measure
    finally:
        buffer = None


def calibrate_counts(funcs, indices, durations, options, active, clear_l2_cache, pre_hook_scope):
    counts = [active] * len(funcs)
    if options.measure_budget_ms is None:
        return counts
    missing = [index for index in indices if index not in durations]
    log_stage(
        options.log_level, "Calibration", f"Reusing slow-filter events: {len(indices) - len(missing)}/"
        f"{len(indices)}; event calibration: {len(missing)} candidates")
    if missing:
        import torch

        with event_timer(clear_l2_cache, pre_hook_scope) as measure:
            for index in missing:
                funcs[index]()
                torch.npu.synchronize()
                durations[index] = measure(funcs[index], options.calibration_runs)
    for index in indices:
        duration = durations[index]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid NPU event calibration time for config {index}: {duration}")
        required = options.measure_budget_ms / duration
        if not math.isfinite(required):
            raise ValueError("Unrepresentable NPU benchmark sample count")
        counts[index] = max(active, math.ceil(required))
    log_stage(options.log_level, "Calibration settings", f"budget_ms={options.measure_budget_ms}, "
              f"event_mean_ms={durations}, active_per_config={counts}", detailed=True)
    return counts


@contextmanager
def result_scope(remove):
    """Keep a primary failure if cleanup also fails."""
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        try:
            remove()
        except Exception:
            if not failed:
                raise


@contextmanager
def profile_scope(buffer, synchronize, pre_hook_scope, remove):

    def evict_cache():
        buffer.sum()
        synchronize()

    try:
        with pre_hook_scope(evict_cache) if buffer is not None and pre_hook_scope else nullcontext():
            yield
    except BaseException:
        with result_scope(remove):
            raise


def benchmark_attempts(measure, funcs, names, options, *, warmup, active, prof_root, keep_res=False,
                       clear_l2_cache=False, pre_hook_scope=None, report_sink=None, timing_sink=None):
    from ._benchmark_quality import central_50_mean

    if options.quality_check and active < 2:
        raise ValueError("quality_check requires active >= 2")
    if options.quality_check:
        from ._benchmark_quality import evaluate_quality
    with timing_stage(timing_sink, "slow_filter"):
        indices, durations = (filter_slow_configs(funcs, options) if options.filter_slow_configs else
                              (list(range(len(funcs))), {}))
    with timing_stage(timing_sink, "calibration"):
        counts = calibrate_counts(funcs, indices, durations, options, active, clear_l2_cache, pre_hook_scope)
    costs = [float("inf")] * len(funcs)
    best_failures = [[] for _ in funcs]
    best_reports = [None] * len(funcs) if report_sink is not None else None
    root = Path(prof_root)
    root.mkdir(parents=True, exist_ok=True)

    def collect(selected, reports):
        execution_state = {"failed": False}
        directory = tempfile.mkdtemp(prefix="npu_bench_", dir=root)
        with result_scope(lambda: shutil.rmtree(directory) if not keep_res else None):
            try:
                return measure([funcs[index] for index in selected], [names[index] for index in selected], warmup,
                               active, directory, _active_counts=[counts[index] for index in selected],
                               _execution_state=execution_state,
                               **({"_report_sink": reports.extend} if reports is not None else {}))
            except (RuntimeError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                if execution_state["failed"]:
                    raise
                raise ProfilerAcquisitionError(str(exc)) from exc

    pending = indices
    last_acquisition_error = None
    with timing_stage(timing_sink, "measurements"):
        for attempt in range(options.max_retries + 1):
            selected = pending
            log_stage(options.log_level, "Profiler measurement", f"Attempt {attempt + 1}: {len(selected)} candidates")
            log_stage(
                options.log_level, "Measurement settings", f"configs={selected}, warmup={warmup}, "
                f"active_per_config={[counts[index] for index in selected]}", detailed=True)
            reports = [] if report_sink is not None else None
            try:
                samples = collect(selected, reports)
            except ProfilerAcquisitionError as exc:
                last_acquisition_error = exc
                log_stage(options.log_level, "Profiler acquisition failed", str(exc), detailed=True)
                continue
            pending = []
            for position, (index, (times, durations)) in enumerate(zip(selected, samples)):
                cost = central_50_mean(durations) / 1000
                metrics, failures = evaluate_quality(times, durations) if options.quality_check else ({}, [])
                if cost < costs[index]:
                    costs[index], best_failures[index] = cost, failures
                    if best_reports is not None:
                        best_reports[index] = report_safely(
                            lambda: replace(reports[position], attempt=attempt + 1, quality_failures=tuple(failures),
                                            quality_checked=options.quality_check, quality_metrics=metrics))
                log_stage(
                    options.log_level, "Config measurement", f"config={index}, attempt={attempt + 1}, "
                    f"active={counts[index]}, score_ms={cost:.8g}, mean_ms={np.mean(durations) / 1000:.8g}, "
                    f"metrics={metrics}, failures={failures}", detailed=True)
                if failures:
                    pending.append(index)
            log_stage(
                options.log_level, "Quality/acquisition result", f"{len(selected) - len(pending)}/"
                f"{len(selected)} candidates passed; retry candidates={len(pending)}")
            if not pending:
                break
    if not any(math.isfinite(value) for value in costs):
        raise RuntimeError("NPU benchmark: no usable profiler measurements") from last_acquisition_error
    best = min(range(len(costs)), key=costs.__getitem__)
    if best_failures[best]:
        report_safely(lambda: print("NPU autotune: the fastest retained measurement failed quality checks; "
                                    "the autotuner is not confident that this is the fastest configuration. "
                                    "The selected score is unchanged. " + "; ".join(best_failures[best])))
    if report_sink is not None:
        report_safely(lambda: report_sink(best_reports))
    return costs


def bench_npu(profile, funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name, *, options=None,
              pre_hook_scope=None, report_sink=None, timing_sink=None):
    options = resolve_options() if options is None else options
    funcs = funcs if isinstance(funcs, list) else [funcs]
    if not funcs:
        return []
    warmup = warmup if options.warmup is None else options.warmup
    active = active if options.active is None else options.active
    clear_l2_cache = clear_l2_cache if options.cache_mode is None else options.cache_mode == "cold"
    if options.quality_check and active < 2:
        raise ValueError("quality_check requires active >= 2")
    log_stage(
        options.log_level, "Benchmark settings", f"cache={'cold' if clear_l2_cache else 'hot'}, "
        f"warmup={warmup}, active={active}, quality_check={options.quality_check}", detailed=True)
    diagnostics = {"_report_sink": report_sink} if report_sink is not None else {}
    if not (options.quality_check or options.max_retries or options.measure_budget_ms is not None
            or options.filter_slow_configs):
        log_stage(options.log_level, "Profiler measurement", f"Attempt 1: {len(funcs)} candidates")
        with timing_stage(timing_sink, "measurements"):
            return profile(funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name,
                           _pre_hook_scope=pre_hook_scope, **diagnostics)

    names = ([target_kernel_name] * len(funcs)
             if target_kernel_name is None or isinstance(target_kernel_name, str) else list(target_kernel_name))
    if len(names) != len(funcs):
        raise ValueError("Provide one target kernel name per function")

    def measure(callables, kernel_names, warmup_count, active_count, directory, **kwargs):
        return profile(callables, warmup_count, active_count, clear_l2_cache, directory, True, kernel_names,
                       _return_samples=True, _pre_hook_scope=pre_hook_scope, **kwargs)

    root = prof_dir if prof_dir is not None else Path(runtime.cache.get_home_dir()) / ".triton" / "profile_results"
    costs = benchmark_attempts(measure, funcs, names, options, warmup=warmup, active=active, prof_root=root,
                               keep_res=keep_res, clear_l2_cache=clear_l2_cache, pre_hook_scope=pre_hook_scope,
                               report_sink=report_sink, timing_sink=timing_sink)
    return costs[0] if len(funcs) == 1 else costs
