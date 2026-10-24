"""Environment-controlled NPU measurements and diagnostics."""

from __future__ import annotations

import math
import os
import pprint
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext, suppress
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from statistics import fmean

import numpy as np
import triton.runtime as runtime

_perf_report_sink = ContextVar("npu_perf_report_sink", default=None)


@contextmanager
def capture_npu_reports(sink):
    """Capture retained measurement rows in this context, without extra profiling."""
    token = _perf_report_sink.set(sink)
    try:
        yield
    finally:
        _perf_report_sink.reset(token)


ENV_PREFIX = "TRITON_NPU_BENCH_"
TIMING_STAGES = ("generation_pruning", "compilation", "slow_filter", "calibration", "measurements")
COUNT_MINIMUMS = {
    "warmup": 0, "active": 1, "max_retries": 0, "batch_runs": 1, "stability_runs": 1, "calibration_runs": 2,
    "slow_config_rounds": 1
}


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
    profiler_mode: str = "summary"
    cache_mode: str | None = "hot"
    warmup: int | None = None
    active: int | None = 250
    lower_quantile: float = 0.25
    upper_quantile: float = 0.75
    quality_check: bool = True
    quality_check_all: bool = False
    stability_only: bool = False
    max_retries: int = 5
    batch_runs: int = 2
    stability_runs: int = 2
    stability_tolerance: float = 0.03
    measure_budget_ms: float | None = 10.0
    calibration_runs: int = 10
    filter_slow_configs: bool = True
    slow_config_rounds: int = 10
    slow_config_factor: float = 2.5
    slow_config_recheck_delay_s: float = 0.0
    log_level: str = "brief"
    report_best_config: bool = True
    report_timing: bool = True

    def effective_policy(self):
        enabled = {
            "log_level": False,
            "report_best_config": False,
            "report_timing": False,
            "quality_check_all": False,
            "stability_only": self.quality_check,
            "batch_runs": self.quality_check,
            "stability_runs": self.quality_check,
            "stability_tolerance": self.quality_check and self.stability_runs > 1,
            "calibration_runs": self.filter_slow_configs,
            "slow_config_rounds": self.filter_slow_configs,
            "slow_config_factor": self.filter_slow_configs,
            "slow_config_recheck_delay_s": self.filter_slow_configs and self.slow_config_rounds > 1,
        }
        return tuple(sorted((name, value) for name, value in asdict(self).items() if enabled.get(name, True)))

    @property
    def is_default(self):
        """Whether measurement policies leave the original benchmark behavior intact."""
        return (self.profiler_mode == "full" and self.cache_mode is None and self.warmup is None and self.active is None
                and self.lower_quantile == 0 and self.upper_quantile == 1
                and not (self.quality_check or self.max_retries or self.measure_budget_ms is not None
                         or self.filter_slow_configs))

    def cache_key(self):
        thresholds = ()
        if self.quality_check and not self.stability_only:
            from ._benchmark_quality import THRESHOLDS
            thresholds = tuple(sorted(THRESHOLDS.items()))
        return ("npu-benchmark", 14, self.effective_policy(), thresholds)


def resolve_options():
    """Resolve each NPU environment setting over its explicit default."""
    values = asdict(NpuBenchmarkOptions())
    positive = {"measure_budget_ms", "slow_config_factor"}
    for name, default in values.items():
        environment = "TRITON_AUTOTUNE_REPORT_TIMING" if name == "report_timing" else ENV_PREFIX + name.upper()
        value = os.getenv(environment)
        if value is None:
            continue
        try:
            if isinstance(default, bool):
                value = {"0": False, "1": True, "false": False, "true": True}[value.strip().lower()]
            elif name in COUNT_MINIMUMS:
                value = int(value)
                if value < COUNT_MINIMUMS[name]:
                    raise ValueError(f"expected >= {COUNT_MINIMUMS[name]}")
            elif name in positive or isinstance(default, float):
                value = float(value)
                if name == "measure_budget_ms" and value == 0:
                    values[name] = None
                    continue
                minimum = 1 if name == "slow_config_factor" else 0
                if not math.isfinite(value) or value < minimum or (name in positive and value == 0):
                    raise ValueError("invalid finite range")
            else:
                value = value.strip().lower()
                allowed = {
                    "cache_mode": {"hot", "cold"}, "log_level": {"off", "brief", "detailed"}, "profiler_mode":
                    {"summary", "full"}
                }[name]
                if value not in allowed:
                    raise ValueError(f"expected one of {sorted(allowed)}")
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Invalid {environment}: {value!r}") from exc
        values[name] = value
    if not 0 <= values["lower_quantile"] < values["upper_quantile"] <= 1:
        raise ValueError("TRITON_NPU_BENCH quantiles must satisfy 0 <= LOWER_QUANTILE < UPPER_QUANTILE <= 1")
    if values["quality_check"] and values["active"] is not None and values["active"] < 2:
        raise ValueError("quality_check requires active >= 2")
    if values["quality_check"] and values["stability_only"] and values["stability_runs"] < 2:
        raise ValueError("STABILITY_ONLY requires STABILITY_RUNS >= 2")
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
def timing_stage(durations, name):
    started = time.perf_counter() if durations is not None else None
    try:
        yield
    finally:
        record_timing(durations, name, started)


def record_timing(durations, name, started):
    if started is not None:
        elapsed = time.perf_counter() - started
        with suppress(Exception):
            durations[name] = durations.get(name, 0) + elapsed


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
    samples: tuple | None = field(default=None, repr=False)
    kernel_timings: tuple = ()
    kernel_rows: tuple = field(default=(), repr=False)
    all_configs_passed: bool = True


def aggregate_profile_rows(frame):
    metrics = {}
    for column in frame.columns:
        data = frame[column].to_numpy()
        absolute = re.search(r"(^|[^a-z])(id|timestamp)([^a-z]|$)|start\s*time|end\s*time", column.lower())
        if data.dtype.kind in "iuf":
            available = data[~np.isnan(data)] if absolute else data[np.isfinite(data)]
            count = len(available)
            value = (str(available[0]) if np.all(available == available[0]) else "varies") if count else "unavailable"
            metrics[column] = ((float(np.mean(available)), count, True) if count and not absolute else
                               (value, count, False))
            continue
        if len(data) and np.all(data == data[0]):
            text = str(data[0]).strip()
            if text.lower() not in {"", "nan", "n/a", "null", "none"}:
                try:
                    number = float(text) if not absolute else None
                except ValueError:
                    number = None
                metrics[column] = ((number, len(data), True) if number is not None and math.isfinite(number) else
                                   (text, len(data), False) if number is None else ("unavailable", 0, False))
                continue
        values = [
            str(value).strip()
            for value in frame[column]
            if str(value).strip().lower() not in {"", "nan", "n/a", "null", "none"}
        ]
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
    lines = [
        f"Triton autotuning result for {function_name}", "-" * 60,
        f"Selected config: {number}/{len(configs)} (1-based, before pruning)"
        if number is not None else "Selected config: not in the original candidate list"
    ]
    parameters = dict(vars(config))
    ub_config = parameters.pop("ubtune_cfg", None)
    lines.extend(["Full config parameters:", pprint.pformat(parameters, sort_dicts=False)])
    if ub_config is not None:
        lines.extend(["ubtune_cfg:", pprint.pformat(ub_config, sort_dicts=False)])
    if measurement is None:
        lines.append("NPU measurements unavailable for this selection; no extra benchmarking was run.")
    else:
        lines.extend([
            f"Selection score: {measurement.score_ms * 1000:.12g} us ({measurement.score_ms:.12g} ms)",
            f"Mean duration: {measurement.mean_ms * 1000:.12g} us ({measurement.mean_ms:.12g} ms)",
            f"cache={measurement.cache_mode}, warmup={measurement.warmup}, active={measurement.active}, "
            f"selected attempt={measurement.attempt}"
        ])
        quality = ("failed"
                   if measurement.quality_failures else "passed" if measurement.quality_checked else "not requested")
        lines.append(f"Quality checks: {quality}; metrics={measurement.quality_metrics}")
        if measurement.quality_failures:
            lines.append("Failure reasons: " + "; ".join(measurement.quality_failures))
        lines.append(f"Profiler aggregates: {measurement.sample_count} measured launches (warmup excluded)")
        for column, (value, count, numeric) in measurement.metrics.items():
            formatted = f"{value:.12g}" if numeric else value
            available = f" ({count}/{measurement.sample_count} available)" if count != measurement.sample_count else ""
            lines.append(f"  {column}: {'mean' if numeric else 'value'}={formatted}{available}")
    print("\n".join(lines))


def print_operation_report(name, measurement):
    print(f"NPU operation kernel breakdown for {name}\n" + "-" * 60)
    if measurement is None:
        print("Measurements unavailable.")
        return
    quality = (
        "failed" if measurement.quality_failures else "passed") if measurement.quality_checked else "not requested"
    print(f"cache={measurement.cache_mode}, warmup={measurement.warmup}, active={measurement.active}, "
          f"selected attempt={measurement.attempt}; operation quality: {quality}")
    for index, (kernel, score, mean, low, high) in enumerate(measurement.kernel_timings, 1):
        print(f"  {index}. {kernel}\n"
              f"     score={score:.6g} us; mean={mean:.6g} us; min={low:.6g} us; max={high:.6g} us")


def print_timing_report(function_name, started, durations, *, total=None):
    total = time.perf_counter() - started if total is None else total
    labels = ("Generation/pruning", "Compilation", "Slow filter", "Calibration", "Measurements (including retries)")
    lines = [f"Triton autotuning timing for {function_name}", "-" * 60, f"  Total: {total:.6f} s"]
    details = (("profiler", "Profiled launches/synchronization"), ("export", "CANN export/analysis"),
               ("csv", "CSV and aggregates"), ("quality", "Quality checks"))
    for stage, label in zip(TIMING_STAGES, labels):
        elapsed = durations.get(stage, 0)
        lines.append(f"  {label}: {elapsed:.6f} s")
        if stage == "measurements" and any(name in durations for name, _ in details):
            lines.extend(f"    {label}: {durations.get(name, 0):.6f} s" for name, label in details)
            remaining = max(0, elapsed - sum(durations.get(name, 0) for name, _ in details))
            lines.append(f"    Other measurement overhead: {remaining:.6f} s")
    lines.append(f"  Other: {max(0, total - sum(durations.get(name, 0) for name in TIMING_STAGES)):.6f} s")
    print("\n".join(lines))


def export_npu_summary(profiler_path, timings=None):
    """Export CANN summary CSVs without the TensorBoard analysis pipeline."""
    cann_path = next((path for path in Path(profiler_path).glob("PROF_*") if path.is_dir()), None)
    msprof = shutil.which("msprof")
    if cann_path is None or msprof is None:
        raise ProfilerAcquisitionError("CANN profiling data or msprof executable was not found")
    relative = "tools/profiler/profiler_tool/analysis/msprof/msprof.py"
    script = next((parent / relative for parent in Path(msprof).resolve().parents if (parent / relative).is_file()),
                  None)
    if script is None:
        raise ProfilerAcquisitionError("CANN msprof.py summary exporter was not found")
    with timing_stage(timings, "export"):
        result = subprocess.run(
            [sys.executable,
             str(script), "export", "summary", "-dir",
             str(cann_path), "--format", "csv"], capture_output=True, text=True)
    if result.returncode:
        raise ProfilerAcquisitionError(f"CANN summary export failed: {result.stderr or result.stdout}")


def _collect_prof_result(base_dir, funcs, num_warmup, num_active, target_kernel_name=None, clear_l2_cache=False,
                         _return_samples=False, _capture_report=False, _quantiles=(0.0, 1.0), _aggregate_calls=False):
    """
    Collect per-function measurements, reading each profiler CSV once.

    Accept kernel_details.csv or native CANN op_summary CSVs. Summary column
    names are normalized to the same schema; split files are merged by start time.

    Split rows in profiler execution order into equal num_warmup + num_active
    blocks. Skip each function's warmup rows and collect its num_active rows.
    Scores and means are in milliseconds;
    optional sample arrays retain the CSV's microsecond units.

    :param base_dir: the profiler output directory
    :type base_dir: str
    :param funcs: functions in profiler execution order
    :type funcs: List[Callable]
    :param num_warmup: warmup rows to skip for each function
    :type num_warmup: int
    :param num_active: shared measured-row count per function
    :type num_active: int
    :param target_kernel_name: one shared target name, or one name per function;
        None entries mean unknown names. Filter by names only when all are known.
    :type target_kernel_name: Optional[Union[str, List[Optional[str]]]]
    :param clear_l2_cache: exclude ReduceSum rows when L2 eviction is enabled
    :type clear_l2_cache: bool
    :param _return_samples: include validated, untrimmed start times and durations
        for quality evaluation; missing CSV then raises instead of returning None
    :type _return_samples: bool
    :param _capture_report: aggregate measured CSV columns for the winner report
    :type _capture_report: bool
    :param _quantiles: lower and upper boundaries of the sorted durations averaged
        for scoring; (0, 1) averages all samples. Indices are rounded down,
        retaining at least one sample. Quality samples are not trimmed.
    :type _quantiles: Tuple[float, float]
    :param _aggregate_calls: sum a fixed kernel sequence into one sample per operation call
    :return: one measurement record per function, or None entries if CSV is missing
        and _return_samples is False; benchmark_result converts None to inf
    :rtype: List[Optional[NpuMeasurementReport]]
    :raises ValueError: invalid counts or number of target names
    :raises ProfilerResultMismatchError: filtered target-row count does not match
    :raises ProfilerAcquisitionError: incomplete rows, unexpected target order,
        invalid requested samples, or missing CSV when _return_samples is True
    """
    import pandas as pd

    files = list(Path(base_dir).rglob("kernel_details.csv"))[:1] or sorted(Path(base_dir).rglob("op_summary*.csv"))
    num_funcs = len(funcs)
    if not files:
        if _return_samples:
            raise ProfilerAcquisitionError("NPU profiler CSV kernel_details.csv or op_summary was not found")
        return [None] * num_funcs

    columns = {
        "Op Name": "Name", "OP Type": "Type", "Task Start Time(us)": "Start Time(us)", "Task Duration(us)":
        "Duration(us)", "Task Type": "Accelerator Core"
    }
    frames = [pd.read_csv(path).rename(columns=columns, copy=False) for path in files]
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True).sort_values(
        "Start Time(us)", kind="stable")
    filter_df = df[~df["Type"].str.contains(r"^ReduceSum$", case=False, na=False)] if clear_l2_cache else df
    names = ([target_kernel_name] * num_funcs
             if target_kernel_name is None or isinstance(target_kernel_name, str) else list(target_kernel_name))
    if all(name is not None for name in names):
        selected = filter_df["Name"].isin(names)
        if not selected.all():
            filter_df = filter_df[selected]

    if len(names) != num_funcs:
        raise ValueError("Provide one kernel name per function")
    if num_warmup < 0 or num_active < 1:
        raise ValueError("Warmup must be nonnegative and active positive")
    expected_rows = num_funcs * (num_warmup + num_active)

    source_df = filter_df
    kernel_timings = [()] * num_funcs
    if _aggregate_calls:
        # A fixed operation must launch the same ordered kernel sequence on every call.
        if len(filter_df) == 0 or len(filter_df) % expected_rows:
            raise ProfilerAcquisitionError("Operation kernel count changed between calls")
        width = len(filter_df) // expected_rows
        sequence = filter_df["Name"].to_numpy().reshape(expected_rows, width)
        if not np.all(sequence == sequence[0]):
            raise ProfilerAcquisitionError("Operation kernel sequence changed between calls")
        all_durations = filter_df["Duration(us)"].to_numpy(dtype=float).reshape(expected_rows, width)
        if _capture_report:
            lower, upper = int(num_active * _quantiles[0]), int(num_active * _quantiles[1])
            for index in range(num_funcs):
                offset = index * (num_warmup + num_active) + num_warmup
                measured = all_durations[offset:offset + num_active]
                scores = np.sort(measured, axis=0)[lower:max(lower + 1, upper)].mean(axis=0)
                kernel_timings[index] = tuple(
                    (str(name), float(score), float(mean), float(low), float(high))
                    for name, score, mean, low, high in zip(sequence[0], scores, measured.mean(
                        axis=0), measured.min(axis=0), measured.max(axis=0)))
        filter_df = pd.DataFrame({
            "Name": [" + ".join(map(str, sequence[0]))] * expected_rows,
            "Type":
            "Operation",
            "Start Time(us)":
            filter_df["Start Time(us)"].to_numpy().reshape(expected_rows, width)[:, 0],
            "Duration(us)":
            all_durations.sum(axis=1),
        })

    if target_kernel_name is not None and len(filter_df) != expected_rows:
        raise ProfilerResultMismatchError(target_kernel_name, expected_rows, len(filter_df))

    results = []
    for index in range(num_funcs):
        offset = index * (num_warmup + num_active) + num_warmup
        measured = filter_df.iloc[offset:offset + num_active]
        durations = measured["Duration(us)"].to_numpy(dtype=float)
        if len(durations) != num_active:
            raise ProfilerAcquisitionError(f"Incomplete profiler rows for config {index}")
        if names[index] is not None and not measured["Name"].eq(names[index]).all():
            raise ProfilerAcquisitionError(f"Unexpected profiler target order for config {index}")
        ranked = np.sort(durations) if _quantiles != (0.0, 1.0) else durations
        lower, upper = int(num_active * _quantiles[0]), int(num_active * _quantiles[1])
        cost = float(np.mean(ranked[lower:max(lower + 1, upper)])) / 1000
        samples = None
        if _return_samples:
            times = measured["Start Time(us)"].to_numpy(dtype=float)
            if (not np.all(np.isfinite(times)) or not np.all(np.isfinite(durations)) or np.any(durations <= 0)
                    or np.any(np.diff(times) <= 0)):
                raise ProfilerAcquisitionError(f"Invalid profiler samples for config {index}")
            samples = (times, durations)
        metrics = report_safely(lambda: aggregate_profile_rows(measured)) if _capture_report else {}
        kernel_rows = ()
        if _perf_report_sink.get() is not None:
            if _aggregate_calls:
                start = index * (num_warmup + num_active) * width + num_warmup * width
                rows = source_df.iloc[start:start + num_active * width]
                kernel_rows = tuple((str(sequence[0, column]), tuple(rows.iloc[column::width].to_dict("records")))
                                    for column in range(width))
            else:
                kernel_rows = ((str(measured["Name"].iloc[0]), tuple(measured.to_dict("records"))), )
        results.append(
            NpuMeasurementReport(metrics, num_active, num_warmup, num_active, "cold" if clear_l2_cache else "hot",
                                 float(np.mean(durations)) / 1000, cost, samples=samples,
                                 kernel_timings=kernel_timings[index], kernel_rows=kernel_rows))
    return results


def filter_slow_configs(funcs, options, clear_l2_cache, pre_hook_scope):
    """Retain minima; restart suspect rounds when another candidate becomes fastest."""
    indices = list(range(len(funcs)))
    if len(funcs) < 2:
        return indices, {}
    log_stage(options.log_level, "Slow filter", f"Round 1/{options.slow_config_rounds}: {len(funcs)} candidates")
    with event_timer(clear_l2_cache, pre_hook_scope) as measure:
        durations = {index: measure(fn, options.calibration_runs) for index, fn in enumerate(funcs)}
        winner = min(indices, key=durations.get)
        round_index = 1
        while True:
            threshold = durations[winner] * options.slow_config_factor
            suspects = [index for index in indices if durations[index] > threshold]
            log_stage(options.log_level, "Slow filter",
                      f"Further rounds needed: {len(suspects)}/{len(funcs)} candidates")
            if not suspects or round_index >= options.slow_config_rounds:
                break
            if options.slow_config_recheck_delay_s:
                time.sleep(options.slow_config_recheck_delay_s)
            log_stage(options.log_level, "Slow filter", f"Round {round_index + 1}/{options.slow_config_rounds}: "
                      f"{len(suspects)} suspects")
            previous_winner = winner
            for index in suspects:
                durations[index] = min(durations[index], measure(funcs[index], options.calibration_runs))
                if durations[index] < durations[winner]:
                    winner = index
            round_index = 1 if winner != previous_winner else round_index + 1
            if winner != previous_winner:
                log_stage(options.log_level, "Slow filter", f"New fastest config: {winner}; restarting rounds")
    kept = [index for index in indices if index not in suspects]
    log_stage(options.log_level, "Slow filter", f"{len(kept)}/{len(funcs)} candidates retained")
    log_stage(options.log_level, "Slow filter settings", f"event_min_ms={durations}, threshold_ms={threshold}",
              detailed=True)
    return kept, durations


@contextmanager
def event_timer(clear_l2_cache, pre_hook_scope):
    """Short event estimates under the same cache policy as profiler measurements."""
    import torch

    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    end.record()
    torch.npu.synchronize()
    buffer = runtime.driver.active.get_empty_cache_for_benchmark().float() if clear_l2_cache else None
    scoped = buffer is not None and pre_hook_scope is not None
    event_pairs = [(start, end)]

    def prepare():
        if buffer is not None:
            buffer.sum()
            torch.npu.synchronize()
        start.record()

    def measure(fn, count):
        nonlocal start

        while len(event_pairs) < count:
            event_pairs.append((torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)))
        pairs = event_pairs[:count]
        fn()
        torch.npu.synchronize()
        for start, end in pairs:
            if not scoped:
                prepare()
            fn()
            end.record()
        torch.npu.synchronize()
        return min(begin.elapsed_time(end) for begin, end in pairs)

    try:
        with pre_hook_scope(prepare) if scoped else nullcontext():
            yield measure
    finally:
        buffer = None


@contextmanager
def result_scope(remove):
    """Keep a primary failure if cleanup also fails."""
    try:
        yield
    except BaseException:
        with suppress(Exception):
            remove()
        raise
    else:
        remove()


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


def benchmark_result(measurements, report_sink):
    if report_sink is not None:
        report_safely(lambda: report_sink(
            [record if record is None or record.metrics is not None else None for record in measurements]))
    costs = [record.score_ms if record is not None else float("inf") for record in measurements]
    return costs[0] if len(costs) == 1 else costs


def bench_npu(profile, funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name, *, options=None,
              pre_hook_scope=None, report_sink=None, timings=None, config_numbers=None):
    """Measure candidates in batches and confirm their minimum scores."""
    options = resolve_options() if options is None else options
    funcs = funcs if isinstance(funcs, list) else [funcs]
    if not funcs:
        return []
    active = active if options.active is None else options.active
    warmup = (active + 9) // 10 if options.warmup is None else options.warmup
    clear_l2_cache = clear_l2_cache if options.cache_mode is None else options.cache_mode == "cold"
    options = replace(options, active=active, cache_mode="cold" if clear_l2_cache else "hot")
    profile = partial(profile, _quantiles=(options.lower_quantile, options.upper_quantile),
                      _profiler_mode=options.profiler_mode)
    if timings is not None:
        profile = partial(profile, _timings=timings)
    if options.quality_check and active < 2:
        raise ValueError("quality_check requires active >= 2")
    log_stage(
        options.log_level, "Benchmark settings", f"cache={'cold' if clear_l2_cache else 'hot'}, "
        f"profiler_mode={options.profiler_mode}, warmup={warmup}, active={active}, quality_check={options.quality_check}, "
        f"quality_check_all={options.quality_check_all}, "
        f"stability_only={options.stability_only}, "
        f"batch_runs={options.batch_runs}, stability_runs={options.stability_runs}, "
        f"stability_tolerance={options.stability_tolerance}, "
        f"quantiles=({options.lower_quantile}, {options.upper_quantile})", detailed=True)
    if not (options.quality_check or options.max_retries or options.measure_budget_ms is not None
            or options.filter_slow_configs):
        log_stage(options.log_level, "Profiler measurement", f"Batch 1: {len(funcs)} candidates, 1 series each")
        with timing_stage(timings, "measurements"):
            measurements = profile(funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name,
                                   _pre_hook_scope=pre_hook_scope, _capture_report=report_sink is not None)
        return benchmark_result(measurements, report_sink)

    names = ([target_kernel_name] * len(funcs)
             if target_kernel_name is None or isinstance(target_kernel_name, str) else list(target_kernel_name))
    if len(names) != len(funcs):
        raise ValueError("Provide one target kernel name per function")
    indices = list(range(len(funcs)))
    with timing_stage(timings, "slow_filter"):
        if options.filter_slow_configs:
            indices, _ = filter_slow_configs(funcs, options, clear_l2_cache, pre_hook_scope)
    measurements = _profile_with_retries(profile, funcs, names, indices, options, prof_dir, keep_res, pre_hook_scope,
                                         capture_report=report_sink is not None, timings=timings,
                                         config_numbers=config_numbers)
    return benchmark_result(measurements, report_sink)


def _profile_with_retries(profile, funcs, names, indices, options, prof_dir, keep_res, pre_hook_scope, capture_report,
                          timings, config_numbers=None):
    if options.quality_check:
        from ._benchmark_quality import evaluate_quality
        if options.stability_only:
            evaluate_quality = partial(evaluate_quality, stability_only=True)
        if options.quality_check_all:
            evaluate_quality = partial(evaluate_quality, check_all=True)
    best_reports = [None] * len(funcs)
    batch_runs = options.batch_runs if options.quality_check else 1
    stability_runs = options.stability_runs if options.quality_check else 1
    histories = {index: [] for index in indices} if stability_runs > 1 else {}
    active, retry_active = options.active, None
    unconfirmed = "stability not confirmed"
    root = Path(prof_dir) if prof_dir is not None else Path(
        runtime.cache.get_home_dir()) / ".triton" / "profile_results"
    root.mkdir(parents=True, exist_ok=True)

    def collect(selected):
        execution_state = {"failed": False}
        directory = tempfile.mkdtemp(prefix="npu_bench_", dir=root)
        with result_scope(lambda: shutil.rmtree(directory) if not keep_res else None):
            try:
                return profile([funcs[index] for index in selected], warmup, active, options.cache_mode == "cold",
                               directory, True, [names[index] for index in selected], _return_samples=True,
                               _pre_hook_scope=pre_hook_scope, _execution_state=execution_state,
                               _capture_report=capture_report)
            except (RuntimeError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                if execution_state["failed"]:
                    raise
                raise ProfilerAcquisitionError(str(exc)) from exc

    pending = indices
    last_acquisition_error = None
    with timing_stage(timings, "measurements"):
        for batch in range(max(1, options.max_retries)):
            selected = pending
            sequence = selected * batch_runs
            warmup = (active + 9) // 10 if options.warmup is None else options.warmup
            log_stage(options.log_level, "Profiler measurement", f"Batch {batch + 1}: {len(selected)} candidates, "
                      f"{batch_runs} series each")
            log_stage(options.log_level, "Measurement settings", f"configs={selected}, warmup={warmup}, "
                      f"active={active}", detailed=True)
            try:
                measurements = collect(sequence)
            except ProfilerAcquisitionError as exc:
                last_acquisition_error = exc
                log_stage(options.log_level, "Profiler acquisition failed", str(exc), detailed=True)
                continue
            if retry_active is None:
                retry_active = active
                if options.measure_budget_ms is not None:
                    duration = min(record.mean_ms for record in measurements)
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError(f"Invalid NPU profiler budget estimate: {duration}")
                    required = options.measure_budget_ms / duration
                    if not math.isfinite(required):
                        raise ValueError("Unrepresentable NPU benchmark sample count")
                    retry_active = max(active, math.ceil(required))
                    log_stage(
                        options.log_level, "Retry budget settings", f"budget_ms={options.measure_budget_ms}, "
                        f"first_profile_min_mean_ms={duration:.8g}, active={retry_active}", detailed=True)
            pending = []
            passed = set()
            for position, (index, measurement) in enumerate(zip(sequence, measurements)):
                attempt = batch * batch_runs + position // len(selected) + 1
                with timing_stage(timings, "quality"):
                    metrics, failures = evaluate_quality(*measurement.samples) if options.quality_check else ({}, [])
                record = replace(measurement, samples=None, attempt=attempt, quality_failures=tuple(failures),
                                 quality_checked=options.quality_check, quality_metrics=metrics)
                if measurement.score_ms < getattr(best_reports[index], "score_ms", float("inf")):
                    best_reports[index] = record
                if histories:
                    if not failures:
                        histories[index].append(record)
                elif not failures:
                    passed.add(index)
                log_stage(
                    options.log_level, "Config measurement", f"config={index}, batch={batch + 1}, attempt={attempt}, "
                    f"active={active}, score_ms={measurement.score_ms:.8g}, mean_ms={measurement.mean_ms:.8g}, "
                    f"metrics={metrics}, failures={failures}", detailed=True)
            measurements.clear()
            if histories:
                for index in selected:
                    with timing_stage(timings, "quality"):
                        best = best_reports[index]
                        history = histories[index]
                        others = sorted((item for item in history if item is not best), key=lambda item: item.mean_ms)
                        confirmed, spread = None, float("inf")
                        # Every group includes the minimum; only quality-passing history can confirm it.
                        for start in range(len(others) - stability_runs + 2):
                            group = [best, *others[start:start + stability_runs - 1]]
                            means = [item.mean_ms for item in group]
                            ratio = (max(means) / min(means) if all(math.isfinite(mean) and mean > 0
                                                                    for mean in means) else float("inf"))
                            spread = min(spread, ratio - 1)
                            if ratio <= 1 + options.stability_tolerance:
                                clean = min((item for item in group if not item.quality_failures), key=lambda item:
                                            (item.score_ms, item.attempt))
                                if confirmed is None or clean.score_ms < confirmed.score_ms:
                                    confirmed = replace(
                                        clean, quality_metrics={
                                            **clean.quality_metrics, "stability_relative_spread": ratio - 1
                                        })
                        best.quality_metrics["stability_relative_spread"] = spread
                        retry = confirmed is None
                        if retry:
                            pending.append(index)
                        else:
                            best_reports[index] = confirmed
                            histories.pop(index)
                    log_stage(
                        options.log_level, "Minimum confirmation", f"config={index}, "
                        f"score_ms={best.score_ms:.8g}, mean_ms={best.mean_ms:.8g}, "
                        f"stability_relative_spread={spread:.8g}, candidate_passed={not retry}", detailed=True)
            else:
                pending = [index for index in selected if index not in passed]
            log_stage(
                options.log_level, "Quality/acquisition result", f"{len(selected) - len(pending)}/"
                f"{len(selected)} candidates passed; retry candidates={len(pending)}")
            if not pending:
                break
            active = retry_active
    if histories:
        for index in pending:
            if best_reports[index] is not None:
                best_reports[index] = replace(best_reports[index],
                                              quality_failures=best_reports[index].quality_failures + (unconfirmed, ))
    measured = [record for record in best_reports if record is not None and math.isfinite(record.score_ms)]
    if not measured:
        raise RuntimeError("NPU benchmark: no usable profiler measurements") from last_acquisition_error
    best = min(measured, key=lambda record: record.score_ms)
    if pending and options.quality_check:
        index = next(index for index, record in enumerate(best_reports) if record is best)
        winner = (f"config #{config_numbers[index]}" if config_numbers is not None and config_numbers[index] is not None
                  else f"benchmark candidate #{index + 1}")
        report_safely(
            lambda: print("\033[33mNot all configurations passed quality checks. "
                          f"Selected {winner}: lowest score {best.score_ms * 1000:.6g} us; "
                          f"selected attempt quality: {'failed' if best.quality_failures else 'pass'}.\033[0m"))
    if options.quality_check:
        best_reports = [
            replace(record, all_configs_passed=not pending) if record is not None else None for record in best_reports
        ]
    return best_reports
