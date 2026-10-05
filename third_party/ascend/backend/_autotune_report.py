"""Optional autotune diagnostics, independent of benchmark and cache policies."""

import io
import math
import os
import pprint
import re
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import fmean

REPORT_ENV = "TRITON_NPU_BENCH_REPORT_BEST_CONFIG"
TIMING_ENV = "TRITON_AUTOTUNE_REPORT_TIMING"
TIMING_STAGES = ("generation_pruning", "compilation", "slow_filter", "calibration", "measurements")


@dataclass(frozen=True)
class NpuMeasurementReport:
    metrics: dict
    sample_count: int
    warmup: int
    active: int
    cache_mode: str
    mean_ms: float
    attempt: int = 1
    quality_failures: tuple = ()
    quality_checked: bool = False
    quality_metrics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScoreReport:
    score: object
    custom: bool


def resolve_flag(argument, name, environment):
    if argument is not None:
        if not isinstance(argument, bool):
            raise ValueError(f"{name} must be a boolean or None")
        return argument
    value = os.getenv(environment, "0").strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError(f"Invalid {environment}: {value!r}")
    return value in {"1", "true"}


def resolve_report_best_config(argument=None):
    return resolve_flag(argument, "report_best_config", REPORT_ENV)


def resolve_report_timing(argument=None):
    return resolve_flag(argument, "report_timing", TIMING_ENV)


def report_safely(action):
    """Diagnostic failures must not affect measurement, retries or execution."""
    try:
        return action()
    except Exception as exc:
        try:
            warnings.warn(f"Autotune report unavailable: {exc}", RuntimeWarning, stacklevel=3)
        except Exception:
            pass


def log_stage(level, title, message, *, detailed=False):
    if level == "off" or (detailed and level != "detailed"):
        return
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


@contextmanager
def yellow_warning():
    """Color an emitted warning in a terminal while preserving warning filters."""
    original = warnings.showwarning

    def show(message, category, filename, lineno, file=None, line=None):
        stream = file if file is not None else sys.stderr
        if issubclass(category, RuntimeWarning) and getattr(stream, "isatty", lambda: False)():
            message = f"\033[33m{message}\033[0m"
        original(message, category, filename, lineno, file=file, line=line)

    warnings.showwarning = show
    try:
        yield
    finally:
        warnings.showwarning = original


def aggregate_profile_rows(columns, rows):
    """Aggregate measured rows; identifiers and absolute times remain text."""
    metrics = {}
    for column in columns:
        values = [
            str(row[column]).strip() for row in rows if row.get(column) is not None
            and str(row[column]).strip().lower() not in {"", "nan", "n/a", "null", "none"}
        ]
        if not values:
            metrics[column] = ("unavailable", 0, False)
            continue
        absolute = re.search(r"(^|[^a-z])(id|timestamp)([^a-z]|$)|start\s*time|end\s*time", column.lower())
        try:
            numbers = [float(value) for value in values] if not absolute else []
        except ValueError:
            numbers = []
        if numbers:
            numbers = [value for value in numbers if math.isfinite(value)]
            metrics[column] = (fmean(numbers), len(numbers), True) if numbers else ("unavailable", 0, False)
        else:
            metrics[column] = (values[0] if len(set(values)) == 1 else "varies", len(values), False)
    return metrics


def make_profile_report(columns, rows, warmup, active, clear_l2_cache, cost):
    return NpuMeasurementReport(aggregate_profile_rows(columns, rows), len(rows), warmup, active,
                                "cold" if clear_l2_cache else "hot", cost)


def print_best_config_report(function_name, configs, config, measurement=None):
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
        stream.write("Measurements unavailable for this selection; no extra benchmarking was run.\n")
    elif isinstance(measurement, ScoreReport):
        if measurement.custom:
            stream.write(f"Returned score: {measurement.score!r}\n")
        elif isinstance(measurement.score, (list, tuple)) and len(measurement.score) == 3:
            median, low, high = measurement.score
            stream.write(f"Median: {median!r} ms; quantiles 20%/80%: {low!r}/{high!r} ms\n")
        else:
            stream.write(f"Returned score: {measurement.score!r}; median and quantiles unavailable.\n")
    else:
        stream.write(f"Mean duration: {measurement.mean_ms * 1000:.12g} us ({measurement.mean_ms:.12g} ms)\n")
        stream.write(f"cache={measurement.cache_mode}, warmup={measurement.warmup}, "
                     f"active={measurement.active}, selected attempt={measurement.attempt}\n")
        quality = "passed" if measurement.quality_checked else "not requested"
        if measurement.quality_failures:
            quality = "failed"
            stream.write("Selected measurement failed quality checks: " + "; ".join(measurement.quality_failures) +
                         "\n")
        stream.write(f"Quality checks: {quality}; metrics={measurement.quality_metrics}\n")
        stream.write(f"Profiler aggregates: {measurement.sample_count} measured launches (warmup excluded)\n")
        for column, (value, count, numeric) in measurement.metrics.items():
            label = "mean" if numeric else "value"
            formatted = f"{value:.12g}" if numeric else value
            available = f" ({count}/{measurement.sample_count} available)" if count != measurement.sample_count else ""
            stream.write(f"  {column}: {label}={formatted}{available}\n")
    print(stream.getvalue(), end="")


def capture_profile_report(filter_df, time_cost, num_warmup, num_active, clear_l2_cache, report_sink):

    def capture():
        total = num_warmup + num_active
        reports = [
            make_profile_report(tuple(filter_df.columns),
                                filter_df.iloc[index * total + num_warmup:(index + 1) * total].to_dict("records"),
                                num_warmup, num_active, clear_l2_cache, cost) for index, cost in enumerate(time_cost)
        ]
        report_sink(reports)

    report_safely(capture)


def print_timing_report(function_name, started, durations):
    total = time.perf_counter() - started
    labels = ("Generation/pruning", "Compilation", "Slow filter", "Calibration", "Measurements (including retries)")
    lines = [f"Triton autotuning timing for {function_name}", "-" * 60, f"  Total: {total:.6f} s"]
    lines.extend(f"  {label}: {durations.get(stage, 0):.6f} s" for stage, label in zip(TIMING_STAGES, labels))
    lines.append(f"  Other: {max(0, total - sum(durations.values())):.6f} s")
    print("\n".join(lines))
