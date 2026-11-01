"""Test-branch selection IDs, cross-method reports and repeated-autotune CSV statistics."""

import csv
import hashlib
import io
import json
import math
import os
import pprint
import re
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from statistics import fmean

from ._npu_benchmark import NpuMeasurementReport
from ._npu_benchmark import print_timing_report as print_timing_report

REPORT_ENV = "TRITON_NPU_BENCH_REPORT_BEST_CONFIG"
TIMING_ENV = "TRITON_AUTOTUNE_REPORT_TIMING"
RUNS_ENV = "TRITON_AUTOTUNE_RUNS"
CSV_DIR_ENV = "TRITON_AUTOTUNE_CSV_DIR"
TIMING_STAGES = ("generation_pruning", "compilation", "slow_filter", "calibration", "measurements")


@dataclass(frozen=True)
class ScoreReport:
    score: object
    custom: bool
    benchmark_method: str | None = None
    warmup: int | None = None
    active: int | None = None
    cache_mode: str | None = None
    quantiles: tuple | None = None


def _canonical(value):
    """Stable diagnostic data, without Python hashes or object addresses."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        value = float(value)
        return value if math.isfinite(value) else {"float": str(value)}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: _canonical(item) for key, item in value.items()}
        items = [(_canonical(key), _canonical(item)) for key, item in value.items()]
        return {"mapping": sorted(items, key=lambda pair: json.dumps(pair[0], sort_keys=True))}
    if isinstance(value, (list, tuple)):
        items = [_canonical(item) for item in value]
        return items if isinstance(value, list) else {"tuple": items}
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item) for item in value]
        return {"set": sorted(items, key=lambda item: json.dumps(item, sort_keys=True))}
    rendered = str(value)
    if callable(value) or re.search(r"\bat 0x[0-9a-fA-F]+\b", rendered):
        raise ValueError(f"Cannot create a stable config ID for {type(value).__qualname__}")
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "value": rendered}


def _fingerprint(value):
    encoded = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _config_parameters(config, *, include_ubtune=True):
    parameters = dict(vars(config))
    # Hooks are still printed in full; their object identity is not a config ID.
    parameters.pop("pre_hook", None)
    if not include_ubtune:
        parameters.pop("ubtune_cfg", None)
    return parameters


def snapshot_candidates(configs, key):
    ids = [_fingerprint(_config_parameters(config, include_ubtune=False)) for config in configs]
    # The NPU measurement policy is not part of the input problem's identity.
    key = [item for item in key if not (isinstance(item, tuple) and item and item[0] == "npu-benchmark")]
    return {
        "key": _canonical(key),
        "requested_method": os.getenv("TRITON_BENCH_METHOD", "default").lower(),
        "candidate_ids": ids,
        "positions": {id(config): index
                      for index, config in enumerate(configs)},
        "scored_ids": None,
    }


def record_scored_candidates(selection, configs):
    ids, positions = selection["candidate_ids"], selection["positions"]
    selection["scored_ids"] = [
        ids[positions[id(config)]] if id(config) in positions else _fingerprint(
            _config_parameters(config, include_ubtune=False)) for config in configs
    ]


def resolve_comparison_options():
    value = os.getenv(RUNS_ENV, "1").strip()
    if not value.isascii() or not value.isdecimal() or int(value) < 1:
        raise ValueError(f"{RUNS_ENV} must be a positive integer, got {value!r}")
    directory = os.getenv(CSV_DIR_ENV)
    if directory is not None and not directory.strip():
        raise ValueError(f"{CSV_DIR_ENV} must be a non-empty directory path")
    return int(value), directory


class AutotuneComparison:
    """Statistics of independent tuning decisions, never raw kernel samples."""

    def __init__(self, runs, directory):
        self.runs = runs
        self.directory = directory
        self.key = None
        self.selection = None
        self.rows = []
        self.completed = 0
        self.cache_hit = False
        self.timings = []

    def prepare(self, configs, key, cache_miss):
        self.cache_hit = not cache_miss
        if self.selection is not None and key != self.key:
            raise RuntimeError("Autotune input key changed between comparison runs")
        if not cache_miss:
            if self.completed:
                raise RuntimeError("Unexpected autotune cache hit during comparison")
            return None
        selection = snapshot_candidates(configs, key)
        if len(selection["positions"]) != len(configs):
            raise ValueError("CSV comparison requires distinct Config objects for each candidate")
        if self.selection is None:
            self.key = key
            self.selection = selection
            self.rows = [{"config": _canonical(_config_parameters(config)), "observed": False, "scores": [], "wins": 0}
                         for config in configs]
        elif selection["candidate_ids"] != self.selection["candidate_ids"]:
            raise RuntimeError("Autotune candidate parameters or order changed between comparison runs")
        selection["run_index"] = self.completed + 1
        selection["total_runs"] = self.runs
        return selection

    def record(self, selection, winner, timings):
        positions = selection["positions"]

        def row_for(config):
            position = positions.get(id(config))
            if position is None:
                raise RuntimeError("Scored config is absent from the original candidate list")
            row = self.rows[position]
            parameters = _canonical(_config_parameters(config))
            if row["observed"] and parameters != row["config"]:
                raise RuntimeError(f"Effective parameters changed for config {position + 1} between comparison runs")
            row["config"] = parameters
            row["observed"] = True
            return row

        for config, score in (timings or {}).items():
            row = row_for(config)
            # Built-in events return median/20%/80%; profiler paths return a scalar.
            score = score[0] if isinstance(score, (list, tuple)) else score
            score = float(score)
            if math.isfinite(score):
                row["scores"].append(score * 1000)
        row_for(winner)["wins"] += 1
        self.completed += 1

    def record_timing(self, total, durations):
        self.timings.append({
            "autotune": total, **{stage: durations.get(stage, 0)
                                  for stage in TIMING_STAGES}, "other":
            max(0, total - sum(durations.get(stage, 0) for stage in TIMING_STAGES))
        })

    def timing_summary(self):
        if not self.timings or len(self.timings) != self.completed:
            raise RuntimeError("Autotune comparison timing count does not match completed decisions")
        summary = {}
        for stage in ("autotune", *TIMING_STAGES, "other"):
            if stage == "compilation":
                summary["compilation_s"] = self.timings[0][stage]
            else:
                summary[f"mean_{stage}_s"] = fmean(row[stage] for row in self.timings)
        summary["first_autotune_s"] = self.timings[0]["autotune"]
        repeats = self.timings[1:]
        summary["mean_repeat_autotune_s"] = fmean(row["autotune"] for row in repeats) if repeats else None
        return summary

    def write_csv(self, function_name):
        summary = self.timing_summary()
        timing_values = [f"{value:.12g}" if value is not None else "" for value in summary.values()]
        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        kernel = re.sub(r"[^A-Za-z0-9_.-]", "_", function_name)[:80]
        method = re.sub(r"[^A-Za-z0-9_.-]", "_", self.selection["requested_method"])
        key_id = _fingerprint(self.selection["key"])[:12]
        path = directory / f"{kernel}.{method}.{key_id}.{time.time_ns()}.{os.getpid()}.csv"
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("config_index", "config", "mean_time_us", "best_time_us", "max_time_us", "best_count",
                             "total_runs", *summary))
            for index, row in enumerate(self.rows, 1):
                scores = row["scores"]
                times = (fmean(scores), min(scores), max(scores)) if scores else (None, None, None)
                parameters = json.dumps(row["config"], sort_keys=True, allow_nan=False)
                formatted_times = [f"{value:.12g}" if value is not None else "" for value in times]
                writer.writerow((index, parameters, *formatted_times, row["wins"], self.completed, *timing_values))
        return path.resolve()


def _decision_record(function_name, config, number, measurement, selection):
    method, estimator, score = "unavailable", "unavailable", None
    if selection is not None and selection["scored_ids"] is None:
        method = "unmeasured"
    if isinstance(measurement, NpuMeasurementReport):
        from .testing import _npu_bench_options
        bounds = _npu_bench_options()[5:7]
        estimator = "mean" if bounds == (0.0, 1.0) else "trimmed_mean"
        method, score = "npu", measurement.score_ms
    elif isinstance(measurement, ScoreReport):
        method = measurement.benchmark_method or ("custom" if measurement.custom else "default")
        if method == "npu":
            bounds = measurement.quantiles
            estimator = "mean" if bounds == (0.0, 1.0) else "trimmed_mean"
            score = measurement.score
        elif not measurement.custom and isinstance(measurement.score, (list, tuple)) and len(measurement.score) == 3:
            estimator, score = "median", measurement.score[0]
        else:
            estimator = "custom_score" if measurement.custom else "returned_score"
    if score is not None:
        score = float(score)
        if not math.isfinite(score):
            score = str(score)
    record = {
        "kernel": function_name,
        "benchmark_method": method,
        "config_id": _fingerprint(_config_parameters(config)),
        "config": _canonical(_config_parameters(config)),
        "original_index": number,
        "score_kind": estimator,
        "score_ms": score,
    }
    if isinstance(measurement, NpuMeasurementReport) or isinstance(measurement, ScoreReport) and method == "npu":
        record["score_quantiles"] = bounds
    if selection is not None:
        ids = selection["candidate_ids"]
        position = selection["positions"].get(id(config))
        record.update(key=selection["key"], requested_method=selection["requested_method"],
                      candidate_id=ids[position] if position is not None else None, candidate_count=len(ids),
                      candidate_set_id=_fingerprint(sorted(ids)), candidate_order_id=_fingerprint(ids))
        scored = selection["scored_ids"]
        if scored is not None:
            record.update(scored_count=len(scored), scored_set_id=_fingerprint(sorted(scored)),
                          scored_order_id=_fingerprint(scored))
        if "run_index" in selection:
            record.update(run_index=selection["run_index"], total_runs=selection["total_runs"])
    return record


def resolve_flag(argument, name, environment, default=False):
    if argument is not None:
        if not isinstance(argument, bool):
            raise ValueError(f"{name} must be a boolean or None")
        return argument
    value = os.getenv(environment, str(int(default))).strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError(f"Invalid {environment}: {value!r}")
    return value in {"1", "true"}


def resolve_report_best_config(argument=None, default=False):
    return resolve_flag(argument, "report_best_config", REPORT_ENV, default)


def resolve_report_timing(argument=None, default=False):
    return resolve_flag(argument, "report_timing", TIMING_ENV, default)


def report_safely(action):
    """Optional diagnostics do not change scores or execution errors."""
    try:
        return action()
    except Exception:
        return None


@contextmanager
def timing_stage(sink, name):
    started = time.perf_counter() if sink is not None else None
    try:
        yield
    finally:
        if sink is not None:
            elapsed = time.perf_counter() - started
            report_safely(lambda: sink(name, elapsed))


def print_best_config_report(function_name, configs, config, measurement=None, *, selection=None):
    if selection is not None:
        position = selection["positions"].get(id(config))
        number = position + 1 if position is not None else None
        total = len(selection["candidate_ids"])
    else:
        number = next((i for i, candidate in enumerate(configs, 1) if candidate is config), None)
        total = len(configs)
    stream = io.StringIO()
    stream.write(f"Triton autotuning result for {function_name}\n" + "-" * 60 + "\n")
    stream.write(f"Selected config: {number}/{total} (1-based, before pruning)\n"
                 if number is not None else "Selected config: not in the original candidate list\n")
    decision = report_safely(lambda: _decision_record(function_name, config, number, measurement, selection))
    if decision is not None:
        stream.write(f"Benchmark path: {decision['benchmark_method']}\n")
        stream.write(f"Selected config ID: {decision['config_id']}\n")
        if selection is not None:
            stream.write(f"Candidate set ID: {decision['candidate_set_id']}\n")
            stream.write(f"Candidate order ID: {decision['candidate_order_id']}\n")
            stream.write(f"Autotune key: {json.dumps(decision['key'], sort_keys=True)}\n")
    else:
        stream.write("Selected config ID: unavailable\n")
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
        elif measurement.benchmark_method == "npu":
            stream.write(f"Selection score: {measurement.score * 1000:.12g} us ({measurement.score:.12g} ms)\n")
            stream.write(f"cache={measurement.cache_mode}, quantiles={measurement.quantiles}\n")
        elif isinstance(measurement.score, (list, tuple)) and len(measurement.score) == 3:
            median, low, high = measurement.score
            stream.write(f"Median: {median!r} ms; quantiles 20%/80%: {low!r}/{high!r} ms\n")
        else:
            stream.write(f"Returned score: {measurement.score!r}; median and quantiles unavailable.\n")
    else:
        if measurement.score_ms is not None:
            stream.write(f"Selection score: {measurement.score_ms * 1000:.12g} us "
                         f"({measurement.score_ms:.12g} ms)\n")
        else:
            stream.write("Selection score unavailable.\n")
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
    if decision is not None:
        stream.write("AUTOTUNE_DECISION " +
                     json.dumps(decision, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    print(stream.getvalue(), end="")
