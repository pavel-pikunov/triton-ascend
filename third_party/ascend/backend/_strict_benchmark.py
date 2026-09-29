"""Strict profiling: conservative host prune, device calibration and quality retries."""

import csv
import json
import math
import os
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

from ._benchmark_quality import THRESHOLDS, central_50_mean, evaluate_quality

DEFAULTS = dict(warmup=25, active=150, measure_budget_ms=5.0, calibration_runs=10, max_attempts=5)


class ProfilerAcquisitionError(RuntimeError):
    pass


def validate_options(options):
    if not isinstance(options["clear_l2_cache"], bool):
        raise ValueError("clear_l2_cache must explicitly be True (cold) or False (hot)")
    for name in ("warmup", "active", "calibration_runs", "max_attempts"):
        value = options[name]
        minimum = 0 if name == "warmup" else 2 if name in ("active", "calibration_runs") else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    budget = options["measure_budget_ms"]
    if isinstance(budget, bool) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("measure_budget_ms must be finite and positive")
    return options


def options_from_env():
    mode = os.getenv("TRITON_NPU_STRICT_CACHE_MODE", "").strip().lower()
    if mode not in ("hot", "cold"):
        raise ValueError("npu-strict requires TRITON_NPU_STRICT_CACHE_MODE=hot or cold")
    options = {
        name: type(value)(os.getenv("TRITON_NPU_STRICT_" + name.upper(), value))
        for name, value in DEFAULTS.items()
    }
    return validate_options(dict(options, clear_l2_cache=mode == "cold"))


def strict_cache_key(options):
    return ("npu-strict", json.dumps([1, options, THRESHOLDS, (20, 40, 5.0)], sort_keys=True))


def _read_profile(directory, names, warmup, active, clear_l2_cache):
    paths = list(Path(directory).rglob("kernel_details.csv"))
    if len(paths) != 1:
        raise ProfilerAcquisitionError(f"Expected one kernel_details.csv, found {len(paths)}")
    targets = set(names) if all(name is not None for name in names) else None
    rows = []
    try:
        with paths[0].open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                name = row["Name"].strip()
                if targets is not None:
                    if name not in targets:
                        continue
                elif clear_l2_cache and row.get("Type", "").strip().lower() == "reducesum":
                    continue
                start, duration = float(row["Start Time(us)"]), float(row["Duration(us)"])
                if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0:
                    raise ValueError("timestamps must be finite and durations positive")
                rows.append((start, duration, name))
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        raise ProfilerAcquisitionError(f"Invalid profiler data: {exc}") from exc
    total = warmup + active
    if len(rows) != len(names) * total:
        raise ProfilerAcquisitionError(f"Expected {len(names) * total} target rows, got {len(rows)}")
    rows.sort(key=lambda row: row[0])
    samples = []
    for index, name in enumerate(names):
        chunk = rows[index * total:(index + 1) * total]
        if name is not None and any(row[2] != name for row in chunk):
            raise ProfilerAcquisitionError(f"Unexpected target order for config {index}: expected {name!r}")
        times, durations = np.asarray([(row[0], row[1]) for row in chunk[warmup:]]).T
        if np.any(np.diff(times) <= 0):
            raise ProfilerAcquisitionError(f"Non-increasing device timestamps for config {index}")
        samples.append((times, durations))
    return samples


def _fast_prune(funcs, synchronize):
    if len(funcs) < 2:
        return list(range(len(funcs)))

    def measure(fn, count):
        best = float("inf")
        for _ in range(count):
            start = time.perf_counter()
            fn()
            synchronize()
            best = min(best, time.perf_counter() - start)
        return best

    first = [measure(fn, 20) for fn in funcs]
    threshold = min(first) * 5.0
    return [i for i, value in enumerate(first) if value <= threshold or measure(funcs[i], 40) <= threshold]


def run_strict_benchmark(funcs, options, *, target_kernel_names=None, prof_dir=None, keep_res=False, verbose=False,
                         prepare_in_call=False):
    from .testing import do_bench_npu_profiler
    from triton.knobs import cache
    import torch

    validate_options(options)
    funcs = list(funcs)
    names = list(target_kernel_names) if target_kernel_names is not None else [None] * len(funcs)
    if len(names) != len(funcs) or any(name is not None and (not isinstance(name, str) or not name) for name in names):
        raise ValueError("Provide one nonempty target kernel name (or None) per function")
    result = dict(timings=[float("inf")] * len(funcs), active=options["active"],
                  cache_mode="cold" if options["clear_l2_cache"] else "hot",
                  rows=[dict(attempts=0, passed=None, status="pruned", reason="", exhausted=False) for _ in funcs])
    if not funcs:
        return result

    def log(message):
        if verbose:
            print(f"npu-strict: {message}")

    def collect(indices, warmup, active):
        root = Path(prof_dir) if prof_dir is not None else Path(cache.get_triton_dir("profile_results"))
        root.mkdir(parents=True, exist_ok=True)
        directory = tempfile.mkdtemp(prefix="npu_strict_", dir=root)
        execution_failed = False

        def track(fn):

            def call(**kwargs):
                nonlocal execution_failed
                try:
                    value = fn(**kwargs)
                    torch.npu.synchronize()
                    return value
                except Exception:
                    execution_failed = True
                    raise

            return call

        try:
            return do_bench_npu_profiler([track(funcs[i]) for i in indices], warmup, active, options["clear_l2_cache"],
                                         directory, True, [names[i] for i in indices], _return_samples=True,
                                         _prepare_in_call=prepare_in_call)
        except (RuntimeError, OSError) as exc:
            if execution_failed:
                raise
            raise ProfilerAcquisitionError(str(exc)) from exc
        finally:
            if not keep_res:
                shutil.rmtree(directory, ignore_errors=True)

    log(f"cache={result['cache_mode']}, options={options}")
    for fn in funcs:  # Materialize JIT work outside both host and profiler measurements.
        fn()
        torch.npu.synchronize()
    kept = _fast_prune(funcs, torch.npu.synchronize)
    log(f"fast prune kept={kept}, total={len(funcs)}")
    for attempt in range(options["max_attempts"]):
        try:
            calibration = collect(kept, 0, options["calibration_runs"])
            break
        except ProfilerAcquisitionError as exc:
            log(f"calibration attempt={attempt + 1}: {exc}")
            if attempt + 1 == options["max_attempts"]:
                raise RuntimeError("npu-strict: calibration budget exhausted; no valid profiler data") from exc
    for _, durations in calibration:
        required = options["measure_budget_ms"] * 1000 / float(np.mean(durations))
        if not math.isfinite(required):
            raise ValueError("Unrepresentable strict sample count")
        result["active"] = max(result["active"], math.ceil(required))
    log(f"main active={result['active']}, warmup={options['warmup']}")
    pending = kept
    for attempt in range(1, options["max_attempts"] + 1):
        for i in pending:
            result["rows"][i]["attempts"] = attempt
        try:
            samples = collect(pending, options["warmup"], result["active"])
        except ProfilerAcquisitionError as exc:
            for i in pending:
                result["rows"][i].update(status="acquisition_failed", reason=str(exc))
            log(f"attempt={attempt}, configs={pending}: {exc}")
            continue
        retry = []
        for i, (times, durations) in zip(pending, samples):
            cost = central_50_mean(durations) / 1000
            metrics, failures = evaluate_quality(times, durations)
            row = result["rows"][i]
            row.update(status="retry" if failures else "pass", reason="; ".join(failures))
            if cost < result["timings"][i]:  # Rejected minima remain eligible.
                result["timings"][i] = cost
                row.update(passed=not failures, best_failures=failures)
            log(f"config={i}, attempt={attempt}, cost_ms={cost:.8g}, {row['status']}, metrics={metrics}, "
                f"failures={failures}")
            if failures:
                retry.append(i)
        pending = retry
        if not pending:
            break
    for i in pending:
        result["rows"][i]["exhausted"] = True
    if not any(math.isfinite(value) for value in result["timings"]):
        raise RuntimeError("npu-strict: no valid profiler measurements; " + "; ".join(row["reason"]
                                                                                      for row in result["rows"]))
    return result


def report_results(result, labels=None, verbose=False):
    timings, rows = result["timings"], result["rows"]
    if not timings:
        return
    best = min(range(len(timings)), key=timings.__getitem__)
    exhausted = [i for i, row in enumerate(rows) if row["exhausted"]]
    reasons = [f"retry budget exhausted for configurations {exhausted}"] if exhausted else []
    if rows[best]["passed"] is False:
        reasons.append(f"selected configuration {best} uses a measurement that failed quality checks")
    if reasons:
        warnings.warn("npu-strict: " + "; ".join(reasons), RuntimeWarning, stacklevel=3)
    if verbose:
        color = sys.stdout.isatty() and os.getenv("TERM") != "dumb" and "NO_COLOR" not in os.environ
        print(f"npu-strict: cache={result['cache_mode']}, samples={result['active']}")
        print("config | best central-half mean (ms) | attempts | best quality | last status | selected")
        for i, row in enumerate(rows):
            status = "N/A" if row["passed"] is None else "PASS" if row["passed"] else "FAIL"
            if color and row["passed"] is not None:
                status = f"\033[{32 if row['passed'] else 31}m{status}\033[0m"
            print(f"{labels[i] if labels is not None else i} | {timings[i]:.8g} | {row['attempts']} | {status} | "
                  f"{row['status']} | {'*' if i == best else ''}")
            if row.get("best_failures") or row["reason"]:
                print(f"  best: {row.get('best_failures', [])}; last: {row['reason']}")
