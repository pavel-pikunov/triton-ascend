"""Private policy routing and profiler acquisition extensions."""

import csv
import os
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path

import triton.runtime as runtime


def bench_npu(profile, funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name, npu_bench_options,
              _pre_hook_scope, _report_sink):
    from ._npu_benchmark import _ResolvedOptions, _inactive_option_messages, benchmark_with_options, resolve_options

    options = resolve_options(npu_bench_options)
    if not isinstance(npu_bench_options, _ResolvedOptions):
        for _, message in _inactive_option_messages(options, npu_bench_options):
            warnings.warn(f"NPU benchmark: {message}", RuntimeWarning, stacklevel=3)
    funcs = funcs if isinstance(funcs, list) else [funcs]
    if not funcs:
        return []
    warmup = warmup if options.warmup is None else options.warmup
    active = active if options.active is None else options.active
    if options.cache_mode is not None:
        clear_l2_cache = options.cache_mode == "cold"
    verbose = options.verbose if options.verbose is not None else os.getenv("TRITON_PRINT_AUTOTUNING") == "1"
    if verbose:
        print(f"npu benchmark: cache={'cold' if clear_l2_cache else 'hot'}, warmup={warmup}, active={active}, "
              f"quality_check={options.quality_check}, filter_slow_configs={options.filter_slow_configs}")
    report_kwargs = {"_report_sink": _report_sink} if _report_sink is not None else {}
    if options.is_default:
        return profile(funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name, **report_kwargs)

    names = ([target_kernel_name] * len(funcs)
             if target_kernel_name is None or isinstance(target_kernel_name, str) else list(target_kernel_name))
    if len(names) != len(funcs) or any(name is not None and (not isinstance(name, str) or not name) for name in names):
        raise ValueError("Provide one nonempty target kernel name (or None) per function")
    if not options.needs_samples and not options.filter_slow_configs:
        return profile(funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name,
                       _pre_hook_scope=_pre_hook_scope, **report_kwargs)

    import torch

    def measure(callables, kernel_names, warmup_count, active_count, directory, **diagnostics):
        return profile(callables, warmup_count, active_count, clear_l2_cache, directory, True, kernel_names,
                       _return_samples=True, _pre_hook_scope=_pre_hook_scope, **diagnostics)

    root = prof_dir if prof_dir is not None else Path(runtime.cache.get_home_dir()) / ".triton" / "profile_results"
    costs = benchmark_with_options(measure, funcs, names, options, warmup=warmup, active=active, prof_root=root,
                                   synchronize=torch.npu.synchronize, verbose=verbose, keep_res=keep_res,
                                   **report_kwargs)
    return costs[0] if len(funcs) == 1 else costs


def _read_profile_samples(directory, names, warmup, active, clear_l2_cache, _report_sink=None):
    """Read validated device samples for optional quality evaluation and retries."""
    import math
    import numpy as np
    from ._npu_benchmark import ProfilerAcquisitionError

    paths = list(Path(directory).rglob("kernel_details.csv"))
    if len(paths) != 1:
        raise ProfilerAcquisitionError(f"Expected one kernel_details.csv, found {len(paths)}")
    targets = set(names) if all(name is not None for name in names) else None
    rows = []
    try:
        with paths[0].open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            columns = tuple(reader.fieldnames or ()) if _report_sink is not None else None
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
                rows.append((start, duration, name, row) if _report_sink is not None else (start, duration, name))
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
    if _report_sink is not None:
        from ._autotune_report import NpuMeasurementReport, report_safely

        def capture():
            reports = [
                NpuMeasurementReport(
                    columns,
                    tuple(row[3] for row in rows[index * total + warmup:(index + 1) * total]),
                    warmup,
                    active,
                    "cold" if clear_l2_cache else "hot",
                    float(durations.mean()) / 1000,
                ) for index, (_, durations) in enumerate(samples)
            ]
            _report_sink(reports)

        report_safely(capture)
    return samples


def collect_samples(directory, funcs, names, warmup, active, clear_l2_cache, return_samples, report_sink):
    samples = _read_profile_samples(directory, names, warmup, active, clear_l2_cache, _report_sink=report_sink)
    if return_samples:
        return samples
    costs = [float(durations.mean()) / 1000 for _, durations in samples]
    return costs[0] if len(funcs) == 1 else costs


def warn_missing_csv(directory):
    warnings.warn(
        f"NPU profiler CSV kernel_details.csv was not found in {directory!r}; "
        "returning inf for each candidate. Check profiler output and prof_dir.", RuntimeWarning, stacklevel=3)


@contextmanager
def result_scope(remove):
    """Clean up without replacing a primary measurement or launch failure."""
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        try:
            remove()
        except Exception as exc:
            if not failed:
                raise
            from ._npu_benchmark import _warn_secondary
            _warn_secondary("NPU profile cleanup failed", exc)


@contextmanager
def profile_scope(buffer, synchronize, pre_hook_scope, remove):
    """Compose cold-cache hooks and clean up if profiling cannot finish."""

    def evict_cache():
        buffer.sum()
        synchronize()

    scope = pre_hook_scope(evict_cache) if buffer is not None and pre_hook_scope else nullcontext()
    try:
        with scope:
            yield
    except BaseException:
        # On success the collector owns cleanup; on failure preserve the original
        # exception even when removing profiler output also fails.
        with result_scope(remove):
            raise
