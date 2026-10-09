"""Private policy routing and profiler acquisition extensions."""

import csv
import time
import warnings
from contextlib import contextmanager, nullcontext
from pathlib import Path

import triton.runtime as runtime

from ._autotune_report import log_stage, make_profile_report, report_safely, timing_stage


def bench_npu(profile, funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name, npu_bench_options,
              _pre_hook_scope, _report_sink, _timing_sink=None):
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
    log_stage(
        options.log_level, "Benchmark settings", f"cache={'cold' if clear_l2_cache else 'hot'}, "
        f"warmup={warmup}, active={active}, quality_check={options.quality_check}, "
        f"filter_slow_configs={options.filter_slow_configs}", detailed=True)
    report_kwargs = {"_report_sink": _report_sink} if _report_sink is not None else {}
    if options.is_default:
        log_stage(options.log_level, "Profiler measurement", f"Attempt 1: {len(funcs)} candidates")
        with timing_stage(_timing_sink, "measurements"):
            return profile(funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name,
                           **report_kwargs)

    names = ([target_kernel_name] * len(funcs)
             if target_kernel_name is None or isinstance(target_kernel_name, str) else list(target_kernel_name))
    if len(names) != len(funcs) or any(name is not None and (not isinstance(name, str) or not name) for name in names):
        raise ValueError("Provide one nonempty target kernel name (or None) per function")
    if not options.needs_samples and not options.filter_slow_configs:
        log_stage(options.log_level, "Profiler measurement", f"Attempt 1: {len(funcs)} candidates")
        with timing_stage(_timing_sink, "measurements"):
            return profile(funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name,
                           _pre_hook_scope=_pre_hook_scope, **report_kwargs)

    import torch

    def measure(callables, kernel_names, warmup_count, active_count, directory, **diagnostics):
        return profile(callables, warmup_count, active_count, clear_l2_cache, directory, True, kernel_names,
                       _return_samples=True, _pre_hook_scope=_pre_hook_scope, **diagnostics)

    root = prof_dir if prof_dir is not None else Path(runtime.cache.get_home_dir()) / ".triton" / "profile_results"
    costs = benchmark_with_options(measure, funcs, names, options, warmup=warmup, active=active, prof_root=root,
                                   synchronize=torch.npu.synchronize, keep_res=keep_res, _timing_sink=_timing_sink,
                                   clear_l2_cache=clear_l2_cache, _pre_hook_scope=_pre_hook_scope, **report_kwargs)
    return costs[0] if len(funcs) == 1 else costs


def _read_profile_samples(directory, names, warmup, active, clear_l2_cache, _report_sink=None, _active_counts=None):
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
    counts = [active] * len(names) if _active_counts is None else list(_active_counts)
    if len(counts) != len(names) or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1 for count in counts):
        raise ValueError("Provide one positive active count per function")
    expected = sum(warmup + count for count in counts)
    if len(rows) != expected:
        raise ProfilerAcquisitionError(f"Expected {expected} target rows, got {len(rows)}")
    rows.sort(key=lambda row: row[0])
    samples = []
    measured_chunks = [] if _report_sink is not None else None
    offset = 0
    for index, (name, count) in enumerate(zip(names, counts)):
        chunk = rows[offset:offset + warmup + count]
        offset += warmup + count
        if name is not None and any(row[2] != name for row in chunk):
            raise ProfilerAcquisitionError(f"Unexpected target order for config {index}: expected {name!r}")
        times, durations = np.asarray([(row[0], row[1]) for row in chunk[warmup:]]).T
        if np.any(np.diff(times) <= 0):
            raise ProfilerAcquisitionError(f"Non-increasing device timestamps for config {index}")
        samples.append((times, durations))
        if measured_chunks is not None:
            measured_chunks.append(chunk[warmup:])
    if _report_sink is not None:

        def capture():
            from ._benchmark_quality import central_50_mean

            reports = [
                make_profile_report(columns, [row[3]
                                              for row in measured_chunks[index]], warmup, counts[index], clear_l2_cache,
                                    float(durations.mean()) / 1000, score_ms=central_50_mean(durations) / 1000)
                for index, (_, durations) in enumerate(samples)
            ]
            _report_sink(reports)

        report_safely(capture)
    return samples


def collect_samples(directory, funcs, names, warmup, active, clear_l2_cache, return_samples, report_sink,
                    active_counts=None):
    names = [names] * len(funcs) if names is None or isinstance(names, str) else names
    samples = _read_profile_samples(directory, names, warmup, active, clear_l2_cache, _report_sink=report_sink,
                                    _active_counts=active_counts)
    if return_samples:
        return samples
    from ._benchmark_quality import central_50_mean
    costs = [central_50_mean(durations) / 1000 for _, durations in samples]
    return costs[0] if len(funcs) == 1 else costs


@contextmanager
def event_timer(clear_l2_cache, pre_hook_scope):
    """Measure host and event time together, restoring preparation before profiling."""
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
        best, total_ms = float("inf"), 0.0
        for _ in range(count):
            if not scoped:
                prepare()
            began = time.perf_counter()
            fn()
            end.record()
            torch.npu.synchronize()
            best = min(best, time.perf_counter() - began)
            total_ms += start.elapsed_time(end)
        return best, total_ms / count

    try:
        with pre_hook_scope(prepare) if scoped else nullcontext():
            yield measure
    finally:
        buffer = None


def profile_costs(filter_df, num_funcs, warmup, active):
    """Score already-read profiler rows, excluding each candidate's warmup."""
    from ._benchmark_quality import central_50_mean

    durations = filter_df["Duration(us)"]
    total = warmup + active
    return [
        central_50_mean(durations.iloc[range(index * total + warmup, (index + 1) * total)].to_numpy()) / 1000
        for index in range(num_funcs)
    ]


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
