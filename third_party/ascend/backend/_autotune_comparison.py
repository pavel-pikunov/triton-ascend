"""Test-branch diagnostics and repeated decisions for NPU and event benchmarks."""

import functools
import os
import time
from contextvars import ContextVar

from . import _autotune_report as report_module
from . import _npu_benchmark as benchmark

_perf_measured = ContextVar("npu_perf_measured", default=False)


def initialize(tuner, options=None, report=None, report_timing=None):
    if os.getenv("TRITON_BENCH_METHOD", "default").lower() == "npu_legacy":
        raise ValueError("npu_legacy was removed; use TRITON_BENCH_METHOD=npu or default")
    tuner.report_best_config = report_module.resolve_report_best_config(report)
    tuner.report_timing = report_module.resolve_report_timing(report_timing)
    tuner._autotune_timing = None
    tuner._npu_timing = None
    tuner._autotune_selection = None
    tuner._autotune_runs, tuner._autotune_csv_dir = report_module.resolve_comparison_options()
    tuner._autotune_comparison = None


def comparison_enabled(tuner):
    return tuner._autotune_runs > 1 or tuner._autotune_csv_dir is not None


def repeat_autotune(tuner, args, kwargs):
    if tuner.user_defined_do_bench:
        raise ValueError("Autotune CSV comparison requires a built-in benchmarker with scores in milliseconds")
    comparison = report_module.AutotuneComparison(tuner._autotune_runs, tuner._autotune_csv_dir)
    tuner._autotune_comparison = comparison
    try:
        for index in range(comparison.runs):
            result = tuner.run(*args, _comparison_pass=True, _comparison_tune_only=index + 1 < comparison.runs,
                               **kwargs)
            if comparison.cache_hit:
                return result
            if index + 1 < comparison.runs:
                # Retain all JIT/compiler caches and choices for other input keys.
                tuner.cache.pop(comparison.key, None)
        summary = comparison.timing_summary()
        message = (f"Triton autotune comparison: {comparison.completed} runs; "
                   f"mean autotune: {summary['mean_autotune_s']:.6f} s")
        if comparison.directory is not None:
            path = comparison.write_csv(tuner.base_fn.__name__)
            message += f"; CSV: {path}"
        print(message)
        return result
    except BaseException:
        if comparison.key is not None:
            tuner.cache.pop(comparison.key, None)
        raise
    finally:
        tuner._autotune_comparison = None
        tuner._autotune_selection = None
        tuner._autotune_timing = None
        tuner._npu_timing = None
        tuner.nargs = None


def capture_selection(tuner, key, cache_miss):
    tuner._autotune_selection = None
    if tuner._autotune_comparison is not None:
        tuner._autotune_selection = tuner._autotune_comparison.prepare(tuner.configs, key, cache_miss)
    elif cache_miss and tuner.report_best_config:
        tuner._autotune_selection = report_module.report_safely(
            lambda: report_module.snapshot_candidates(tuner.configs, key))


def record_comparison(tuner, config, timings, cache_miss):
    if cache_miss and tuner._autotune_comparison is not None:
        tuner._autotune_comparison.record(tuner._autotune_selection, config, timings)


def record_scored_candidates(tuner, configs):
    if configs:
        _perf_measured.set(True)
    if tuner._autotune_selection is not None:
        report_module.report_safely(lambda: report_module.record_scored_candidates(tuner._autotune_selection, configs))


def profile_scores(run_fns, costs, report_sink):
    if report_sink is None:
        return

    def capture():
        from .testing import _npu_bench_options
        options = _npu_bench_options(clear_l2_cache=True)
        report_sink({
            config:
            report_module.ScoreReport(cost, False, "npu", cache_mode=options[-1], quantiles=options[5:7])
            for config, cost in zip(run_fns, costs)
        })

    report_module.report_safely(capture)


def profile_npu(tuner, measure, *args, **kwargs):
    from unittest.mock import patch
    from . import testing

    original = testing._profile_npu_batch
    batch = 0

    def profile(funcs, warmup, active, *rest):
        nonlocal batch
        batch += 1
        level = os.getenv("TRITON_NPU_BENCH_LOG_LEVEL", "off")
        series = int(os.getenv("TRITON_NPU_BENCH_BATCH_RUNS", "1"))
        benchmark.log_stage(
            level, "Profiler measurement", f"Batch {batch}: {len(funcs) // series} candidates, {series} series each; "
            f"warmup={warmup}, active={active}, export={os.getenv('TRITON_NPU_BENCH_PROFILER_MODE', 'full')}")
        started = time.perf_counter()
        result = original(funcs, warmup, active, *rest)
        benchmark.log_stage(level, "Profiler complete", f"Batch {batch}: {time.perf_counter() - started:.3f} s")
        return result

    with patch.object(testing, "_profile_npu_batch", profile):
        return call_stage(tuner, "measurements", measure, *args, **kwargs)


def start_timing(tuner):
    _perf_measured.set(True)  # A cache hit is also handled by the autotuner.
    enabled = tuner.report_timing or tuner._autotune_comparison is not None
    tuner._autotune_timing = (time.perf_counter(), {}) if enabled else None
    # Production NPU helpers and comparison diagnostics write to the same stages.
    tuner._npu_timing = tuner._autotune_timing


def add_timing(tuner, name, elapsed):
    if tuner._autotune_timing is not None:
        durations = tuner._autotune_timing[1]
        durations[name] = durations.get(name, 0) + elapsed


def begin_stage(tuner, title=None, message=""):
    if title is not None and os.getenv("TRITON_BENCH_METHOD", "default").lower() == "npu":
        benchmark.log_stage(os.getenv("TRITON_NPU_BENCH_LOG_LEVEL", "off"), title, message)
    return time.perf_counter() if tuner._autotune_timing is not None else None


def log_candidates(tuner, title, count, total):
    if os.getenv("TRITON_BENCH_METHOD", "default").lower() == "npu":
        benchmark.log_stage(os.getenv("TRITON_NPU_BENCH_LOG_LEVEL", "off"), title,
                            f"{count}/{total} candidates continue")


def compilation_result(tuner, configs, run_fns):
    log_candidates(tuner, "Compilation result", len(run_fns), len(configs))


def end_stage(tuner, name, started):
    if started is not None:
        add_timing(tuner, name, time.perf_counter() - started)


def call_stage(tuner, name, fn, *args, **kwargs):
    with report_module.timing_stage(timing_sink(tuner), name):
        return fn(*args, **kwargs)


def timing_sink(tuner):
    return functools.partial(add_timing, tuner) if tuner._autotune_timing is not None else None


def finish_timing(tuner, cache_miss):
    timing = tuner._autotune_timing
    tuner._autotune_timing = None
    tuner._npu_timing = None
    if cache_miss and timing is not None:
        total = time.perf_counter() - timing[0]
        if tuner._autotune_comparison is not None:
            tuner._autotune_comparison.record_timing(total, timing[1])
        if tuner.report_timing:
            report_module.report_safely(
                lambda: report_module.print_timing_report(tuner.base_fn.__name__, *timing, total=total))


def bench_events(tuner, run_fns, report_sink):
    with report_module.timing_stage(timing_sink(tuner), "measurements"):
        costs = {config: tuner.do_bench(fn, quantiles=(0.5, 0.2, 0.8)) for config, fn in run_fns.items()}
    if report_sink is not None:
        report_module.report_safely(lambda: report_sink(
            {config: report_module.ScoreReport(score, tuner.user_defined_do_bench)
             for config, score in costs.items()}))
    return costs


def print_winner(tuner, config, reports):
    if reports is not None:
        try:
            sink = benchmark._perf_report_sink.get()
            if sink is not None and reports.get(config) is not None:
                sink(tuner.base_fn.__name__, reports[config])
            if not tuner.report_best_config:
                return
            report_module.report_safely(lambda: report_module.print_best_config_report(
                tuner.base_fn.__name__, tuner.configs, config, reports.get(config), selection=tuner._autotune_selection)
                                        )
        finally:
            reports.clear()
            tuner._autotune_selection = None


def capture_reports(tuner, cache_miss):
    return {} if cache_miss and (tuner.report_best_config or benchmark._perf_report_sink.get() is not None) else None
