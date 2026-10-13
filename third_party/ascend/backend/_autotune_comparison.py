"""Test-branch diagnostics and repeated decisions shared by three benchmark paths."""

import functools
import os
import time

from . import _autotune_report as report_module
from . import _autotune_npu as npu_tuning


def initialize(tuner, options=None, report=None, report_timing=None):
    defaults = npu_tuning.get_options(tuner)
    tuner.report_best_config = report_module.resolve_report_best_config(
        report, defaults.report_best_config if defaults else False)
    tuner.report_timing = report_module.resolve_report_timing(report_timing,
                                                              defaults.report_timing if defaults else False)
    tuner._autotune_timing = None
    tuner._npu_timing = None
    tuner._npu_options = None
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
    if tuner._autotune_selection is not None:
        report_module.report_safely(lambda: report_module.record_scored_candidates(tuner._autotune_selection, configs))


def legacy_profile_scores(run_fns, costs, report_sink, warmup, active):
    if report_sink is None:
        return

    def capture():
        cache_mode = os.getenv("TRITON_NPU_BENCH_CACHE_MODE", "cold").strip().lower()
        report_sink({
            config: report_module.ScoreReport(cost, False, "npu_legacy", warmup, active, cache_mode)
            for config, cost in zip(run_fns, costs)
        })

    report_module.report_safely(capture)


def start_timing(tuner):
    enabled = tuner.report_timing or tuner._autotune_comparison is not None
    tuner._autotune_timing = (time.perf_counter(), {}) if enabled else None
    # Production NPU helpers and comparison diagnostics write to the same stages.
    tuner._npu_timing = tuner._autotune_timing


def add_timing(tuner, name, elapsed):
    if tuner._autotune_timing is not None:
        durations = tuner._autotune_timing[1]
        durations[name] = durations.get(name, 0) + elapsed


def begin_stage(tuner, title=None, message=""):
    return time.perf_counter() if tuner._autotune_timing is not None else None


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
            report_module.report_safely(lambda: report_module.print_best_config_report(
                tuner.base_fn.__name__, tuner.configs, config, reports.get(config), selection=tuner._autotune_selection)
                                        )
        finally:
            reports.clear()
            tuner._autotune_selection = None
