"""Private integration of optional NPU policies with the existing tuner."""

import copy
import functools
import math
import os
import time
import warnings
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict

from . import _autotune_report as report_module


def initialize(tuner, options, report, report_timing=None):
    tuner._warning_reasons = set()
    tuner._warning_registry = {}
    tuner.npu_bench_options = copy.copy(options)
    tuner.report_best_config = report_module.resolve_report_best_config(report)
    tuner.report_timing = report_module.resolve_report_timing(report_timing)
    tuner._autotune_timing = None


def start_timing(tuner):
    tuner._autotune_timing = (time.perf_counter(), {}) if tuner.report_timing else None


def add_timing(tuner, name, elapsed):
    if tuner._autotune_timing is not None:
        durations = tuner._autotune_timing[1]
        durations[name] = durations.get(name, 0) + elapsed


def begin_stage(tuner, title=None, message=""):
    if title is not None:
        options = getattr(tuner, "_npu_benchmark_options", None)
        if options is not None:
            report_module.log_stage(options.log_level, title, message)
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
    if cache_miss and timing is not None:
        report_module.report_safely(lambda: report_module.print_timing_report(tuner.base_fn.__name__, *timing))


def log_candidates(tuner, title, count, total):
    options = getattr(tuner, "_npu_benchmark_options", None)
    if options is not None:
        report_module.log_stage(options.log_level, title, f"{count}/{total} candidates continue")


def compilation_result(tuner, configs, run_fns):
    log_candidates(tuner, "Compilation result", len(run_fns), len(configs))
    options = getattr(tuner, "_npu_benchmark_options", None)
    if options is not None and options.log_level == "detailed":
        for config in configs:
            number = next((i for i, candidate in enumerate(tuner.configs, 1) if candidate is config), None)
            status = "compiled" if config in run_fns else "failed"
            report_module.report_safely(lambda: report_module.log_stage(
                options.log_level, "Config compilation", f"config={number}, status={status}, "
                f"parameters={vars(config)}", detailed=True))


def bench_events(tuner, run_fns, report_sink):
    with report_module.timing_stage(timing_sink(tuner), "measurements"):
        costs = {config: tuner.do_bench(fn, quantiles=(0.5, 0.2, 0.8)) for config, fn in run_fns.items()}
    if report_sink is not None:
        report_module.report_safely(lambda: report_sink(
            {config: report_module.ScoreReport(score, tuner.user_defined_do_bench)
             for config, score in costs.items()}))
    return costs


def warn_once(tuner, reason, message):
    """Emit each reason once per tuner, respecting ignore and error filters."""
    if reason in tuner._warning_reasons:
        return
    kernel = tuner.base_fn
    warnings.warn_explicit(
        f"Autotuning kernel {kernel.__name__}: {message}",
        RuntimeWarning,
        filename=kernel.__code__.co_filename,
        lineno=kernel.__code__.co_firstlineno,
        module=kernel.__module__,
        registry=tuner._warning_registry,
    )
    tuner._warning_reasons.add(reason)


def get_options(tuner):
    from ._npu_benchmark import _inactive_option_messages, _supplied_option_fields, resolve_options

    if not tuner.user_defined_do_bench and os.getenv("TRITON_BENCH_METHOD", "default").lower() == "npu":
        options = resolve_options(tuner.npu_bench_options)
        for reason, message in _inactive_option_messages(options, tuner.npu_bench_options):
            tuner._warn_once(reason, message)
        return options
    fields = _supplied_option_fields(tuner.npu_bench_options)
    if fields or tuner.npu_bench_options is not None:
        reason = ("a custom do_bench was provided"
                  if tuner.user_defined_do_bench else "TRITON_BENCH_METHOD is not set to 'npu'")
        tuner._warn_once(
            "ignored_npu_options", f"npu_bench_options and TRITON_NPU_BENCH_* measurement settings "
            f"({', '.join(sorted(fields)) or 'npu_bench_options'}) are ignored because {reason}. "
            "Use TRITON_BENCH_METHOD=npu with the built-in benchmarker to apply them.")
    return None


@contextmanager
def cache_pre_hook(tuner, evict_cache):
    """Compose measurement preparation after both existing kernel hooks."""
    original = tuner.pre_hook

    def pre_hook(args, reset_only=False):
        if reset_only:
            return original(args, reset_only=True)
        original(args)
        try:
            evict_cache()
        except Exception as exc:
            # Preparation may have saved tensors for restore_value. Pair
            # it with the existing cleanup even if eviction prevents launch.
            try:
                tuner.post_hook(args, exception=exc)
            except Exception as cleanup_error:
                from ._npu_benchmark import _warn_secondary
                _warn_secondary("NPU post_hook failed after cache eviction", cleanup_error)
            raise

    tuner.pre_hook = pre_hook
    try:
        yield
    finally:
        tuner.pre_hook = original


def append_policy_key(key, options):
    if options is not None and not options.is_default:
        key.append(options.cache_key())


def check_scores(tuner, timings):
    if tuner._npu_benchmark_options is None:
        return
    if not any(math.isfinite(cost[0] if isinstance(cost, Sequence) else cost) for cost in timings.values()):
        tuner._warn_once(
            "unusable_measurements", "All final benchmark scores are non-finite; "
            "selecting a config without usable measurements. Check profiler output "
            "and benchmark settings.")


def warn_skipped_measurements(tuner):
    options = tuner._npu_benchmark_options
    if options is not None and not options.is_default:
        tuner._warn_once(
            "skipped_measurements", "Only one config remains after pruning; autotuning "
            "skips measurements and the effective NPU benchmark policy is not applied. "
            "Provide multiple surviving configs or benchmark the kernel directly.")


def print_winner(tuner, config, reports):
    if reports is not None:
        try:
            report_module.report_safely(lambda: report_module.print_best_config_report(
                tuner.base_fn.__name__, tuner.configs, config, reports.get(config)))
        finally:
            reports.clear()


def prune_candidates(tuner, run_fns, options):
    use_existing_counts = options is None or (options.active is None and options.measure_budget_ms is None)
    cv_mode = (use_existing_counts and len(run_fns) > 1 and tuner.parser_mode in ("cube", "mix")
               and tuner.cv_parse_result is not None)
    if cv_mode:
        begin_stage(tuner, "Time-limit pruning", f"Starting: {len(run_fns)} candidates")
        if options is not None and options.warmup is not None:
            run_fns = tuner._prune_by_time_limit(run_fns, warmup=options.warmup)
        else:
            run_fns = tuner._prune_by_time_limit(run_fns)
    return run_fns, cv_mode


def prepare_profile(tuner, options, kernels_call, run_fns, report_sink):
    from ._npu_benchmark import _ResolvedOptions
    # Resolved defaults must override conflicting environment values without
    # repeating diagnostics for fields synthesized by the tuner.
    kwargs = {"npu_bench_options": _ResolvedOptions(asdict(options))}
    if timing_sink(tuner) is not None:
        kwargs["_timing_sink"] = timing_sink(tuner)
    reports = [] if report_sink is not None else None
    if reports is not None:
        kwargs["_report_sink"] = reports.extend
    if not options.is_default:
        if options.cache_mode == "cold":
            kwargs["_pre_hook_scope"] = tuner._npu_cache_pre_hook
        names = [getattr(kernels_call[config], "target_kernel_name", None) for config in run_fns]
    else:
        names = tuner._resolve_target_kernel_name(kernels_call, run_fns.keys())
    return names, kwargs, reports


def finish_profile(run_fns, costs, reports, report_sink):
    if len(run_fns) == 1:
        costs = [costs]
    assert len(costs) == len(run_fns)
    if report_sink is not None:
        report_module.report_safely(lambda: report_sink(dict(zip(run_fns, reports))))
    return dict(zip(run_fns, costs))


def warn_extra_profile_options(tuner):
    tuner._warn_once(
        "extra_profile_options", "The additional winner _profile does not receive explicit "
        "npu_bench_options; it uses TRITON_NPU_BENCH_* environment settings and defaults. "
        "Set those environment variables for this profile or omit auto_prof_dir.")
