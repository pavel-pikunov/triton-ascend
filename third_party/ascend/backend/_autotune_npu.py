"""NPU-only integration with the existing autotuner."""

import functools
import os
import time
from contextlib import contextmanager

from . import _npu_benchmark as benchmark


def get_options(tuner):
    if not tuner.user_defined_do_bench and os.getenv("TRITON_BENCH_METHOD", "default").lower() == "npu":
        return benchmark.resolve_options()
    return None


def start_timing(tuner):
    options = tuner._npu_options
    tuner._npu_timing = (time.perf_counter(), {}) if options is not None and options.report_timing else None


def durations(tuner):
    timing = getattr(tuner, "_npu_timing", None)
    return timing[1] if timing is not None else None


def stage(tuner, name):
    return benchmark.timing_stage(durations(tuner), name)


def begin_stage(tuner, title=None, message=""):
    options = tuner._npu_options
    if options is not None and title is not None:
        benchmark.log_stage(options.log_level, title, message)
    return time.perf_counter() if getattr(tuner, "_npu_timing", None) is not None else None


def end_stage(tuner, name, started):
    benchmark.record_timing(durations(tuner), name, started)


def log_candidates(tuner, title, count, total):
    options = tuner._npu_options
    if options is not None:
        benchmark.log_stage(options.log_level, title, f"{count}/{total} candidates continue")


def compilation_result(tuner, configs, run_fns):
    log_candidates(tuner, "Compilation result", len(run_fns), len(configs))
    options = tuner._npu_options
    if options is not None and options.log_level == "detailed":
        for config in configs:
            number = next((i for i, candidate in enumerate(tuner.configs, 1) if candidate is config), None)
            status = "compiled" if config in run_fns else "failed"
            benchmark.report_safely(lambda: benchmark.log_stage(
                options.log_level, "Config compilation", f"config={number}, status={status}, parameters={vars(config)}",
                detailed=True))


@contextmanager
def cache_pre_hook(tuner, evict_cache):
    original = tuner.pre_hook

    def pre_hook(args, reset_only=False):
        if reset_only:
            return original(args, reset_only=True)
        original(args)
        try:
            evict_cache()
        except Exception as exc:
            try:
                tuner.post_hook(args, exception=exc)
            except Exception:
                pass
            raise

    tuner.pre_hook = pre_hook
    try:
        yield
    finally:
        tuner.pre_hook = original


def profile_candidates(tuner, kernels_call, run_fns, warmup, active, report_sink):
    from .testing import _profile_npu

    options = tuner._npu_options
    names = (tuner._resolve_target_kernel_name(kernels_call, run_fns.keys()) if options.is_default else
             [getattr(kernels_call[config], "target_kernel_name", None) for config in run_fns])
    costs = benchmark.bench_npu(
        _profile_npu, list(run_fns.values()), warmup, active, True, None, False, names, options=options,
        pre_hook_scope=functools.partial(cache_pre_hook, tuner) if options.cache_mode == "cold" else None,
        report_sink=(lambda records: report_sink(dict(zip(run_fns, records)))) if report_sink is not None else None,
        timings=durations(tuner), config_numbers=[
            next((index for index, candidate in enumerate(tuner.configs, 1) if candidate is config),
                 None) for config in run_fns
        ])
    costs = [costs] if len(run_fns) == 1 else costs
    assert len(costs) == len(run_fns)
    return dict(zip(run_fns, costs))


def finish_reports(tuner, config, reports, cache_miss):
    timing = tuner._npu_timing
    tuner._npu_timing = None
    if cache_miss and timing is not None:
        benchmark.report_safely(lambda: benchmark.print_timing_report(tuner.base_fn.__name__, *timing))
    if reports is not None:
        try:
            benchmark.report_safely(lambda: benchmark.print_best_config_report(tuner.base_fn.__name__, tuner.configs,
                                                                               config, reports.get(config)))
        finally:
            reports.clear()
