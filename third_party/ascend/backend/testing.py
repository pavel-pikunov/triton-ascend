# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import builtins
import math
import multiprocessing
import os
from datetime import datetime, timezone
from typing import Optional

import triton.runtime as runtime


class ProfilerResultMismatchError(RuntimeError):
    def __init__(self, target_kernel_name: str, expected_rows: int, actual_rows: int):
        self.target_kernel_name = target_kernel_name
        self.expected_rows = expected_rows
        self.actual_rows = actual_rows
        super().__init__(
            "Profiler rows filtered by target kernel name do not match the expected count. "
            f"target_kernel_name={target_kernel_name!r}, expected_rows={expected_rows}, actual_rows={actual_rows}"
        )


def _npu_bench_options(warmup=5, active=30, clear_l2_cache=False):
    values = tuple(
        type(default)(os.getenv("TRITON_NPU_BENCH_" + name, default)) for name, default in (
            ("WARMUP", 5 if warmup is None else warmup),
            ("ACTIVE", active),
            ("BATCH_RUNS", 1),
            ("STABILITY_RUNS", 1),
            ("STABILITY_TOLERANCE", 0.01),
            ("LOWER_QUANTILE", 0.0),
            ("UPPER_QUANTILE", 1.0),
            ("MEASURE_BUDGET_MS", 0.0),
            ("MAX_RETRIES", 0),
            ("CACHE_MODE", "cold" if clear_l2_cache else "hot"),
        ))
    warmup, active, batch_runs, confirmations, tolerance, lower, upper, budget, batches, mode = values
    if not (warmup >= 0 and active > 0 and batch_runs > 0 and confirmations > 0 and batches >= 0
            and math.isfinite(tolerance) and tolerance >= 0 and math.isfinite(budget) and budget >= 0
            and 0 <= lower < upper <= 1 and mode in ("hot", "cold")):
        raise ValueError("Invalid TRITON_NPU_BENCH settings")
    return values


def do_bench_npu(
    funcs,
    warmup=5,
    active=30,
    clear_l2_cache=False,
    prof_dir=None,
    keep_res=False,
    target_kernel_name: Optional[str] = None,
    _active_limit=None,
):
    """Confirm minima when TRITON_NPU_BENCH_STABILITY_RUNS > 1, within STABILITY_TOLERANCE.
    Retry budgets use ordinary means from the first profile.
    """
    import torch
    import torch_npu

    if not isinstance(funcs, list):
        funcs = [funcs]
    warmup_limit = warmup
    warmup, active, batch_runs, confirmations, tolerance, lower, upper, budget, batches, mode = _npu_bench_options(
        warmup, active, clear_l2_cache)
    clear_l2_cache = mode == "cold"
    if _active_limit is not None:
        warmup = min(warmup, warmup_limit)
        active = min(active, _active_limit)
        budget = 0

    # warmup kernel
    for fn in funcs:
        fn()
        torch.npu.synchronize()

    aic_metrics = torch_npu.profiler.AiCMetrics.AiCoreNone
    if os.getenv("TRITON_NPU_BENCH_PIPE_UTILIZATION", "1") == "1":
        aic_metrics = torch_npu.profiler.AiCMetrics.PipeUtilization
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=aic_metrics,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )

    if prof_dir is not None:
        torch_path = prof_dir
    else:
        process = multiprocessing.current_process()
        pid = process.pid
        process_name = process.name
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(
            runtime.cache.get_home_dir(), ".triton", "profile_results"
        )
        torch_path = os.path.join(base_path, f"prof_{timestamp}_{process_name}-{pid}")

    buffer = None
    if clear_l2_cache:
        buffer = runtime.driver.active.get_empty_cache_for_benchmark()
        buffer = buffer.float()  # to avoid type cast
        buffer.sum()
        torch.npu.synchronize()  # shake out of any npu error

    histories = [[] for _ in funcs]
    pending = list(builtins.range(len(funcs)))
    for batch in builtins.range(max(1, batches)):
        selected = [funcs[index] for index in pending] * batch_runs
        batch_path = os.path.join(torch_path, f"batch_{batch}") if batches > 1 else torch_path
        _profile_npu_batch(selected, warmup, active, buffer, batch_path, experimental_config)
        means = []
        try:
            scores = _collect_prof_result(batch_path, selected, warmup, active, target_kernel_name, clear_l2_cache,
                                          (lower, upper), means)
        finally:
            _rm_dic(keep_res, torch_path)
        if len(selected) == 1:
            scores = [scores]
        for position, index in enumerate(pending):
            histories[index] = sorted(histories[index] + scores[position::len(pending)])[:confirmations]
        pending = [
            index for index in pending
            if confirmations > 1 and (len(histories[index]) < confirmations or not math.isfinite(histories[index][-1])
                                      or histories[index][0] <= 0 or histories[index][-1] > histories[index][0] *
                                      (1 + tolerance))
        ]
        if not pending:
            break
        if batch == 0 and batches > 1 and budget and means and math.isfinite(min(means)):
            previous_active = active
            active = max(active, math.ceil(budget / min(means)))
            if active != previous_active:
                for index in pending:
                    histories[index].clear()
    if clear_l2_cache:
        del buffer
    if pending:
        print(f"NPU benchmark: minimum not confirmed for configurations {[index + 1 for index in pending]}")
    time_cost = [history[0] for history in histories]
    return time_cost[0] if len(funcs) == 1 else time_cost


def _profile_npu_batch(funcs, warmup, active, buffer, torch_path, experimental_config):
    import torch
    import torch_npu

    total = warmup + active
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for fn in funcs:
            for _ in builtins.range(total):
                if buffer is not None:
                    buffer.sum()  # use buffer read to clear l2 cache
                    torch.npu.synchronize()
                fn()
                torch.npu.synchronize()


def _rm_dic(keep_res, torch_path):
    if keep_res:
        return
    import shutil

    if os.path.exists(torch_path):
        shutil.rmtree(torch_path)


def _collect_prof_result(
    base_dir: str,
    funcs,
    num_warmup: int,
    num_active: int,
    target_kernel_name: Optional[str] = None,
    clear_l2_cache: bool = False,
    quantiles=(0, 1),
    means=None,
):
    """
    Collect kernel performance from kernel_details.csv, returned in millisecond.
    The first `num_warmup` rows of each function are warmup data and will be ignored, the next `num_active` rows will be averaged.

    :param base_dir: the profiler path
    :type base_dir: str
    :param funcs: a list of Callable being profiled
    :type funcs: List[Callable]
    :param num_warmup: warmup count in kernel_details.csv of each fn
    :type num_warmup: int
    :param num_active: active count in kernel_details.csv of each fn
    :type num_active: int
    :param target_kernel_name: target triton kernel name reported by profiler
    :type target_kernel_name: Optional[str]
    :param quantiles: lower and upper sorted-sample boundaries; retain at least one duration
    :param means: optional output list of ordinary means in milliseconds, before trimming
    """

    import numpy as np
    import pandas as pd

    kernel_details_file = None
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "kernel_details.csv":
                kernel_details_file = os.path.join(root, file)
                break
    num_funcs = len(funcs)
    if kernel_details_file is None:
        if num_funcs == 1:
            return float("inf")
        else:
            return [float("inf")] * num_funcs

    df = pd.read_csv(kernel_details_file)
    # filter out l2 cache clearing operation
    filter_cond = (not clear_l2_cache) | ~df["Type"].str.contains(r"^ReduceSum$", case=False, na=False)
    filter_df = df[filter_cond]
    if target_kernel_name is not None:
        filter_df = filter_df[filter_df["Name"] == target_kernel_name]

    expected_rows = num_funcs * (num_warmup + num_active)
    actual_rows = len(filter_df)
    if target_kernel_name is not None and actual_rows != expected_rows:
        raise ProfilerResultMismatchError(target_kernel_name, expected_rows, actual_rows)

    durations = filter_df["Duration(us)"].to_numpy(dtype=float)[:expected_rows].reshape(
        num_funcs, num_warmup + num_active)
    samples = durations[:, num_warmup:] / 1e3
    valid = np.isfinite(samples).all(axis=1) & (samples > 0).all(axis=1)
    if means is not None:
        means.extend(np.where(valid, samples.mean(axis=1), np.inf).tolist())
    start, end = (int(num_active * quantile) for quantile in quantiles)
    time_cost = (samples.mean(axis=1) if quantiles == (0, 1) else np.where(
        valid,
        np.sort(samples, axis=1)[:, start:max(start + 1, end)].mean(axis=1), np.inf)).tolist()

    if num_funcs == 1:
        return time_cost[0]
    else:
        return time_cost


def benchmark_npu(kernel, grid=None, *args, **kwargs):
    """Reuse autotune measurements, or benchmark a fixed launch through do_bench_npu."""
    from ._autotune_comparison import _perf_measured

    launch = kernel if grid is None else lambda: kernel[grid](*args, **kwargs)
    fn = kernel
    while grid is not None and hasattr(fn, "fn"):
        if isinstance(fn, runtime.Autotuner):
            return launch()
        fn = fn.fn
    token = _perf_measured.set(False)
    try:
        result = launch()
        measured = _perf_measured.get()
    finally:
        _perf_measured.reset(token)
    if measured:
        _perf_measured.set(True)
    else:
        compiled = (result[0] if isinstance(result, tuple) else result) if grid is not None else None
        target = getattr(compiled, "packed_metadata", {}).get("kernel_name")
        do_bench_npu(launch, target_kernel_name=target)
    return result
