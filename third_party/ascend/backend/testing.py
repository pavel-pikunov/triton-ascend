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


def do_bench_npu(
    funcs,
    warmup=5,
    active=30,
    clear_l2_cache=False,
    prof_dir=None,
    keep_res=False,
    target_kernel_name: Optional[str] = None,
    *,
    npu_bench_options=None,
    _pre_hook_scope=None,
    _report_sink=None,
    _timing_sink=None,
):
    """Profile NPU kernels, optionally controlling L2, quality and retries.

    npu_bench_options overrides TRITON_NPU_BENCH_* per field. Unspecified
    fields inherit the existing arguments. Counts are launches, budget is ms,
    and max_retries counts additional attempts after the initial measurement.
    The result is the mean of the central 50% of measured durations, as a scalar
    for one callable and a list for multiple ones.
    """
    from ._npu_profiler import bench_npu
    return bench_npu(_profile_npu, funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name,
                     npu_bench_options, _pre_hook_scope, _report_sink, _timing_sink)


def _profile_npu(
    funcs,
    warmup=5,
    active=30,
    clear_l2_cache=False,
    prof_dir=None,
    keep_res=False,
    target_kernel_name=None,
    *,
    _return_samples: bool = False,
    _pre_hook_scope=None,
    _report_sink=None,
    _active_counts=None,
):
    import torch
    import torch_npu
    from ._npu_profiler import profile_scope, result_scope

    if not isinstance(funcs, list):
        funcs = [funcs]

    # warmup kernel
    for fn in funcs:
        fn()
        torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
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

    if clear_l2_cache:
        buffer = runtime.driver.active.get_empty_cache_for_benchmark()
        buffer = buffer.float()  # to avoid type cast
        buffer.sum()
        torch.npu.synchronize()  # shake out of any npu error

    scope = profile_scope(buffer if clear_l2_cache else None, torch.npu.synchronize, _pre_hook_scope,
                          lambda: _rm_dic(keep_res, torch_path))
    with scope, torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
    ):
        for index, fn in enumerate(funcs):
            total = warmup + (active if _active_counts is None else _active_counts[index])
            for _ in builtins.range(total):
                if clear_l2_cache and _pre_hook_scope is None:
                    buffer.sum()  # use buffer read to clear l2 cache
                    torch.npu.synchronize()
                fn()
                torch.npu.synchronize()
    if clear_l2_cache:
        del buffer

    with result_scope(lambda: _rm_dic(keep_res, torch_path)):
        return _collect_prof_result(
            torch_path,
            funcs,
            warmup,
            active,
            target_kernel_name=target_kernel_name,
            clear_l2_cache=clear_l2_cache,
            _return_samples=_return_samples,
            **({"_active_counts": _active_counts} if _active_counts is not None else {}),
            **({"_report_sink": _report_sink} if _report_sink is not None else {}),
        )


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
    _return_samples: bool = False,
    _report_sink=None,
    _active_counts=None,
):
    """
    Collect kernel performance from kernel_details.csv, returned in millisecond.
    The first `num_warmup` rows of each function are warmup data and will be ignored.
    The score is the mean of the central 50% of the next `num_active` durations.

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
    """

    if _return_samples or _active_counts is not None or isinstance(target_kernel_name, (list, tuple)):
        from ._npu_profiler import collect_samples
        return collect_samples(base_dir, funcs, target_kernel_name, num_warmup, num_active, clear_l2_cache,
                               _return_samples, _report_sink, _active_counts)

    import pandas as pd

    kernel_details_file = None
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == "kernel_details.csv":
                kernel_details_file = os.path.join(root, file)
                break
    num_funcs = len(funcs)
    if kernel_details_file is None:
        from ._npu_profiler import warn_missing_csv
        warn_missing_csv(base_dir)
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

    from ._npu_profiler import profile_costs
    time_cost = profile_costs(filter_df, num_funcs, num_warmup, num_active)

    if _report_sink is not None:
        from ._autotune_report import capture_profile_report
        capture_profile_report(filter_df, time_cost, num_warmup, num_active, clear_l2_cache, _report_sink)

    if num_funcs == 1:
        return time_cost[0]
    else:
        return time_cost
