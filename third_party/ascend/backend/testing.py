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
import csv
import multiprocessing
import os
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
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
):
    """Profile NPU kernels, optionally controlling L2, quality and retries.

    npu_bench_options overrides TRITON_NPU_BENCH_* per field. Unspecified
    fields inherit the existing arguments. Counts are launches, budget is ms,
    and max_retries counts additional attempts after the initial measurement.
    The result remains a scalar for one callable and a list for multiple ones.
    """
    from ._npu_benchmark import benchmark_with_options, resolve_options

    options = resolve_options(npu_bench_options)
    funcs = funcs if isinstance(funcs, list) else [funcs]
    if not funcs:
        return []
    warmup = warmup if options.warmup is None else options.warmup
    active = active if options.active is None else options.active
    if options.cache_mode is not None:
        clear_l2_cache = options.cache_mode == "cold"
    if options.is_default:
        return _profile_npu(funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name)

    names = ([target_kernel_name] * len(funcs)
             if target_kernel_name is None or isinstance(target_kernel_name, str) else list(target_kernel_name))
    if len(names) != len(funcs) or any(name is not None and (not isinstance(name, str) or not name) for name in names):
        raise ValueError("Provide one nonempty target kernel name (or None) per function")
    verbose = options.verbose if options.verbose is not None else os.getenv("TRITON_PRINT_AUTOTUNING") == "1"
    if verbose:
        print(f"npu benchmark: cache={'cold' if clear_l2_cache else 'hot'}, warmup={warmup}, active={active}, "
              f"quality_check={options.quality_check}, filter_slow_configs={options.filter_slow_configs}")
    if not options.needs_samples and not options.filter_slow_configs:
        return _profile_npu(funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name,
                            _pre_hook_scope=_pre_hook_scope)

    import torch

    def measure(callables, kernel_names, warmup_count, active_count, directory):
        return _profile_npu(callables, warmup_count, active_count, clear_l2_cache, directory, True,
                            kernel_names, _return_samples=True, _pre_hook_scope=_pre_hook_scope)

    root = prof_dir if prof_dir is not None else Path(runtime.cache.get_home_dir()) / ".triton" / "profile_results"
    costs = benchmark_with_options(measure, funcs, names, options, warmup=warmup, active=active,
                                   prof_root=root, synchronize=torch.npu.synchronize,
                                   verbose=verbose, keep_res=keep_res)
    return costs[0] if len(funcs) == 1 else costs


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
):
    import torch
    import torch_npu

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

    def evict_cache():
        buffer.sum()
        torch.npu.synchronize()

    total = warmup + active
    # Only an explicitly selected cold cache policy composes the autotuner hook.
    # Ordinary callers retain cache eviction before fn(), as in release/3.2.2.
    hook_scope = _pre_hook_scope(evict_cache) if clear_l2_cache and _pre_hook_scope else nullcontext()
    try:
        with hook_scope, torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
        ):
            for fn in funcs:
                for _ in builtins.range(total):
                    if clear_l2_cache and _pre_hook_scope is None:
                        evict_cache()
                    fn()
                    torch.npu.synchronize()
        return _collect_prof_result(
            torch_path,
            funcs,
            warmup,
            active,
            target_kernel_name=target_kernel_name,
            clear_l2_cache=clear_l2_cache,
            _return_samples=_return_samples,
        )
    finally:
        if clear_l2_cache:
            del buffer
        _rm_dic(keep_res, torch_path)


def _rm_dic(keep_res, torch_path):
    if keep_res:
        return
    import shutil

    if os.path.exists(torch_path):
        shutil.rmtree(torch_path)


def _read_profile_samples(directory, names, warmup, active, clear_l2_cache):
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
            for row in csv.DictReader(stream):
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


def _collect_prof_result(
    base_dir: str,
    funcs,
    num_warmup: int,
    num_active: int,
    target_kernel_name: Optional[str] = None,
    clear_l2_cache: bool = False,
    _return_samples: bool = False,
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
    """

    if _return_samples or isinstance(target_kernel_name, (list, tuple)):
        samples = _read_profile_samples(base_dir, target_kernel_name, num_warmup, num_active, clear_l2_cache)
        if _return_samples:
            return samples
        # Per-configuration names use the same arithmetic mean as the existing
        # shared-name collector, while excluding unrelated preparation kernels.
        costs = [float(durations.mean()) / 1000 for _, durations in samples]
        return costs[0] if len(funcs) == 1 else costs

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

    time_cost = [0] * num_funcs
    for func_idx in np.arange(0, num_funcs):
        for active_index in np.arange(0, num_active):
            row_index = func_idx * (num_warmup + num_active) + num_warmup + active_index
            time_cost[func_idx] += filter_df.iloc[row_index]["Duration(us)"]
    time_cost = [x / num_active / 1e3 for x in time_cost]

    if num_funcs == 1:
        return time_cost[0]
    else:
        return time_cost
