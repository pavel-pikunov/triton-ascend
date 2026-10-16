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

from ._npu_benchmark import ProfilerResultMismatchError as ProfilerResultMismatchError


def do_bench_npu(
    funcs,
    warmup=5,
    active=30,
    clear_l2_cache=False,
    prof_dir=None,
    keep_res=False,
    target_kernel_name: Optional[str] = None,
):
    """Profile NPU kernels with optional TRITON_NPU_BENCH_* environment policies."""
    from ._npu_benchmark import bench_npu
    return bench_npu(_profile_npu, funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name)


def _profile_npu(funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name, *,
                 _return_samples=False, _pre_hook_scope=None, _execution_state=None, _capture_report=False,
                 _quantiles=(0.0, 1.0), _timings=None, _profiler_mode="summary"):
    import torch
    import torch_npu
    from ._npu_benchmark import _collect_prof_result, export_npu_summary, profile_scope, result_scope, timing_stage

    if not isinstance(funcs, list):
        funcs = [funcs]

    def run(fn):
        try:
            fn()
            torch.npu.synchronize()
        except Exception:
            if _execution_state is not None:
                _execution_state["failed"] = True
            raise

    # warmup kernel
    for fn in funcs:
        run(fn)

    summary = _profiler_mode == "summary"
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=(torch_npu.profiler.AiCMetrics.AiCoreNone
                     if summary else torch_npu.profiler.AiCMetrics.PipeUtilization),
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
        base_path = os.path.join(runtime.cache.get_home_dir(), ".triton", "profile_results")
        torch_path = os.path.join(base_path, f"prof_{timestamp}_{process_name}-{pid}")

    if clear_l2_cache:
        buffer = runtime.driver.active.get_empty_cache_for_benchmark()
        buffer = buffer.float()  # to avoid type cast
        buffer.sum()
        torch.npu.synchronize()  # shake out of any npu error

    handler = torch_npu.profiler.tensorboard_trace_handler(torch_path, analyse_flag=not summary)

    def on_trace_ready(prof):
        with timing_stage(_timings, "export"):
            handler(prof)

    try:
        scope = profile_scope(buffer if clear_l2_cache else None, torch.npu.synchronize, _pre_hook_scope,
                              lambda: _rm_dic(keep_res, torch_path))
        with scope, torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                on_trace_ready=on_trace_ready,
                schedule=lambda step: (torch_npu.profiler.ProfilerAction.RECORD_AND_SAVE
                                       if step == 0 else torch_npu.profiler.ProfilerAction.NONE),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
        ) as prof:
            with timing_stage(_timings, "profiler"):
                for fn in funcs:
                    for _ in builtins.range(warmup + active):
                        if clear_l2_cache and _pre_hook_scope is None:
                            buffer.sum()  # use buffer read to clear l2 cache
                            torch.npu.synchronize()
                        run(fn)
            prof.step()  # Complete one recording step for the entire candidate batch.
    finally:
        if clear_l2_cache:
            del buffer

    with result_scope(lambda: _rm_dic(keep_res, torch_path)):
        if summary:
            export_npu_summary(prof.prof_if.prof_path, _timings)
        with timing_stage(_timings, "csv"):
            return _collect_prof_result(
                prof.prof_if.prof_path,
                funcs,
                warmup,
                active,
                target_kernel_name=target_kernel_name,
                clear_l2_cache=clear_l2_cache,
                _return_samples=_return_samples,
                _capture_report=_capture_report,
                _quantiles=_quantiles,
            )


def _rm_dic(keep_res, torch_path):
    if keep_res:
        return
    import shutil

    if os.path.exists(torch_path):
        shutil.rmtree(torch_path)
