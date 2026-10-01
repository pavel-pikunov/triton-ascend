"""Built-in profiler integration for both public NPU autotune decorators."""

import math
import os

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.ascend import _autotune_report, testing
from triton.backends.ascend.runtime import autotuner


@pytest.mark.parametrize("decorator", ["autotune", "max_autotune"])
def test_npu_vector_tuning_and_cache(monkeypatch, capsys, tmp_path, decorator):
    for name in tuple(os.environ):
        if name.startswith("TRITON_NPU_BENCH_") or name == "TRITON_PRINT_AUTOTUNING":
            monkeypatch.delenv(name)
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    monkeypatch.setenv("TRITON_AUTOTUNE_PARALLEL_COMPILE", "0")
    monkeypatch.setattr(testing.runtime.cache, "get_home_dir", lambda: str(tmp_path))
    profiles, reports = [], []
    original_profile = testing._profile_npu
    original_report = _autotune_report.print_best_config_report

    def profile(*args, **kwargs):
        profiles.append(True)
        return original_profile(*args, **kwargs)

    def report(function, configs, selected, measurement):
        reports.append(measurement)
        return original_report(function, configs, selected, measurement)

    monkeypatch.setattr(testing, "_profile_npu", profile)
    monkeypatch.setattr(_autotune_report, "print_best_config_report", report)

    @triton.jit
    def add_one(x, y, N, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(x + offsets, offsets < N, other=0)
        tl.store(y + offsets, values + 1, offsets < N)

    configs = [triton.Config({"BLOCK": block}, num_stages=1) for block in (64, 128)]
    kwargs = {"kernel_type": "vector", "enable_ubuf_saving": [False]} if decorator == "max_autotune" else {}
    kernel = getattr(autotuner, decorator)(
        configs,
        ["N"],
        npu_bench_options={"cache_mode": "hot", "warmup": 0, "active": 2},
        report_best_config=True,
        **kwargs,
    )(add_one)
    assert len(kernel.configs) == 2
    x = torch.arange(257, dtype=torch.float32, device="npu")
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta["BLOCK"]), )
    kernel[grid](x, y, x.numel())
    torch.npu.synchronize()
    torch.testing.assert_close(y, x + 1)
    assert len(profiles) == len(reports) == 1
    measurement = reports[0]
    assert measurement is not None and len(measurement.rows) == 2
    assert measurement.cache_mode == "hot" and measurement.warmup == 0 and measurement.active == 2
    assert math.isfinite(measurement.mean_ms) and measurement.mean_ms > 0
    assert all(row["Name"] and float(row["Duration(us)"]) > 0 for row in measurement.rows)
    assert "Selected config:" in capsys.readouterr().out
    selected = kernel.best_config
    y.zero_()
    kernel[grid](x, y, x.numel())
    torch.npu.synchronize()
    torch.testing.assert_close(y, x + 1)
    assert kernel.best_config is selected and len(kernel.cache) == 1
    assert len(profiles) == len(reports) == 1 and capsys.readouterr().out == ""
