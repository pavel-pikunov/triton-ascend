"""Local fixtures using the installed Ascend backend and real JIT functions."""

import csv
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.ascend import _autotune_report, _npu_benchmark, _npu_profiler, testing
from triton.backends.ascend.runtime import autotuner
from triton.runtime.driver import driver


@pytest.fixture
def backend(monkeypatch, tmp_path):
    from triton.backends.ascend import _benchmark_quality

    for name in tuple(os.environ):
        if name.startswith(_npu_benchmark.ENV_PREFIX) or name in {
                "TRITON_PRINT_AUTOTUNING", "TRITON_ENABLE_UBTUNER", "TRITON_AUTOTUNE_PARALLEL_COMPILE"
        }:
            monkeypatch.delenv(name)
    monkeypatch.setenv("TRITON_BENCH_METHOD", "default")
    monkeypatch.setattr(torch.npu, "synchronize", lambda: None)
    monkeypatch.setattr(testing.runtime.cache, "get_home_dir", lambda: str(tmp_path))
    return SimpleNamespace(policy=_npu_benchmark, quality=_benchmark_quality, testing=testing, report=_autotune_report,
                           profiler=_npu_profiler)


@pytest.fixture
def jit_kernel():

    @triton.jit
    def kernel(x, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        values = tl.load(x + offsets)
        tl.store(x + offsets, values)

    return kernel


@pytest.fixture
def make_tuner(backend, monkeypatch, jit_kernel):

    def unexpected_benchmark(*args, **kwargs):
        pytest.fail("unexpected event benchmark")

    monkeypatch.setattr(driver.active, "get_benchmarker", lambda: unexpected_benchmark)

    def create(options=None, configs=None, **kwargs):
        tuner = autotuner.autotune(configs if configs is not None else [triton.Config({})], [],
                                   npu_bench_options=options, **kwargs)(jit_kernel)
        monkeypatch.setattr(jit_kernel, "run",
                            lambda *a, **k: SimpleNamespace(packed_metadata={"kernel_name": "kernel"}))
        tuner.compile_parallel = False
        tuner.nargs = {}
        # Exercise the CV route without running the shape parser or compiler.
        tuner.parser_mode = "cube"
        tuner.cv_parse_result = object()
        return tuner

    return create


@pytest.fixture
def configure_run(make_tuner, monkeypatch):

    def configure(selected, configs, reports_enabled):
        tuner = make_tuner(configs=configs, report_best_config=reports_enabled)
        monkeypatch.setattr(tuner, "prune_configs", lambda kwargs: [selected, configs[0]])
        monkeypatch.setattr(tuner.fn, "run", lambda *a, **k: "result")
        return tuner

    return configure


@pytest.fixture
def sample():

    def samples(duration=10, count=4):
        return np.arange(count, dtype=float) + 1, np.full(count, duration, dtype=float)

    return samples


@pytest.fixture
def run_policy(backend, tmp_path):

    def run(measure, options, funcs=None, names=None):
        funcs = funcs or [lambda: None, lambda: None]
        return backend.policy.benchmark_with_options(
            measure,
            funcs,
            names or ["a", "b"][:len(funcs)],
            backend.policy.resolve_options(options),
            warmup=5,
            active=30,
            prof_root=tmp_path,
            synchronize=lambda: None,
        )

    return run


@pytest.fixture
def fake_profiler(backend, monkeypatch):

    @contextmanager
    def profile(**kwargs):
        yield

    monkeypatch.setattr(torch_npu.profiler, "profile", profile)
    monkeypatch.setattr(torch_npu.profiler, "tensorboard_trace_handler", lambda path: None)


@pytest.fixture
def write_profile():

    def write(path, groups, warmup, active):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        columns = ("Name", "Type", "Start Time(us)", "Duration(us)", "Unexpected metric(%)", "Task ID")
        with (path / "kernel_details.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerow(["hook", "Other", 0, 1, "unused", "0000"])
            writer.writerow(["flush", "ReduceSum", 1, 1, "unused", "0001"])
            timestamp = 2
            for name, duration in groups:
                for iteration in range(warmup + active):
                    writer.writerow([
                        name, "Kernel", timestamp, "999.000" if iteration < warmup else duration,
                        "warmup" if iteration < warmup else "001.2300", f"{timestamp:04d}"
                    ])
                    timestamp += 1

    return write
