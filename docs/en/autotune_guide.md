# Triton-Ascend Autotune Guide

## Positioning

This guide is intended for users who already know how to write Triton kernels and already understand the basic concepts of community `triton.autotune`. It focuses on the recommended Triton-Ascend usage:

- the recommended autotune usage on Triton-Ascend;
- what `configs=[]` means on the Ascend backend;
- when automatic tiling is a good fit, and when you should fall back to handwritten `triton.Config`.

## Recommended Usage

On Triton-Ascend, the recommended usage is to keep the community-style `@triton.autotune` interface and set `configs=[]` when you want the backend to generate and evaluate candidate configurations automatically:

```python
import triton
import triton.language as tl
import triton.backends.ascend.runtime


@triton.autotune(
    configs=[],
    key=["M", "N"],
)
@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    ...
```

This means:

- `key` has the same semantics as in community Triton and determines which runtime-input changes trigger re-selection of a configuration;
- `configs=[]` on Triton-Ascend means "let the Ascend backend generate candidate configurations and tune them", not "there are no configurations".

## Prerequisites

### 1. Enable the Ascend autotune extension first

Only after importing the following line will Triton-Ascend enter the autotune extension path:

```python
import triton.backends.ascend.runtime
```

Without this step, you are still using the community `triton.autotune`, and `configs=[]` does not trigger Ascend automatic tiling.

### 2. `@triton.autotune` must wrap `@triton.jit` directly

It must be written in the following order:

```python
@triton.autotune(configs=[], key=["M", "N"])
@triton.jit
def kernel(...):
    ...
```

`@triton.autotune` must wrap `@triton.jit` directly, and no other decorator should be inserted between them. Otherwise, the kernel DSL may not be parsed correctly, and Triton-Ascend may fail to enter the automatic-tiling generation and tuning path.

### 3. `key` has the same meaning as in community Triton

`key` is effectively the autotune cache key. Any parameter included in `key` triggers autotune again when its value changes.

In most cases, `key` contains shape parameters such as `M/N/K`, `seq_len`, or `hidden_size`, because they often have a significant impact on the optimal tiling choice. However, `key` is not limited to shape parameters. If another parameter affects configuration selection, it can also be included.

### 4. Do not fix parameters that should participate in automatic tuning too early

If you want a `tl.constexpr` parameter to participate in automatic tiling generation, all of the following must be true:

- it must be a tiling parameter, meaning that it affects how much data each block processes or how tile sizes are formed;
- it must not be fixed explicitly in the launch call;
- it must not have a default value in the kernel definition.

For example, in the following code, `BLOCK_M` participates in automatic tuning:

```python
kernel[grid](
    x,
    y,
    out,
    M,
    N,
)
```

If you explicitly pass:

```python
kernel[grid](
    x,
    y,
    out,
    M,
    N,
    BLOCK_M=128,
)
```

then the parameter is already fixed and is no longer part of the automatic-generation space.

Likewise, if a tunable parameter is given a default value in the kernel definition, for example:

```python
@triton.jit
def kernel(
    ...,
    BLOCK_M: tl.constexpr = 128,
):
    ...
```

then the parameter does not participate in automatic tuning. For parameters that should be generated and tuned automatically by the framework, keep them as `tl.constexpr` parameters that are neither passed explicitly at launch time nor given default values in the kernel definition.

### 5. If a tiling parameter affects the grid, the grid must be written as a lambda

If a tiling parameter affects the grid size, the grid must not be written as a fixed value or a static expression that depends only on runtime parameters. Instead, it must be written as a `lambda` that depends on meta parameters, just as in community autotune.

```python
grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
```

The reason is that when autotune evaluates different candidate configurations, parameters such as `BLOCK_M` change. If the grid does not change with the candidate configuration, Triton-Ascend cannot guarantee that each candidate is launched correctly.

## Practical Notes

The benchmark semantics are the same as in community autotune: the kernel may be executed multiple times. If the kernel has side effects, such as atomic operations, in-place writes, or accumulated modifications to input or output buffers, the existing community hook mechanism is still required.

## What Triton-Ascend Adds Beyond Community Autotune

The typical community `triton.autotune` workflow is that the user provides a set of handwritten `triton.Config` objects, the framework benchmarks them, and the best result is cached.

Triton-Ascend keeps the same usage style but extends it in several important ways.

### 1. Support for `configs=[]` to generate candidate configurations automatically

This is the core extension. Users do not have to handwrite a list of `triton.Config` objects up front. Instead, they can leave `configs` empty and let the Ascend backend generate candidate configurations from the kernel DSL semantics and runtime shapes.

### 2. Parallel compilation of multiple candidates

When autotune needs to evaluate multiple candidate configurations, Triton-Ascend compiles them in parallel by default to reduce first-tune latency.

This behavior is enabled by default and can be disabled with `TRITON_AUTOTUNE_PARALLEL_COMPILE=0`.

### 3. Support for profiler-based performance collection

Triton-Ascend supports switching the performance-collection method used during autotune benchmark. In addition to the default benchmark path, you can use profiler-based collection for each candidate configuration. This path focuses on on-chip execution time and is more accurate than the default method for short-running kernels, but it also increases autotune time. This feature can be enabled by setting `TRITON_BENCH_METHOD="npu"`.

## Scope and Behavior of Automatic Tiling Generation

The first extension above shows that Triton-Ascend supports automatic generation of candidate configurations with `configs=[]`. The following points further clarify the behavior of this capability.

### 1. Automatic generation focuses on tiling parameters

Automatic generation in Ascend focuses on tiling-related `tl.constexpr` parameters, that is, parameters that affect how much data each block processes or how tile sizes are formed.

This capability does not mean "automatically tune every parameter". Parameters such as `num_warps` and `num_stages`, as well as non-tiling kernel parameters, are outside the current automatic-generation scope.

### 2. Candidate configurations are filtered with Ascend hardware constraints

When generating candidates, the Ascend backend filters them using NPU-specific constraints such as on-chip memory capacity, alignment constraints, and core-utilization limits, instead of simply enumerating many configurations and benchmarking them blindly.

### 3. The goal of automatic tiling is to provide a reasonably good configuration

To balance tuning cost and tuning result quality, current automatic tiling prunes the candidate space heavily. As a result, the automatically generated result is not guaranteed to reach the upper bound of carefully hand-tuned performance.

The goal of this feature is to lower the usage barrier and tuning cost while making it easy to obtain a configuration with reasonably good performance.

### 4. When automatic tiling fails, users need handwritten `triton.Config`

If automatic tiling cannot generate any valid candidate configurations, you need to switch back to handwritten `triton.Config`. It is also recommended to file an issue for such cases so that Triton-Ascend can improve parsing and automatic-generation coverage later.

## Handwritten `triton.Config` Mode

If automatic tiling fails, or if the generated tiling result does not meet your performance target, you can return directly to the standard community-style handwritten configuration path. Triton-Ascend keeps this part of the interface compatible:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}),
    ],
    key=["M", "N"],
)
@triton.jit
def kernel(...):
    ...
```

In this mode:

- configurations are defined explicitly by the user;
- the framework still handles benchmarking, best-config selection, and cache reuse;
- the usage pattern remains consistent with community Triton autotune.

## Advanced Usage: Combine Automatic Tiling with Other Tunable Parameters

The following content is advanced usage and should only be considered when you want to continue tuning non-tiling kernel parameters or compilation parameters together with automatic tiling.

### 1. Community autotune: enumerate all tunable parameters manually

In the standard community style, users enumerate all candidate configurations manually:

```python
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "BLOCK_K": BK,
                "GROUP_SIZE_M": GS,
            },
            num_warps=num_warps,
        )
        for BM in [16, 32, 64]
        for BN in [16, 32, 64]
        for BK in [16, 32, 64]
        for GS in [1, 2, 4, 8]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(a, b, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M):
    ...
```

### 2. Triton-Ascend: enumerate tiling parameters and Ascend compilation parameters manually

In Triton-Ascend, if you still want to use manual enumeration, you can also put Ascend-side parameters into the handwritten configuration space:

```python
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "BLOCK_K": BK,
                "GROUP_SIZE_M": GS,
                "multibuffer": MS,
            }
        )
        for BM in [16, 32, 64]
        for BN in [16, 32, 64]
        for BK in [16, 32, 64]
        for GS in [1, 2, 4, 8]
        for MS in [False, True]
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(a, b, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M):
    ...
```

### 3. Triton-Ascend: generate tiling automatically while tuning other parameters jointly

If you want tiling parameters to continue being generated automatically through `configs=[]`, but also want to tune additional non-tiling parameters or compilation parameters, you can pass those extra search dimensions through `hints`:

```python
@triton.autotune(
    configs=[],
    key=["M", "N", "K"],
    hints={
        "GROUP_SIZE_M": [1, 2, 4, 8],
        "multibuffer": [False, True],
    },
)
@triton.jit
def matmul_kernel(a, b, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M):
    ...


matmul_kernel[grid](a, b, M, N, K)
```

This means:

- tiling-related parameters are still generated automatically by Triton-Ascend;
- non-tiling parameters or compilation parameters (such as GROUP_SIZE_M and multibuffer) are provided explicitly by the user through `hints`;
- autotune evaluates the combined search space of both parts.

## Optional NPU benchmark policies

With `TRITON_BENCH_METHOD=npu`, the built-in NPU profiler can be configured through
`npu_bench_options` on `autotune` or `max_autotune`. The same mapping is accepted
by `do_bench_npu` for standalone profiling. Each field uses the following priority:
explicit mapping value, corresponding environment variable, then the existing benchmark setting.
`None` leaves a field unspecified; `False` and `0` are explicit overrides.
Custom `do_bench` callbacks retain control of their measurements and do not use
these autotuner policies.

```python
@triton.autotune(
    configs=configs,
    key=["N"],
    npu_bench_options={
        "cache_mode": "hot",
        "quality_check": True,
        "max_retries": 4,
        "active": 150,
        "measure_budget_ms": 5.0,
        "filter_slow_configs": True,
    },
)
@triton.jit
def kernel(x, y, N, BLOCK: tl.constexpr):
    ...
```

All policies are independent and optional. Enabling quality checks does not
enable retries, calibration or slow-configuration filtering, or change launch
counts and cache state. Without new settings, autotuning retains cold L2 and the
existing measurement counts (normally 5 warmup and 30 active launches; the CV
time estimate may derive its own counts). Standalone `do_bench_npu` retains its existing
`clear_l2_cache=False` default unless `cache_mode` is specified.

| Mapping field | Environment variable | Default / meaning |
| --- | --- | --- |
| `cache_mode` | `TRITON_NPU_BENCH_CACHE_MODE` | Inherit existing cache setting. `cold` evicts L2; `hot` warms up and repeats without eviction. |
| `warmup` | `TRITON_NPU_BENCH_WARMUP` | Inherit existing warmup launch count; integer >= 0. With explicit `active` or a budget, defaults to `5` in autotuning. |
| `active` | `TRITON_NPU_BENCH_ACTIVE` | Inherit existing measured launch count; integer >= 1, or >= 2 with quality checks. With a budget, defaults to `30` in autotuning and sets the minimum count per candidate. |
| `quality_check` | `TRITON_NPU_BENCH_QUALITY_CHECK` | `False`; evaluate predefined sample-quality metrics. |
| `max_retries` | `TRITON_NPU_BENCH_MAX_RETRIES` | `0`; additional attempts for failed quality checks or unusable profiler data. |
| `measure_budget_ms` | `TRITON_NPU_BENCH_MEASURE_BUDGET_MS` | Unset; optional minimum accumulated device time per candidate, not a wall-clock timeout. |
| `calibration_runs` | `TRITON_NPU_BENCH_CALIBRATION_RUNS` | `10`; used only when a measurement budget is supplied. |
| `filter_slow_configs` | `TRITON_NPU_BENCH_FILTER_SLOW_CONFIGS` | `False`; independently exclude especially slow candidates using rough host timings. |
| `slow_config_runs` | `TRITON_NPU_BENCH_SLOW_CONFIG_RUNS` | `20`; initial rough launches for slow-configuration filtering. |
| `slow_config_recheck_runs` | `TRITON_NPU_BENCH_SLOW_CONFIG_RECHECK_RUNS` | `40`; recheck slow candidates before excluding them. |
| `slow_config_factor` | `TRITON_NPU_BENCH_SLOW_CONFIG_FACTOR` | `5.0`, >= 1; candidates above this multiple of the best rough time are rechecked. |
| `verbose` | `TRITON_NPU_BENCH_VERBOSE` | Inherit `TRITON_PRINT_AUTOTUNING`; print policy and per-attempt diagnostics. |

Boolean environment values accept `0`, `1`, `false`, and `true`. Generic
`prune_configs` still runs first. Without explicit `active` or
`measure_budget_ms`, applicable CV cases retain the existing calculation of
`cv_warmup/cv_repeat` and candidate pruning by an estimated 200-second measurement
cost. This estimate is not a strict timeout for the whole autotuning process.
An explicit `warmup` alone preserves that calculation and is included in the
time estimate.

An explicit `active` or `measure_budget_ms` from the mapping or environment
bypasses both the CV count calculation and its time-limit candidate pruning.
Even `active=30` selects this behavior; internally derived counts do not.
Unspecified autotuning counts then default to `warmup=5` and `active=30`.
Standalone profiling retains its `warmup` and `active` arguments as defaults.

Slow-configuration filtering is independent of count selection. If enabled,
it runs on the remaining candidates before calibration and profiling. It uses
the best rough host time, including synchronization, and rechecks candidates
above `slow_config_factor` times the best candidate's time before excluding them.

With a budget, calibration determines `duration_i_ms` for each candidate, then
`active_i = max(active, ceil(measure_budget_ms / duration_i_ms))`. For example,
`active=30` and a `5 ms` budget give candidates taking `0.01 ms` and `1 ms`
respectively **500 and 30 measured launches**. Candidates with the same count
are profiled together; retries retain each candidate's calculated count.

The former `pruning`, `prune_runs`, `prune_recheck_runs`, and `prune_factor`
mapping fields and their `TRITON_NPU_BENCH_*` variables are replaced by the
slow-configuration options above.

An explicit `cache_mode="cold"` composes cache eviction after the autotuner's
existing preparation hooks during profiling. Those hooks and the `kernel_call`
interface are preserved. `hot` does not guarantee that the working set fits in
L2; it means that the benchmark does not evict it between measured launches.

Quality checks cover tail contamination, gaps, multimodality, drift, and change
points, using the predefined thresholds in `_benchmark_quality.py`. Candidate
ranking keeps the existing arithmetic mean of kernel durations. Only failing
candidates are remeasured. After exhausting retries, the fastest collected
measurement remains eligible, with a warning if quality checks failed or some
candidates still lack usable data. If no usable measurements were collected,
profiling raises an error. Kernel execution errors propagate without retries.
An explicit policy does not fall back to event timing with different cache
conditions. Effective policies participate in the autotune result cache key.

The separate `npu-strict` mode is removed. Select `TRITON_BENCH_METHOD=npu` and
enable the desired fields above; `TRITON_NPU_STRICT_*` variables are replaced by
`TRITON_NPU_BENCH_*` variables. Existing arguments take precedence over env
values, including overrides that disable a feature enabled in the environment.

## Summary

Triton-Ascend extends community autotune with automatic tiling-candidate generation, joint parameter tuning, and optional NPU benchmark policies. For most users, the recommended usage is:

- keep the community-style `@triton.autotune` interface;
- set `configs=[]`;
- let the Ascend backend generate, filter, benchmark, and cache candidate configurations automatically based on the kernel DSL and runtime shapes.

If your case is not suitable for automatic tiling, you can return to handwritten `triton.Config`.
