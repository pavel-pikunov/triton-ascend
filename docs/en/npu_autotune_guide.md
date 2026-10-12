# NPU autotune policies and diagnostics

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
| `calibration_runs` | `TRITON_NPU_BENCH_CALIBRATION_RUNS` | `10`; event calibration launches with a budget when no slow-filter estimate is available. |
| `filter_slow_configs` | `TRITON_NPU_BENCH_FILTER_SLOW_CONFIGS` | `False`; independently exclude especially slow candidates using rough host timings. |
| `slow_config_runs` | `TRITON_NPU_BENCH_SLOW_CONFIG_RUNS` | `20`; initial rough launches for slow-configuration filtering. |
| `slow_config_recheck_runs` | `TRITON_NPU_BENCH_SLOW_CONFIG_RECHECK_RUNS` | `40`; recheck slow candidates before excluding them. |
| `slow_config_factor` | `TRITON_NPU_BENCH_SLOW_CONFIG_FACTOR` | `5.0`, >= 1; candidates above this multiple of the best rough time are rechecked. |
| `slow_config_recheck_delay_s` | `TRITON_NPU_BENCH_SLOW_CONFIG_RECHECK_DELAY_S` | `0.5`; finite seconds >= 0 between the first rough-measurement round and the recheck round. `0` disables the pause. |
| `log_level` | `TRITON_NPU_BENCH_LOG_LEVEL` | `off`; `brief` prints stages and candidate counts, `detailed` also prints configuration and quality diagnostics. |

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
The first round measures all remaining candidates. If any require rechecking,
one pause of `slow_config_recheck_delay_s` seconds separates the completed first
round from the recheck round, reducing the chance that both sample the same
period of temporary degradation. Only suspect candidates are rechecked. No
pause is taken with fewer than two candidates or when no recheck is needed.
The delay is configurable independently of launch counts; it is inactive when
filtering is disabled and affects the cache key only when filtering is enabled.
This wall-clock pause is separate from the optional device-time measurement budget.

With a budget, calibration determines `duration_i_ms` for each candidate, then
`active_i = max(active, ceil(measure_budget_ms / duration_i_ms))`. For example,
`active=30` and a `5 ms` budget give candidates taking `0.01 ms` and `1 ms`
respectively **500 and 30 measured launches**. All current candidates share one
profiler session per attempt, each with its own count. The common CSV is read
once and split by cumulative launch counts in candidate order, including when
kernel names are identical. Warmup rows are excluded per candidate. Calibration
uses NPU events, without a profiler session or CSV. With slow filtering enabled,
event times are collected in the same rough-measurement rounds and reused for
surviving candidates; rechecked candidates use their second round's event mean.
Otherwise, one preliminary call and `calibration_runs` event measurements are
performed per candidate. A single candidate skipped by the slow filter also
uses this short event calibration. Event preparation follows the requested
hot/cold cache policy, excluding explicit eviction from the device event span.
Event estimates control launch counts only; final scores still come from the
profiler. When filter samples are reused, their time is counted in `Slow filter`,
and `Calibration` contains count calculation only. Retries retain each
candidate's calculated count, without recalibration.
A failure to acquire the common CSV retries the entire current batch.

An explicit `cache_mode="cold"` composes cache eviction after the autotuner's
existing preparation hooks during profiling. Those hooks and the `kernel_call`
interface are preserved. `hot` does not guarantee that the working set fits in
L2; it means that the benchmark does not evict it between measured launches.

Quality checks cover tail contamination, gaps, multimodality, drift, and change
points, using the predefined thresholds in `_benchmark_quality.py`. Candidate
ranking uses the mean of the central 50% of measured kernel durations. Only failing
candidates are remeasured. After exhausting retries, the fastest collected
measurement remains eligible, with a warning if quality checks failed or some
candidates still lack usable data. If no usable measurements were collected,
profiling raises an error. Kernel execution errors propagate without retries.
An explicit policy does not fall back to event timing with different cache
conditions. Effective policies participate in the autotune result cache key.

### Warnings

The tuner emits `RuntimeWarning` once per reason per instance, attributed to the
kernel's definition. Python warning filters (`default`, `ignore`, and `error`)
apply. Explicit measurement settings are diagnosed when ignored by another
benchmark method, a custom `do_bench`, or selection with only one surviving
configuration. `calibration_runs` is inactive without a budget; slow-filter
fields are inactive when `filter_slow_configs=False`. These inactive fields and
`log_level` do not change measurement routing or cache keys.

Missing NPU profiler CSV and non-finite final scores from the built-in NPU
benchmarker also warn with the fallback and a suggested action.
The optional `auto_prof_dir` winner profile uses environment settings and warns
when explicit `npu_bench_options` are not forwarded to it. Cleanup and report
diagnostics preserve an exception already in flight, even with warnings as errors.

### Stage logging

`log_level="brief"` prints the initial combination count, stage starts, compilation
and pruning results, whether the slow filter needs a second round, profiler
attempt numbers, and retry candidate counts. `detailed` additionally prints
settings, per-candidate launch counts, mean durations, quality metrics and failure
reasons. Neither level prints individual launches or raw CSV rows. The default
is `off`; an explicit field overrides `TRITON_NPU_BENCH_LOG_LEVEL`.
This log is independent of `TRITON_PRINT_AUTOTUNING`, which retains its existing
debug behavior. Disable that legacy flag when using these examples:

```bash
export TRITON_PRINT_AUTOTUNING=0
export TRITON_NPU_BENCH_LOG_LEVEL=brief
```

Retry-exhaustion diagnostics remain `RuntimeWarning` and respect Python warning
filters. An emitted warning is yellow only on a TTY; redirected files contain
no ANSI color codes and the log does not print a duplicate warning.

### Selected-configuration report

To print a detailed report after selecting a configuration, set
`report_best_config=True` on `autotune` or `max_autotune`, or set
`TRITON_NPU_BENCH_REPORT_BEST_CONFIG=1`. This diagnostic option is separate from
`npu_bench_options` and `log_level`. An explicit argument overrides the environment;
the default is disabled. Environment values accept `0`, `1`, `false`, and `true`.

```python
@triton.autotune(
    configs=configs,
    key=["N"],
    report_best_config=True,
    npu_bench_options={"cache_mode": "hot", "active": 30},
)
@triton.jit
def kernel(x, y, N, BLOCK: tl.constexpr):
    ...
```

The report prints the configuration's **1-based number in the original candidate
list before pruning**, all configuration fields (including defaults and `None`),
and `ubtune_cfg` when present. With NPU profiler data available, it also prints
the profiler kernel name, central-half score and ordinary mean duration in
microseconds and milliseconds, cache
mode, warmup and measured launch counts, and the selected measurement attempt.
Numeric profiler metrics are averaged from the already-read measured rows of
that attempt. IDs and absolute timestamps are never averaged. Constant text
fields appear once; changing text fields appear as `varies`. Incomplete metrics
show the number of available values, without substituting zeros. Raw rows are
not retained for reporting, and the CSV is not read again.

CSV columns depend on the profiler's settings and version. Reporting enables
no additional metrics or profiling runs. With retries, the report describes the
attempt whose central-half score was retained for ranking, including its quality metrics and
any failed checks. It does not combine attempts.

For the ordinary benchmarker the report prints the returned median and 20%/80%
quantiles. For a custom `do_bench` it prints the returned score without assigning
units or interpreting its meaning. When measurements were skipped or unavailable,
the report states that explicitly and performs no additional measurement.
Reporting does not change launch counts, pruning, selection or autotune cache
keys. Cache hits do not repeat it. Temporary files retain their cleanup behavior.

### Autotune timing report

Set `report_timing=True` on either decorator, or
`TRITON_AUTOTUNE_REPORT_TIMING=1`, to print one final block after configuration
selection. An explicit argument overrides the environment; the default is off.
This control is independent of `log_level`, `report_best_config`, and benchmarker.

The block lists total wall time, generation/pruning, compilation, slow filtering,
calibration, main measurements including retries, and other expenses. Measurements
include profiler completion, CSV reading and quality checks. Stages do not overlap;
other expenses are the remaining total. The timing ends at selection and excludes
the optional extra winner profile, final kernel invocation and reporting itself.
There are no intermediate timing blocks and cache hits remain silent.

```python
@triton.autotune(
    configs=configs,
    key=["N"],
    npu_bench_options={"log_level": "brief", "measure_budget_ms": 5},
    report_best_config=True,
    report_timing=True,
)
@triton.jit
def kernel(x, y, N, BLOCK: tl.constexpr):
    ...
```
