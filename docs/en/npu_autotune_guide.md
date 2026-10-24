# Current NPU measurements on the comparison branch

The `npu` route uses the measurement implementation from `StableAutotuner_3.2.2dev`.
The old sample-quality metrics, event calibration and fast filter do not run.
The original CV event-based sorting and time-limit pruning remain in place.
`npu_legacy` is no longer a selectable comparison route; `default` retains the original event tuner.

## Algorithm

1. Compile candidates using the existing compiler path. For CV, perform the original rough event
   measurement, sorting and time-limit pruning, then cap WARMUP/ACTIVE by cv_warmup/cv_repeat.
2. Run M interleaved series in one profiler session: `ABC...`, repeated M times.
   Exclude warmup rows. Sort each series' durations, retain the configured quantile interval,
   and average it to obtain that series' score.
3. Keep each candidate's N smallest scores from its history. Its minimum is confirmed when
   all N scores, including the minimum itself, are at most `minimum * (1 + tolerance)`.
   Retry only candidates whose minima are not confirmed.
4. Outside CV, only before batch 2, calculate a common active count from the fastest ordinary
   mean of batch 1: `max(active, ceil(budget_ms / fastest_mean_ms))`. If active increases,
   discard the previous scores of pending candidates. Confirmed candidates retain their scores.
   There is no later recalculation and no separate calibration session. CV never uses this budget.
5. Select the smallest retained score across candidates. An exhausted candidate still returns
   its minimum and prints that confirmation failed. The utils CSV marks WrongData if the
   winner is unconfirmed or any measured candidate exhausts its batches unconfirmed.

## Measurement flags

Each name below starts with `TRITON_NPU_BENCH_`. Defaults preserve the original dev behavior.

| Suffix | Default | Meaning |
|---|---|---|
| WARMUP | Caller/CV count | Warmup launches; explicit 0 is valid. CV caps it from above. |
| ACTIVE | Caller/CV count | Measured launches per series. CV caps it from above. |
| BATCH_RUNS | 1 | M series per candidate in a profiler batch. |
| STABILITY_RUNS | 1 | N scores needed to confirm the minimum, including itself. 1 disables confirmation. |
| STABILITY_TOLERANCE | 0.01 | Relative tolerance above the minimum; 0.01 means 1%. |
| LOWER_QUANTILE | 0 | Lower retained duration quantile. |
| UPPER_QUANTILE | 1 | Upper retained duration quantile. 0.25/0.75 retains the central half. |
| MEASURE_BUDGET_MS | 0 | Device-time budget for subsequent vector batches. 0 disables it. |
| MAX_RETRIES | 0 | Maximum batches in total. 0 permits one initial batch. |
| CACHE_MODE | Caller mode | hot disables explicit eviction; cold retains the existing eviction mechanism. |
| PIPE_UTILIZATION | 1 | 0 disables collection of pipe-utilization metrics. |
| SINGLE_CONFIG | 0 | 1 also measures a single autotune candidate with NPU profiling. |

Without new env, an initial single config skips measurement, while a single surviving
compiled candidate uses the original event benchmark. SINGLE_CONFIG does not gate direct do_bench_npu calls.
Measurement env is not part of the selection-cache key; use a new Python process after changing it.
TRITON_ALWAYS_COMPILE controls compiler caching, not the tuner's selection cache.

## Example

```bash
export TRITON_BENCH_METHOD=npu
export TRITON_NPU_BENCH_ACTIVE=30
export TRITON_NPU_BENCH_WARMUP=5
export TRITON_NPU_BENCH_BATCH_RUNS=5
export TRITON_NPU_BENCH_STABILITY_RUNS=5
export TRITON_NPU_BENCH_STABILITY_TOLERANCE=0.01
export TRITON_NPU_BENCH_LOWER_QUANTILE=0.25
export TRITON_NPU_BENCH_UPPER_QUANTILE=0.75
export TRITON_NPU_BENCH_MEASURE_BUDGET_MS=15
export TRITON_NPU_BENCH_MAX_RETRIES=5
export TRITON_NPU_BENCH_CACHE_MODE=hot
export TRITON_NPU_BENCH_PIPE_UTILIZATION=0
export TRITON_NPU_BENCH_SINGLE_CONFIG=1
```

The former QUALITY_CHECK, QUALITY_CHECK_ALL, STABILITY_ONLY, FILTER_SLOW_CONFIGS,
CALIBRATION_RUNS, SLOW_CONFIG_* and PROFILER_MODE policies do not control this NPU path.
Remove them from old shell scripts. Use STABILITY_RUNS=1 for a benchmark without confirmation,
and PIPE_UTILIZATION=0 for the lighter metrics configuration.

[Test-only repetitions, CSV and reports](npu_benchmark_comparison.md) remain available.
