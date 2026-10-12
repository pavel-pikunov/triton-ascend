# NPU autotune policies and diagnostics

Set `TRITON_BENCH_METHOD=npu` before running Python to configure the built-in
NPU profiler used by `autotune` and `max_autotune`. All new controls use environment
variables; the decorator signatures are unchanged. Unset fields inherit existing
benchmark settings. The ordinary and custom benchmarkers ignore these controls.
Standalone `do_bench_npu` also reads the NPU measurement environment.

```bash
export TRITON_BENCH_METHOD=npu
export TRITON_NPU_BENCH_CACHE_MODE=hot
export TRITON_NPU_BENCH_MEASURE_BUDGET_MS=5
export TRITON_NPU_BENCH_QUALITY_CHECK=1
export TRITON_NPU_BENCH_MAX_RETRIES=4
export TRITON_NPU_BENCH_FILTER_SLOW_CONFIGS=1
export TRITON_NPU_BENCH_SLOW_CONFIG_ROUNDS=2
export TRITON_NPU_BENCH_LOG_LEVEL=brief
export TRITON_NPU_BENCH_REPORT_BEST_CONFIG=1
export TRITON_AUTOTUNE_REPORT_TIMING=1
export TRITON_PRINT_AUTOTUNING=0
```

## Measurement settings

| Environment variable | Default / meaning |
| --- | --- |
| `TRITON_NPU_BENCH_CACHE_MODE` | Inherit the caller's cache policy. `cold` evicts L2; `hot` does not explicitly evict it between profiler launches. |
| `TRITON_NPU_BENCH_WARMUP` | Inherit the warmup launch count; integer >= 0. |
| `TRITON_NPU_BENCH_ACTIVE` | Inherit the active launch count; integer >= 1, or >= 2 with quality checks. |
| `TRITON_NPU_BENCH_QUALITY_CHECK` | `0`; evaluate the predefined sample-quality metrics. |
| `TRITON_NPU_BENCH_MAX_RETRIES` | `0`; additional attempts for failing quality checks or unusable profiler data. |
| `TRITON_NPU_BENCH_MEASURE_BUDGET_MS` | Unset; estimated accumulated device-time budget per candidate, finite and positive, in milliseconds. |
| `TRITON_NPU_BENCH_CALIBRATION_RUNS` | `10`; short event-calibration launch count when no slow-filter result is available; integer >= 2. |
| `TRITON_NPU_BENCH_FILTER_SLOW_CONFIGS` | `0`; enable preliminary filtering through ordinary `triton.testing.do_bench`. |
| `TRITON_NPU_BENCH_SLOW_CONFIG_ROUNDS` | `2`; total filtering rounds, including the first round; integer >= 1. |
| `TRITON_NPU_BENCH_SLOW_CONFIG_FACTOR` | `5.0`; recheck candidates above this multiple of the best first-round time; finite and >= 1. |
| `TRITON_NPU_BENCH_SLOW_CONFIG_RECHECK_DELAY_S` | `0.5`; finite, nonnegative pause before each needed subsequent round. `0` disables the pause. |

Boolean values accept `0`, `1`, `false`, and `true`. Settings do not enable one
another: quality checks do not automatically enable retries, filtering, or a budget.

Without explicit counts, NPU autotuning normally retains `warmup=5`, `active=30`,
and cold cache. Applicable CV cases retain their existing count calculation and
200-second estimated measurement cap. Explicit `ACTIVE` or `MEASURE_BUDGET_MS`
bypasses that calculation and cap; unspecified counts then use 5 and 30.
Explicit `WARMUP` alone preserves CV count calculation and enters its estimate.
Standalone profiling retains its existing `warmup`, `active`, and
`clear_l2_cache=False` argument defaults. Hot mode cannot guarantee L2 residency.

## Slow filtering and calibration

The first filtering round benchmarks every candidate with ordinary `do_bench`
and its default warmup/repetition time budgets, returning the mean event duration.
The threshold is the best first-round duration multiplied by `SLOW_CONFIG_FACTOR`.
Subsequent rounds benchmark only candidates still above that threshold. A candidate
that recovers is retained and is not measured in later filtering rounds. Candidates
still slow after the configured number of rounds are excluded. One round permits
immediate filtering; the default two rounds provide one recheck.

There is one pause before each needed recheck round. No pause or recheck occurs
without suspects, with fewer than two candidates, or with one configured round.
`SLOW_CONFIG_RUNS` and `SLOW_CONFIG_RECHECK_RUNS` are not used: ordinary `do_bench`
derives launch counts from time budgets, rather than those former launch counts.

Ordinary `do_bench` clears L2 independently of the main profiler cache policy.
Thus filtering uses its standard cold-cache event measurements even if the main
profiler uses `CACHE_MODE=hot`.

A measurement budget sets
`active_i = max(active, ceil(measure_budget_ms / estimated_duration_i_ms))`.
Surviving candidates reuse their latest slow-filter event means as estimates.
When those results are unavailable, one preliminary call and `CALIBRATION_RUNS`
event measurements estimate each candidate under the requested profiler cache
policy. Calibration creates no profiler session or CSV. Event estimates determine
counts only; final ranking uses profiler kernel durations. The budget is an estimate,
not a guarantee of accumulated kernel time or a wall-clock timeout.

## Profiler attempts and selection

All current candidates share one profiler session per attempt, including when
their active counts differ. The original pandas collector reads the common CSV
once, filters target kernels and cache operations, and splits candidates by
cumulative counts. Each candidate's warmup rows are excluded. Identical target
kernel names are supported.

The selection score is the mean of the middle half of sorted measured durations.
Quality checks cover tail contamination, gaps, multimodality, drift, and change
points using the existing algorithms and thresholds in `_benchmark_quality.py`.
Only failing candidates receive quality retries. Failed acquisition of the shared
CSV retries the current batch. Counts are retained across retries without recalibration.
Kernel execution and synchronization failures propagate without retries.

Each candidate retains its minimum measured score regardless of quality. A faster
failed-quality attempt remains eligible after a slower successful attempt. If the
selected measurement failed quality checks, one plain-text message states the
autotuner's uncertainty independently of the logging level. It does not change the
winner or raise a warning exception. No ANSI coloring or global warning handler is used.

Existing unavailable-data behavior is retained: a simple missing-CSV result is
`inf`; the retry path raises when no usable measurements were acquired. The
unconfigured tuner retains its existing event fallback on target-row mismatch;
explicit measurement policies do not fall back to different measurement conditions.

Effective measurement policies participate in the selection-cache key. Logging
and reports do not. Calibration settings are inactive without a budget; filtering
settings are inactive when filtering is disabled. A recheck delay is inactive with
one filtering round. No additional warnings are emitted for inactive settings.

## Console diagnostics

| Environment variable | Default / output |
| --- | --- |
| `TRITON_NPU_BENCH_LOG_LEVEL` | `off`; `brief` prints stages, candidate counts, filtering rounds, profiler attempts, and retry counts. `detailed` also prints settings, per-candidate scores/means, and quality diagnostics. |
| `TRITON_NPU_BENCH_REPORT_BEST_CONFIG` | `0`; print the original 1-based candidate number, all configuration parameters, `ubtune_cfg`, selected-attempt score/mean, counts, quality results, and profiler aggregates. |
| `TRITON_AUTOTUNE_REPORT_TIMING` | `0`; print one final NPU autotune time breakdown. |

These controls are independent of each other and of `TRITON_PRINT_AUTOTUNING`,
which retains its existing output. They are supported only for the built-in NPU
autotune path. Neither logging level prints individual launches or raw CSV rows.

Winner reports aggregate already-read measured rows. Numeric metrics are averaged;
IDs and absolute timestamps are not. Constant text appears once, changing text as
`varies`, and incomplete metrics show availability counts without substituting zeros.
The report describes only the retained attempt and prints its central-half score
and ordinary mean separately. Unavailable measurements are explicit. Reporting
adds no measurements and does not read the CSV again.

Timing lists total wall time, generation/pruning, compilation, slow filtering,
calibration, measurements including retries, and remaining overhead. Measurement
time includes profiler completion, CSV reading, and quality evaluation. Reused
filter measurements belong to filtering time; calibration does not count them
again. Timing ends before the optional extra winner profile, final kernel call,
and final report output. Cache hits repeat neither stages nor reports.

The optional `auto_prof_dir` profile retains its existing invocation and reads the
NPU measurement environment through `do_bench_npu`.
