# NPU autotune policies and diagnostics

This guide describes the production NPU path. In the comparison test branch,
winner/timing reports also support default and legacy routes. Measurement
controls apply only to current NPU, except for shared cache mode.
See [Three benchmark paths](npu_benchmark_comparison.md).

Set `TRITON_BENCH_METHOD=npu` before running Python to configure the built-in
NPU profiler used by `autotune` and `max_autotune`. All new controls use environment
variables; the decorator signatures are unchanged. Unset fields use the explicit
defaults below. The ordinary and custom benchmarkers ignore these controls.
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
| `TRITON_NPU_BENCH_CACHE_MODE` | `hot`; `cold` evicts L2; `hot` does not explicitly evict it between profiler launches. |
| `TRITON_NPU_BENCH_WARMUP` | `50`; warmup launches per candidate; integer >= 0. |
| `TRITON_NPU_BENCH_ACTIVE` | `250`; minimum measured launches per candidate; integer >= 1, or >= 2 with quality checks. |
| `TRITON_NPU_BENCH_LOWER_QUANTILE` | `0.25`; lower quantile boundary of the measured durations averaged for selection. |
| `TRITON_NPU_BENCH_UPPER_QUANTILE` | `0.75`; upper quantile boundary of the measured durations averaged for selection. |
| `TRITON_NPU_BENCH_QUALITY_CHECK` | `1`; evaluate the predefined sample-quality metrics. |
| `TRITON_NPU_BENCH_QUALITY_CHECK_ALL` | `0`; stop after the first failed check. `1` computes all quality metrics for diagnostics, including rejected attempts. Does not change retry decisions or the selection-cache key. |
| `TRITON_NPU_BENCH_MAX_RETRIES` | `5`; additional attempts for failing quality checks or unusable profiler data; at most six attempts in total. |
| `TRITON_NPU_BENCH_MEASURE_BUDGET_MS` | `10`; estimated accumulated device-time budget per candidate, in milliseconds. `0` disables it; other values must be finite and positive. |
| `TRITON_NPU_BENCH_CALIBRATION_RUNS` | `50`; measured launches per short event estimate, used by filtering and budget calibration; integer >= 2. |
| `TRITON_NPU_BENCH_FILTER_SLOW_CONFIGS` | `1`; enable preliminary filtering through NPU events under the selected cache policy. |
| `TRITON_NPU_BENCH_SLOW_CONFIG_ROUNDS` | `5`; total filtering rounds per winner, including the first; integer >= 1. |
| `TRITON_NPU_BENCH_SLOW_CONFIG_FACTOR` | `2`; recheck candidates above this multiple of the current winner's retained minimum; finite and >= 1. |
| `TRITON_NPU_BENCH_SLOW_CONFIG_RECHECK_DELAY_S` | `0.2`; finite, nonnegative pause before each needed subsequent round. `0` disables the pause. |

Boolean values accept `0`, `1`, `false`, and `true`. Settings do not enable one
another: quality checks do not automatically enable retries, filtering, or a budget.

The new NPU defaults override the call's warmup, active and cache arguments,
including standalone `do_bench_npu`, without changing public signatures.
Fixed active counts and the budget bypass CV count calculation and its
200-second estimated measurement cap. Hot mode cannot guarantee L2 residency.

## Slow filtering and calibration

The first filtering round benchmarks every candidate through NPU events, returning
the minimum event duration. Each estimate uses one preliminary call followed by
`CALIBRATION_RUNS` measured launches. All filtering rounds, budget calibration and
main profiler measurements use the same selected `CACHE_MODE`. Cold event estimates
evict L2 before recording each measured launch; hot estimates do not explicitly evict it.
Each candidate retains its lowest event duration across rounds. The threshold is
the current winner's retained minimum multiplied by `SLOW_CONFIG_FACTOR`.
Subsequent rounds benchmark only candidates above that threshold, excluding the
current winner. If another candidate obtains a strictly lower minimum, it becomes
the winner: all original candidates are reclassified and the round counter restarts
at 1. Ties preserve the current winner. Previously retained candidates can become
suspects under the new threshold. Candidates still slow after the configured number
of rounds are excluded. One round permits immediate filtering; the default five
rounds provide four rechecks per winner.

There is one pause before each needed recheck round. No pause or recheck occurs
without suspects, with fewer than two candidates, or with one configured round.
`SLOW_CONFIG_RUNS` and `SLOW_CONFIG_RECHECK_RUNS` are not used; `CALIBRATION_RUNS`
controls the short event estimates. The ordinary and custom benchmarkers are unchanged.
The release CV time-limit pruning also retains its original implementation; explicit
`ACTIVE` or `MEASURE_BUDGET_MS` bypasses that separate pruning and count calculation.

A measurement budget sets
one shared `active = max(active, ceil(measure_budget_ms / minimum_event_duration_ms))`
for all surviving candidates, using the fastest retained event launch.
Calibration reuses slow-filter minima.
When those results are unavailable, one preliminary call and `CALIBRATION_RUNS`
event measurements estimate each candidate under the requested profiler cache
policy. Calibration creates no profiler session or CSV. Event estimates determine
the shared count only; final ranking uses profiler kernel durations. Without a
budget, the original active count is retained. The budget is an estimate,
not a guarantee of accumulated kernel time or a wall-clock timeout.

## Profiler attempts and selection

All current candidates share one profiler session and one active count per attempt.
The original pandas collector reads the common CSV
once, filters target kernels and cache operations, and splits candidates by
equal `warmup + active` blocks. Each candidate's warmup rows are excluded. Identical target
kernel names are supported.

The selection score defaults to the mean of the central 50% of measured durations.
Set `LOWER_QUANTILE=0` and `UPPER_QUANTILE=1` to average all measured durations.
The finite boundaries must satisfy `0 <= LOWER_QUANTILE < UPPER_QUANTILE <= 1`.
Sorted sample indices are rounded down, retaining at least one measurement.
These settings select a range to average; `do_bench(quantiles=...)` instead returns
quantile values. Trimming affects selection only, participates
in the selection-cache key, and does not remove samples from quality checks.
Quality checks cover tail contamination, gaps, multimodality, drift, and change
points using the existing algorithms in `_benchmark_quality.py`. Their thresholds
are 0.030, 0.2, 0.027, 0.3 and 0.275, respectively.
Checks normally stop at the first failure; detailed logs then contain only the
metrics already evaluated. Set `TRITON_NPU_BENCH_QUALITY_CHECK_ALL=1` to evaluate
all five checks even after a failure. Both modes use all untrimmed samples and
the same statistics and thresholds. Pairwise statistics use linear auxiliary
memory; sampled slopes suggest search pivots, never replace the full-sample median.
Only failing candidates receive quality retries. Failed acquisition of the shared
CSV retries the current batch. The shared count is retained across retries without recalibration.
Kernel execution and synchronization failures propagate without retries.

Each candidate retains its minimum measured score regardless of quality. A faster
failed-quality attempt remains eligible after a slower successful attempt. If the
selected measurement failed quality checks, one plain-text message states the
autotuner's uncertainty independently of the logging level. It does not change the
winner or raise a warning exception. No ANSI coloring or global warning handler is used.

Existing unavailable-data behavior is retained: a simple missing-CSV result is
`inf`; the retry path raises when no usable measurements were acquired. The
new NPU defaults are active measurement policies: single surviving candidates
still use profiling, and row mismatches do not silently fall back to event timing.

Effective measurement policies participate in the selection-cache key. Logging
and reports do not. `CALIBRATION_RUNS` is inactive unless a budget or filtering is
enabled; other filtering settings are inactive when filtering is disabled. A recheck
delay is inactive with one filtering round. No additional warnings are emitted for inactive settings.

## Console diagnostics

| Environment variable | Default / output |
| --- | --- |
| `TRITON_NPU_BENCH_LOG_LEVEL` | `brief`; prints stages, candidate counts, filtering rounds, profiler attempts, and retry counts. `off` disables these logs; `detailed` adds settings, per-candidate scores/means, and quality diagnostics. |
| `TRITON_NPU_BENCH_REPORT_BEST_CONFIG` | `1`; print the original 1-based candidate number, all configuration parameters, `ubtune_cfg`, selected-attempt score/mean, counts, quality results, and profiler aggregates. `0` disables it. |
| `TRITON_AUTOTUNE_REPORT_TIMING` | `1`; print one final NPU autotune time breakdown; `0` disables it. |

These controls are independent of each other and of `TRITON_PRINT_AUTOTUNING`,
which retains its existing output. They are supported only for the built-in NPU
autotune path. Neither logging level prints individual launches or raw CSV rows.

Winner reports aggregate already-read measured rows. Numeric metrics are averaged;
IDs and absolute timestamps are not. Constant text appears once, changing text as
`varies`, and incomplete metrics show availability counts without substituting zeros.
The report describes only the retained attempt and prints its selection score
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
