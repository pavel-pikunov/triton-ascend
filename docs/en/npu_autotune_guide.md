# NPU autotune policies and diagnostics

This guide describes the production NPU path. In the comparison test branch,
winner/timing reports also support default and legacy routes. Measurement
controls apply only to current NPU, except for shared cache mode.
See [Three benchmark paths](npu_benchmark_comparison.md).

This test branch additionally defaults to reduced NPU profiler export.
`TRITON_NPU_BENCH_PROFILER_MODE=summary|full` switches between summary-only
export and the previous full analysis. This setting is not in the PR branch;
see the [export experiment](npu_benchmark_comparison.md#npu-profiler-export-experiment).

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
| `TRITON_NPU_BENCH_WARMUP` | Unset: `ceil(active / 10)` for the current batch. An explicit integer >= 0 overrides it; `0` disables profiler warmup. |
| `TRITON_NPU_BENCH_ACTIVE` | `30`; measured launches per series in the first batch and minimum for subsequent batches; integer >= 1, or >= 2 with quality checks. |
| `TRITON_NPU_BENCH_LOWER_QUANTILE` | `0.25`; lower quantile boundary of the measured durations averaged for selection. |
| `TRITON_NPU_BENCH_UPPER_QUANTILE` | `0.75`; upper quantile boundary of the measured durations averaged for selection. |
| `TRITON_NPU_BENCH_QUALITY_CHECK` | `1`; enable quality checks and minimum confirmation. |
| `TRITON_NPU_BENCH_QUALITY_CHECK_ALL` | `0`; stop after the first failed check. `1` computes all quality metrics for diagnostics, including rejected attempts. Does not change retry decisions or the selection-cache key. |
| `TRITON_NPU_BENCH_STABILITY_ONLY` | `0`; `1` skips all five sample-quality metrics and retains only minimum confirmation. Requires `STABILITY_RUNS >= 2` when quality is enabled; inactive with `QUALITY_CHECK=0`. |
| `TRITON_NPU_BENCH_MAX_RETRIES` | `5`; maximum profiler batches in total, including the first. `0` permits one initial batch. Failed acquisition also consumes a batch. |
| `TRITON_NPU_BENCH_BATCH_RUNS` | `5`; M series per candidate in one profiler batch; integer >= 1. Independent of N. With `QUALITY_CHECK=0`, each batch has one series. |
| `TRITON_NPU_BENCH_STABILITY_RUNS` | `7`; N results needed to confirm each candidate's minimum using its complete history; integer >= 1. `1` disables cross-series confirmation. |
| `TRITON_NPU_BENCH_STABILITY_TOLERANCE` | `0.005`; require `max(means) / min(means) <= 1 + tolerance`. Finite and >= 0; `0.005` permits a 0.5% spread between extremes. Inactive with N=1 or quality disabled. |
| `TRITON_NPU_BENCH_MEASURE_BUDGET_MS` | `15`; estimated device time per series in subsequent batches, based on the first successful profile. No extra batch is forced. `0` disables it; other values must be finite and positive. |
| `TRITON_NPU_BENCH_CALIBRATION_RUNS` | `10`; measured launches per short event estimate for slow filtering only; integer >= 2. |
| `TRITON_NPU_BENCH_FILTER_SLOW_CONFIGS` | `1`; enable preliminary filtering through NPU events under the selected cache policy. |
| `TRITON_NPU_BENCH_SLOW_CONFIG_ROUNDS` | `10`; total filtering rounds per winner, including the first; integer >= 1. |
| `TRITON_NPU_BENCH_SLOW_CONFIG_FACTOR` | `2.5`; recheck candidates above this multiple of the current winner's retained minimum; finite and >= 1. |
| `TRITON_NPU_BENCH_SLOW_CONFIG_RECHECK_DELAY_S` | `0`; finite, nonnegative pause before each needed subsequent round. `0` disables the pause. |

Boolean values accept `0`, `1`, `false`, and `true`. Settings do not enable one
another: quality checks do not automatically enable retries, filtering, or a budget.
The default M=5 and N=7 require at least two successful profiler batches;
the first uses 30 active launches per series, and later batches use the retry budget.
The tolerance bounds the full relative range of a confirming group's ordinary
means, not a confidence interval. The group is searched across candidate history;
other series can lie outside it. Selection still uses the configured quantile score.

The new NPU defaults override the call's warmup, active and cache arguments,
including standalone `do_bench_npu`, without changing public signatures.
Fixed active counts and the budget bypass CV count calculation and its
200-second estimated measurement cap. Hot mode cannot guarantee L2 residency.

## Slow filtering and retry budget

The first filtering round benchmarks every candidate through NPU events, returning
the minimum event duration. Each estimate uses one preliminary call followed by
`CALIBRATION_RUNS` measured launches. All filtering rounds and main profiler
measurements use the same selected `CACHE_MODE`. Cold event estimates
evict L2 before recording each measured launch; hot estimates do not explicitly evict it.
Each candidate retains its lowest event duration across rounds. The threshold is
the current winner's retained minimum multiplied by `SLOW_CONFIG_FACTOR`.
Subsequent rounds benchmark only candidates above that threshold, excluding the
current winner. If another candidate obtains a strictly lower minimum, it becomes
the winner: all original candidates are reclassified and the round counter restarts
at 1. Ties preserve the current winner. Previously retained candidates can become
suspects under the new threshold. Candidates still slow after the configured number
of rounds are excluded. One round permits immediate filtering; the default ten
rounds provide nine rechecks per winner.

There is one pause before each needed recheck round. No pause or recheck occurs
without suspects, with fewer than two candidates, or with one configured round.
`SLOW_CONFIG_RUNS` and `SLOW_CONFIG_RECHECK_RUNS` are not used; `CALIBRATION_RUNS`
controls the short event estimates. The ordinary and custom benchmarkers are unchanged.
Event pairs are reused between candidates, with storage bounded by the largest
requested series. Hot estimates synchronize after warmup and after the complete
series; cold estimates also retain the existing cache-preparation synchronization.
The release CV time-limit pruning also retains its original implementation; explicit
`ACTIVE` or `MEASURE_BUDGET_MS` bypasses that separate pruning and count calculation.

The first successful profiler batch always uses `ACTIVE`. If candidates need
another batch, its shared count becomes
`max(ACTIVE, ceil(MEASURE_BUDGET_MS / first_profile_min_mean_ms))`, where the
denominator is the smallest ordinary series mean from the first successful batch.
All subsequent batches reuse this count, including when only some candidates retry.
No separate calibration is run, and event minima from slow filtering do not set
the budget. If the first batch confirms every candidate, no second batch is added.
With the budget disabled, every batch uses `ACTIVE`.
Profiler warmup defaults to `(active + 9) // 10` for each batch; an explicit
`WARMUP` is preserved, including zero. Earlier series keep their original counts
and remain in the confirmation history. The budget estimates device time per
series in later batches, not total batch time or a wall-clock timeout.

## Profiler attempts and selection

All current candidates share one profiler session and one active count per batch.
With M=3, launch all candidates once, then repeat the same order twice:
`A1 B1 C1, A2 B2 C2, A3 B3 C3`. Each series has its own warmup and active launches.
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
are 0.04, 0.26, 0.04, 0.39 and 0.36, respectively.
The [per-metric detection audit](npu_quality_detection.md) separately tests present
defects and false alarms on 50-launch Gaussian controls; four metrics currently
fail the zero-false-alarm target on those controls.
Checks normally stop at the first failure; detailed logs then contain only the
metrics already evaluated. Set `TRITON_NPU_BENCH_QUALITY_CHECK_ALL=1` to evaluate
all five checks even after a failure. Both modes use all untrimmed samples and
the same statistics and thresholds. Pairwise statistics use linear auxiliary
memory; sampled slopes suggest search pivots, never replace the full-sample median.
Only failing candidates receive quality retries. Failed acquisition of the shared
CSV retries the current batch. Before the first successful profile, the count remains `ACTIVE`.
Kernel execution and synchronization failures propagate without retries.

With quality enabled, each unfinished candidate receives `BATCH_RUNS` series
per batch, subject to acquisition failures. `MAX_RETRIES=5` permits at most five
batches, not the first plus five repeats. Each candidate retains its minimum score
regardless of quality. After processing the entire batch, compare
the untrimmed mean of that attempt with N-1 other attempts that passed their own
sample-quality checks, searching the entire history of this candidate. Every
comparison group includes the retained minimum, and its maximum/minimum mean
ratio must not exceed `1 + STABILITY_TOLERANCE`. For example, 10, 11, 11.1 does
not confirm the minimum 10 at N=2; a subsequent quality-passing 10.1 can confirm
it. Means of different candidates are never compared by this check.

When the full group meets the tolerance, return the minimum attempt if its own
sample-quality checks passed. Otherwise return the lowest-score quality-passing
attempt from a matching group: their times are equivalent within the requested
tolerance. Failed attempts cannot corroborate another minimum. For example,
10 (failed), 9.9 (passed) requires another quality-passing measurement near 9.9;
the old failed 10 cannot confirm it. Older passing attempts can confirm a new
minimum immediately. Only unfinished candidates retry.
History spans all batches of the same candidate: a second batch at M=3 adds
three means to the first three, even if its active count increased. N independently
sets the size of the confirming group. A new minimum must be confirmed against that
complete history. Confirmation earlier inside a batch cannot hide a later,
lower unconfirmed result. Compare candidates for the final winner only after
all candidates finish or the batch limit is reached.
At exhaustion, an unfinished candidate returns its absolute minimum with
`stability not confirmed`, even if later attempts passed sample quality. Thus
neither a later slow stable group nor an unrelated clean attempt ends confirmation.

N=1 disables confirmation; a candidate completes when any series in its batch
passes sample quality, retaining its minimum regardless of quality. Set M=1 too
for the previous one-series-per-batch behavior. No extra compilation, filtering
or calibration is performed for retries. Keep the minimum report and
sample-free quality-passing reports, bounded by M * max(1, MAX_RETRIES) series. The
compatibility helper retains aggregated CSV results for passing history and raw
rows only for the minimum, rather than retaining every attempt's sample arrays.
`quality_failures` describes the returned attempt; `all_configs_passed` separately
reports whether any candidates remain pending. At exhaustion with unfinished
candidates, a yellow message reports
the selected configuration number, lowest retained score and selected-attempt
pass/failed status independently of the logging
level. ANSI coloring is unconditional, including redirected output. The message
does not change the winner or raise a warning exception.

Existing unavailable-data behavior is retained: a simple missing-CSV result is
`inf`; the retry path raises when no usable measurements were acquired. The
new NPU defaults are active measurement policies: single surviving candidates
still use profiling, and row mismatches do not silently fall back to event timing.

Effective measurement policies participate in the selection-cache key. Logging
and reports do not. `CALIBRATION_RUNS` and other filtering settings are inactive
when filtering is disabled. A recheck
delay is inactive with one filtering round. No additional warnings are emitted for inactive settings.

## Minimum confirmation without sample-quality metrics

The comparison branch supports a minimal experiment with fixed launch counts,
no event calibration or slow filtering, and only confirmation of each candidate's
minimum. For three agreeing attempts, a 3% tolerance and 30 measured launches:

```bash
export TRITON_BENCH_METHOD=npu
export TRITON_NPU_BENCH_ACTIVE=30
export TRITON_NPU_BENCH_WARMUP=3
export TRITON_NPU_BENCH_MEASURE_BUDGET_MS=0
export TRITON_NPU_BENCH_FILTER_SLOW_CONFIGS=0
export TRITON_NPU_BENCH_QUALITY_CHECK=1
export TRITON_NPU_BENCH_STABILITY_ONLY=1
export TRITON_NPU_BENCH_BATCH_RUNS=3
export TRITON_NPU_BENCH_STABILITY_RUNS=3
export TRITON_NPU_BENCH_STABILITY_TOLERANCE=0.03
export TRITON_NPU_BENCH_MAX_RETRIES=5
```

`QUALITY_CHECK=1` keeps confirmation enabled. In stability-only mode, the five
sample metrics are not calculated and all usable attempts can corroborate the
minimum. Invalid profiler data and execution errors retain their existing handling.
Only unfinished candidates retry. With these settings, each profiler batch
contains three series per unfinished candidate, and at most five batches are
allowed: fifteen series per candidate. Unconfirmed minima retain the existing
failure status at exhaustion.
The compatibility utils inherit this mode through the shared benchmark path.
The score's quantile boundaries and the selected cache mode remain unchanged.
Set `STABILITY_ONLY=0` to restore the five statistical checks.

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

In this test branch, the final measurement line additionally contains exclusive
times for profiled launches/synchronization, CANN export/analysis, CSV/aggregates,
quality and remaining measurement overhead (including setup/finalization).

The optional `auto_prof_dir` profile retains its existing invocation and reads the
NPU measurement environment through `do_bench_npu`.
