# Repeated autotune comparison

This branch compares two selectable paths:

| TRITON_BENCH_METHOD | Path |
|---|---|
| npu | Current StableAutotuner profiler batches and minimum confirmation. |
| default or unset | Original event benchmark and original tuner. |

`npu_legacy` has been removed from selection. The private copied tuner remains because the
`default` route uses it; its obsolete legacy-profiler branch has been removed.

The comparison branch additionally supports TRITON_NPU_BENCH_PROFILER_MODE=summary
to skip TensorBoard analysis and read native CANN summary CSVs. The default is full.
This changes export, not the minimum-confirmation algorithm. PIPE_UTILIZATION independently
controls whether utilization counters are collected.

## Repeating decisions

```bash
export TRITON_AUTOTUNE_RUNS=100
export TRITON_AUTOTUNE_CSV_DIR=./autotune_statistics
export TRITON_NPU_BENCH_REPORT_BEST_CONFIG=1
export TRITON_AUTOTUNE_REPORT_TIMING=1
export TRITON_NPU_BENCH_LOG_LEVEL=brief
```

The default run count is 1. Setting CSV_DIR alone records a single decision.
For each new input key, the tuner repeats candidate selection the requested number of times,
clearing only that key's selected config between runs. It preserves JIT/compiler caches,
other input keys and candidate order. Intermediate tuning runs omit the final winner launch;
the final tuning run launches its winner normally. A cache hit returns the cached choice
without repeating the experiment or creating another statistics CSV.

The CSV retains the existing columns: config_index, config, mean_time_us, best_time_us,
max_time_us, best_count, total_runs and the existing timing summaries. There is one row per
original candidate. best_count counts selections; measured score summaries include only
finite measurements. A candidate removed by CV pruning may have empty score summaries.
The first compilation timing and mean repeat-autotune timing remain separate.
These files contain per-decision statistics, not raw profiler samples.

## Diagnostics

- TRITON_NPU_BENCH_REPORT_BEST_CONFIG=1 prints the original candidate number, parameters,
  stable config/candidate IDs and an AUTOTUNE_DECISION JSON record. NPU scores identify
  their quantile estimator; default reports identify median and 20%/80% quantiles.
- TRITON_AUTOTUNE_REPORT_TIMING=1 prints the existing stage totals. The CV rough pass is
  charged to slow_filter; there is no separate NPU fast filter or calibration stage.
  Measurements includes every profiler batch and its CSV processing.
- TRITON_NPU_BENCH_LOG_LEVEL=brief or detailed prints generation, compilation and actual
  profiler batch counts, including the active/warmup used. off is the default.

These controls observe the current benchmark; they do not restore the retired quality engine
or add measurements. NPU confirmation failures are reported by the benchmark itself.
The updated utils export confirmation status and compact aggregates to the result CSV.

## Utils

Install the comparison-specific bundle before the first autotune call:

- utils.py replaces benchmark/utils.py, preserving the result CSV schema.
- utils_compact.py replaces mojo_opset/tests/utils.py. utils_npu_compat.py is the same file
  under its older name; install one of these aliases.
- source npu_bench.env enables the configured NPU path.
- source repeat_autotune.env additionally enables 100 selections and their statistics CSV.

The helper captures both the current NPU tuner and the original event tuner. It reuses the
last repetition's retained result for the result CSV; the statistics CSV contains all repetitions.
A cache hit retains SelectedConfig and the saved report. A fixed kernel without an autotuner
uses the same public do_bench_npu, including batching, confirmation and vector budgeting.
Preparatory aclnn kernels are excluded from standalone output. No raw CSV is created.

For hot comparison, retain TRITON_NPU_BENCH_CACHE_MODE=hot and change only:

```bash
TRITON_BENCH_METHOD=npu python your_case.py
TRITON_BENCH_METHOD=default python your_case.py
```

Use separate Python processes after changing the method or measurement env. The helper's
capture hooks act on the process; concurrent tests should run in separate processes.
