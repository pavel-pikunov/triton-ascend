# Three benchmark paths (test branch only)

Branch: `AutotunerBenchCompare-3.2.2`, based on `0f6ade525`.
The legacy implementation is pinned to release commit
`7aa09ec5f1ba564b5544007bca4f97ce0e493182`, rather than a moving release ref.

| `TRITON_BENCH_METHOD` | Tuner and measurement path | Selection score |
| --- | --- | --- |
| Unset or `default` | Current tuner, original device-event `triton.testing.do_bench` | Median (autotuning requests quantiles 50%, 20%, 80%) |
| `npu_legacy` | Copied release tuner and copied release NPU profiler | Arithmetic mean of all measured durations |
| `npu` | Current tuner and current NPU profiler | Mean of the central 50% of measured durations |

Both `autotune` and `max_autotune` route to the complete copied tuner for
`npu_legacy`, before processing the new options. Direct calls to the backend's
public `do_bench_npu` also route to the copied profiler in this mode.
Set the mode before imports and decorator creation, and keep it fixed for the
process lifetime. Each comparison run must start a separate Python process;
switching the environment on an already-created tuner is unsupported.

## What is preserved in legacy mode

`runtime/_legacy_autotuner.py` copies release `runtime/autotuner.py`, with the
mode name, its two imports of the copied testing module, and optional final
diagnostics changed. Trailing whitespace on one original line was removed.
`_legacy_testing.py` copies release `testing.py` with only the two environment
overrides described below. Shared parsers, generators, UBTuner and compiler
are unchanged from the pinned release.

The original common profiler session, CV pruning/count calculation, warmup,
kernel-name filtering, CSV aggregation, synchronization, exceptions and cleanup
are preserved. L2 eviction in legacy mode still occurs before kernel hooks.
The new quality checks, retries, budget, slow filter and stage logs are not
applied to legacy. The `npu_bench_options` decorator argument is ignored there;
`report_best_config` and `report_timing` are supported by all three paths.
original `auto_prof_dir` remains supported and can perform an extra profile.

Original fallbacks are also preserved: one remaining configuration uses the
event benchmarker, and mismatched profiler rows issue the original
`RuntimeWarning` and fall back to events. A custom `do_bench` is respected in
all modes. These cases are **not legacy-profiler measurements**. A single
configuration before tuning can also skip measurement altogether. For a pure
profiler comparison, use multiple surviving candidates and the built-in
benchmarker, and inspect warnings.

## Two shared controls

| Environment variable | Default events and legacy NPU | Current NPU |
| --- | --- | --- |
| `TRITON_NPU_BENCH_CACHE_MODE=hot\|cold` | Override intentional L2 eviction; unset preserves original behavior | Existing cache policy, unchanged |
| `TRITON_NPU_BENCH_ACTIVE=N` | Positive integer; exact measured launch count per candidate | Existing active policy; exact count without a budget, minimum count with a budget |

For events these overrides apply only to the Ascend target (`backend="npu"`).
Other backends and custom benchmarkers are not overridden. The event path
retains its original initial call, five-call runtime estimate and warmup-count
calculation. Hot mode disables eviction in both estimation and measurement.
Legacy retains its original preliminary call, profiled warmup and CV pruning;
the active override only replaces the final measured launch count.
Warmup and estimation launches are additional to `ACTIVE`.

Hot means no intentional eviction; hooks, working-set size and other operations
can still change cache contents. Cold eviction also uses the original operations
and hook order of each path, so their cache preparation is not identical.
No other `TRITON_NPU_BENCH_*` measurement options are resolved by events or legacy.
For events, the `npu_bench_options` mapping remains ignored; the two shared
controls are environment-only.

## Run an existing case in separate processes

Omit custom `do_bench`, `auto_prof_dir`, explicit benchmark policies and enabled
diagnostics from the test's decorators. Keep input data, candidate order, device,
compiler options and device load the same for each path. For the baseline,
clear inherited NPU policies so only cache mode and active are set:

```bash
for name in "${!TRITON_NPU_BENCH_@}"; do unset "$name"; done
export TRITON_PRINT_AUTOTUNING=0
export TRITON_DEBUG=0
export TRITON_AUTOTUNE_REPORT_TIMING=0
unset TRITON_AUTOTUNE_RUNS TRITON_AUTOTUNE_CSV_DIR
export PYTHONWARNINGS=default
export TRITON_NPU_BENCH_CACHE_MODE=hot  # Repeat the comparison with cold.
export TRITON_NPU_BENCH_ACTIVE=1000

unset TRITON_BENCH_METHOD
python your_existing_case.py
TRITON_BENCH_METHOD=npu_legacy python your_existing_case.py
TRITON_BENCH_METHOD=npu python your_existing_case.py
```

Here quality checks, retries, measurement budget, slow filter and new logs/reports
are all off by default. Do not set a budget when comparing equal measured counts.
The current NPU path bypasses CV time-limit pruning with explicit `ACTIVE`;
legacy and default preserve CV pruning. Use a case where the candidate set stays
the same, or record the actual surviving configurations when interpreting results.

Repeat the same case across fresh processes and rotate mode order to observe
run-to-run stability. Save the chosen configuration and evaluate each winner
with the same independent measurement method, cache condition and launch count.
The three paths use different estimators; their internal scores alone are not
a common measurement of the selected kernels. Measure total autotuning time
externally for all three, with compilation-cache conditions kept consistent.
There is no additional benchmark runner in this branch.

## Record selection decisions for all three paths

The same independent flags work for `autotune` and `max_autotune`, including
legacy mode. Explicit decorator arguments override the environment:

```bash
export TRITON_NPU_BENCH_REPORT_BEST_CONFIG=1
export TRITON_AUTOTUNE_REPORT_TIMING=1
unset TRITON_NPU_BENCH_LOG_LEVEL  # Stage logs stay off.

TRITON_BENCH_METHOD=default python your_existing_case.py > default.log
TRITON_BENCH_METHOD=npu_legacy python your_existing_case.py > legacy.log
TRITON_BENCH_METHOD=npu python your_existing_case.py > npu.log
```

The winner report contains all config parameters and `ubtune_cfg`, its original
1-based position before pruning, its selection score, and its stable SHA-256
ID. The reported time is the already-measured median for events, arithmetic
mean for legacy, or central-half score and ordinary mean of the selected attempt
for current NPU. Custom benchmarker scores keep their original meaning. An
unmeasured selection is explicitly marked; no additional kernel launches or
CSV reads are performed for reporting. Legacy fallback is marked as the
actual `default` benchmark path rather than `npu_legacy`.

Each winner block also contains one machine-readable line beginning with
`AUTOTUNE_DECISION `, followed by JSON. Extract those lines to collect decisions:

```bash
grep '^AUTOTUNE_DECISION ' default.log legacy.log npu.log
```

Compare decisions for the same `kernel` and input `key`, grouping winners by
`config_id`, not by their list positions. `config_id` describes full config
parameters, including `ubtune_cfg`; `candidate_id` identifies the original
candidate before UBTuner changes. IDs exclude `pre_hook`, which is still printed
in the full parameters. Dictionary insertion order, object addresses and
`PYTHONHASHSEED` do not affect IDs for supported values. Unsupported values
with process-dependent representations make the ID unavailable with a warning.

`candidate_set_id` identifies the original candidate list independently of
order, preserving duplicates; `candidate_order_id` also includes order.
Equal set IDs with different order IDs indicate reordered candidates.
Different set IDs indicate changed candidates or parameters. The analogous
`scored_set_id` and `scored_order_id` describe the returned score dictionary
after pruning, including non-finite scores. They are absent when benchmarking
was skipped. These diagnostics observe order; they do not sort or shuffle the
actual tuning candidates. Compare the same input key and generation settings
when checking order between methods or processes.

Timing is printed once after selection and before the final kernel execution,
optional extra winner profile and garbage collection. Legacy reports the same
stages as the current tuner; its CV time-limit pruning is included in the
`Slow filter` stage and calibration remains zero. Report formatting is outside
the measured total. Cache hits repeat neither report nor decision line.

For a separate check of the current path's policies, keep `TRITON_BENCH_METHOD=npu`
and enable only the desired options, for example:

```bash
export TRITON_BENCH_METHOD=npu
export TRITON_NPU_BENCH_QUALITY_CHECK=1
export TRITON_NPU_BENCH_MAX_RETRIES=4
export TRITON_NPU_BENCH_LOG_LEVEL=brief
export TRITON_NPU_BENCH_REPORT_BEST_CONFIG=1
export TRITON_AUTOTUNE_REPORT_TIMING=1
python your_existing_case.py
```

This is a policy check with potentially multiple attempts, separate from the
single-attempt comparison above. The minimum score across measured attempts
remains the selection criterion regardless of quality; the winner report shows
that score and its attempt's ordinary mean separately.

## Repeat autotuning and save per-config CSV statistics

These test-branch controls apply to all three built-in benchmark paths and both
decorators. Set them before constructing the decorators:

```bash
export TRITON_AUTOTUNE_RUNS=100
export TRITON_AUTOTUNE_CSV_DIR=/tmp/autotune-comparison
export TRITON_NPU_BENCH_REPORT_BEST_CONFIG=1  # Optional: each decision and its ID.
export TRITON_AUTOTUNE_REPORT_TIMING=1       # Optional: each tuning pass.

TRITON_BENCH_METHOD=default python your_existing_case.py
TRITON_BENCH_METHOD=npu_legacy python your_existing_case.py
TRITON_BENCH_METHOD=npu python your_existing_case.py
```

`RUNS` is the total number of independent autotune decisions per uncached input
key, including the first decision (default: 1). Every pass repeats config
generation, pruning, calibration where applicable, measurements, retries and
winner selection. Between passes only that key's selected-config cache entry
is removed. JIT/compiler caches remain intact: successful configurations are
reused from those caches. The original compilation preparation calls still
run, including their hooks; compilation failures can be attempted again.
There is no process restart or extra compilation-cache reset.

Kernel calls needed by each benchmark retain their original behavior. The
final application kernel and optional extra winner profile run once, using
the last pass's winner. Victory counts and CSV averages never influence that
choice. Later cache hits do not repeat autotuning, reports or CSV output.

The directory flag enables CSV output even with one run. Without it, `RUNS`
still repeats decisions and prints the completed count. Both flags are
independent of winner reports, timing reports and stage logs. Unset both to
disable comparison mode. Custom `do_bench` is rejected in comparison mode
because its score need not be a duration; its ordinary winner reports remain
supported.

One new CSV is created for each kernel/input-key/method comparison. Its filename
contains the kernel name, requested method, logical input-key ID and a unique
run suffix; existing files are not overwritten. The saved absolute path is
printed after completion. Separate method invocations therefore produce
separate files in the same directory.

| Column | Meaning |
| --- | --- |
| `config_index` | Original 1-based position before pruning. |
| `config` | Stable JSON of config parameters, including `ubtune_cfg` when present, excluding `pre_hook`. |
| `mean_time_us` | Mean of this config's finite selection scores across autotune passes, in microseconds. |
| `best_time_us` | Minimum of those scores, in microseconds. |
| `max_time_us` | Maximum of those scores, in microseconds. |
| `best_count` | Number of passes that selected this config as winner. |
| `total_runs` | Number of completed autotune passes for this input key. |

The per-pass score is the event median for default, the original mean for
legacy, and the central-half score of the fastest measured attempt for new
NPU, irrespective of its quality status. Original event fallback scores are
used when fallback occurs. This CSV aggregates selection scores, not raw
kernel launches and not all retry attempts. The mean column does not replace
the estimator used to choose each winner.

All original candidates get a row, including pruned or failed configurations.
Missing/non-finite scores are excluded from time aggregates, without zero
substitution. All three time cells are empty when no finite score is available;
`total_runs` still counts decisions, not the number of available measurements
for that row. A single surviving config can be selected without measurements.
The sum of `best_count` equals `total_runs`.

Within a comparison, changes to the original candidate parameters/order,
input key or already-observed effective parameters stop collection with an
explicit error rather than combining incompatible configurations. Each
candidate must be a distinct `Config` object. When winner reporting is enabled,
`AUTOTUNE_DECISION` also includes `run_index` and `total_runs`; stable set/order
IDs still allow comparison across fresh processes. No completed CSV is emitted
if tuning fails before all requested passes finish.
