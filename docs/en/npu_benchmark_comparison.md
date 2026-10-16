# Three benchmark paths (test branch only)

Branch: `AutotunerBenchCompare-3.2.2`. It additionally tests a reduced NPU
profiler export; this experiment is not in `AutotunerNpuPolicies-3.2.2-dev`.
The comparison branch keeps its existing release base. The original default and legacy benchmark paths
are pinned to `release/3.2.2-dev` commit
`3595ed00385d8f533091b9bd50caf7bd7a7ccd24`. NPU measurement-policy environment
variables apply only to the current `npu` path, except for the shared cache-mode
override described below.

| `TRITON_BENCH_METHOD` | Tuner and measurement path | Selection score |
| --- | --- | --- |
| Unset or `default` | Copied dev tuner, original device-event `triton.testing.do_bench` | Median (autotuning requests quantiles 50%, 20%, 80%) |
| `npu_legacy` | Copied dev tuner and copied dev NPU profiler | Arithmetic mean of all measured durations |
| `npu` | Current tuner and current NPU profiler | Central-half mean by default; mean within configured quantile boundaries |

Both `autotune` and `max_autotune` route to the complete copied tuner for
`default` and `npu_legacy`, before processing the new options. Direct calls to the backend's
public `do_bench_npu` also route to the copied profiler in this mode.
Set the mode before imports and decorator creation, and keep it fixed for the
process lifetime. Each comparison run must start a separate Python process;
switching the environment on an already-created tuner is unsupported.

## NPU profiler export experiment

Only the current `npu` path reads `TRITON_NPU_BENCH_PROFILER_MODE`:

- `summary` (default): Level1 with `AiCoreNone`, native CANN summary export and
  direct pandas reading of `op_summary` CSVs. The TensorBoard analysis pipeline
  and timeline export are skipped. Type information is retained for cold-cache
  filtering. `msprof` must be on PATH with its installed `msprof.py` script.
- `full`: Level1 with `PipeUtilization`, the usual TensorBoard analysis and
  `kernel_details.csv`. Use this to compare against the previous NPU export.

```bash
export TRITON_BENCH_METHOD=npu
export TRITON_NPU_BENCH_PROFILER_MODE=summary
python your_existing_case.py
TRITON_NPU_BENCH_PROFILER_MODE=full python your_existing_case.py
```

Both modes use the same candidate order, launch counts, cache policy, scores and
five quality checks. The mode participates in the selection-cache key. Each
attempt remains one profiler session, completed as one recording step for the
entire batch. No step is added per kernel launch. Failures follow the existing
acquisition-retry policy; a failed summary export does not silently switch modes.

The final timing block splits `Measurements` into launch/synchronization time,
export/analysis, CSV/aggregates, quality and remaining overhead. The latter
includes profiler setup/finalization and cleanup. Nested times are not counted
again in the autotune total. Native NPU export speed has not been measured locally;
the CANN 9.2.beta2 command interface was checked in the WSL build installation.
Raw-data parsing is still required, so this is not a promise of a fixed speedup.

## What is preserved in legacy mode

`runtime/_legacy_autotuner.py` copies dev `runtime/autotuner.py`, with the
mode name, its two imports of the copied testing module, and optional final
diagnostics changed. Trailing whitespace on one original line was removed.
`_legacy_testing.py` copies dev `testing.py`, adding only the cache-mode override.
The standard `python/triton/testing.py` also adds only that override to dev.
Shared parsers, generators, UBTuner and compiler are unchanged by this restoration.

The original common profiler session, CV pruning/count calculation, warmup,
kernel-name filtering, CSV aggregation, synchronization, exceptions and cleanup
are preserved. L2 eviction in legacy mode still occurs before kernel hooks.
The new quality checks, retries, budget, slow filter, quantile boundaries and
stage logs are not applied to legacy. Public decorators retain their original
signatures: all new measurement and reporting controls are environment-only.
Winner and timing reports remain supported by all three paths in this test branch.
The original `auto_prof_dir` remains supported and can perform an extra profile.

Original fallbacks are also preserved: one remaining configuration uses the
event benchmarker, and mismatched profiler rows issue the original
`RuntimeWarning` and fall back to events. A custom `do_bench` is respected in
all modes. These cases are **not legacy-profiler measurements**. A single
configuration before tuning can also skip measurement altogether. For a pure
profiler comparison, use multiple surviving candidates and the built-in
benchmarker, and inspect warnings.

## Original measurement settings

`TRITON_NPU_BENCH_CACHE_MODE=hot|cold` is shared by all three paths on Ascend.
Unset preserves each path's original cache preparation. Hot skips intentional
L2 eviction in estimation and measurement; cold uses the original eviction
operations and hook order of each path. Hot does not guarantee cache residency.
Other backends and custom benchmarkers are not overridden.

The original paths ignore all other new measurement-policy environment variables,
including `TRITON_NPU_BENCH_ACTIVE`. Launch counts remain native to each path.

Default events retain the original preliminary call, five-call estimate,
time-based warmup/repeat calculation and L2 clearing. The ordinary defaults
are `warmup=25` ms and `rep=100` ms; these are time targets, not launch counts.
Legacy profiling defaults to cold cache and normally `warmup=5`, `active=30`; cube/mix
CV cases retain the original count calculation and time-limit pruning.
The shared cache mode is the only added measurement override for these paths.

The original arguments and custom benchmarker API remain available. Winner/timing
reports, repeated autotune decisions and CSV recording observe these paths and
add no measurements within an individual selection pass.

## Run an existing case in separate processes

Omit custom `do_bench`, `auto_prof_dir`, explicit benchmark policies and enabled
diagnostics from the test's decorators. Keep input data, candidate order, device,
compiler options and device load the same for each path. For the baseline,
clear inherited NPU policies:

```bash
for name in "${!TRITON_NPU_BENCH_@}"; do unset "$name"; done
export TRITON_PRINT_AUTOTUNING=0
export TRITON_DEBUG=0
export TRITON_AUTOTUNE_REPORT_TIMING=0
unset TRITON_AUTOTUNE_RUNS TRITON_AUTOTUNE_CSV_DIR
export PYTHONWARNINGS=default

unset TRITON_BENCH_METHOD
python your_existing_case.py
TRITON_BENCH_METHOD=npu_legacy python your_existing_case.py
TRITON_BENCH_METHOD=npu python your_existing_case.py
```

The original paths retain their native measurement counts and disabled report
defaults. Current NPU uses the enabled policies and defaults listed in
[NPU autotune settings](npu_autotune_guide.md), including active>=250 and warmup=50.
To compare all three with hot cache, set `TRITON_NPU_BENCH_CACHE_MODE=hot` before
the three commands. Other new NPU settings do not alter the original two paths.
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

The same independent environment flags work for `autotune` and `max_autotune`,
including legacy mode:

```bash
export TRITON_NPU_BENCH_REPORT_BEST_CONFIG=1
export TRITON_AUTOTUNE_REPORT_TIMING=1
export TRITON_NPU_BENCH_LOG_LEVEL=off  # Override current NPU's brief default.

TRITON_BENCH_METHOD=default python your_existing_case.py > default.log
TRITON_BENCH_METHOD=npu_legacy python your_existing_case.py > legacy.log
TRITON_BENCH_METHOD=npu python your_existing_case.py > npu.log
```

The winner report contains all config parameters and `ubtune_cfg`, its original
1-based position before pruning, its selection score, and its stable SHA-256
ID. The reported time is the already-measured median for events, arithmetic
mean for legacy, or selection score and ordinary mean of the selected attempt
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
with process-dependent representations make the optional ID unavailable.

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
export TRITON_NPU_BENCH_LOWER_QUANTILE=0.25
export TRITON_NPU_BENCH_UPPER_QUANTILE=0.75
export TRITON_NPU_BENCH_MEASURE_BUDGET_MS=5
export TRITON_NPU_BENCH_FILTER_SLOW_CONFIGS=1
export TRITON_NPU_BENCH_SLOW_CONFIG_ROUNDS=2
export TRITON_NPU_BENCH_QUALITY_CHECK=1
export TRITON_NPU_BENCH_MAX_RETRIES=4
export TRITON_NPU_BENCH_LOG_LEVEL=brief
export TRITON_NPU_BENCH_REPORT_BEST_CONFIG=1
export TRITON_AUTOTUNE_REPORT_TIMING=1
python your_existing_case.py
```

This is a policy check with potentially multiple attempts, separate from the
original-path comparison above. The minimum score across measured attempts
remains the selection criterion regardless of quality; the winner report shows
that score and its attempt's ordinary mean separately.
Filtering and budget calibration share minimum event estimates. A new filtering
winner restarts suspect rounds against its lower threshold. A budget sets one
common active count from the fastest retained event launch; quality retries keep
that count even when only a subset of candidates is measured.

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
still repeats decisions and prints the completed count and mean autotune time. Both flags are
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
| `mean_autotune_s` | Mean total autotune duration across all passes, including the first. |
| `mean_generation_pruning_s` | Mean config generation/pruning duration per pass. |
| `compilation_s` | Full compilation/preparation stage duration from the first pass, including cache lookup and hooks. |
| `mean_slow_filter_s` | Mean slow-filter/CV time-limit pruning duration per pass. |
| `mean_calibration_s` | Mean calibration duration per pass. |
| `mean_measurements_s` | Mean main measurement duration per pass, including profiler completion, profiler CSV and quality/retries. |
| `mean_other_s` | Mean remaining autotune overhead per pass, excluding the named stages. |
| `first_autotune_s` | Total duration of the first autotune pass. |
| `mean_repeat_autotune_s` | Mean total duration excluding the first pass; empty with one pass. |

The added `_s` columns contain wall-clock seconds for the whole input-key
comparison, repeated identically on every config row. They are not per-config
costs and must not be summed across rows. Timing collection is automatic in
comparison mode even when `TRITON_AUTOTUNE_REPORT_TIMING=0`; that flag controls
only the individual console timing blocks. All three paths use the same
existing stage timers and total-time boundary.

The total runs from autotuner entry through selection and statistics collection,
before optional timing/winner report formatting, final application kernel, optional extra
profile, garbage collection and writing the aggregate CSV. Compilation is not
forcibly disabled: the first pass may compile, and later passes retain JIT
caches. The compilation stage still performs its original preparation/cache
lookup calls and hooks, so its duration can remain nonzero on cache hits.
`compilation_s` always reports the first pass's stage duration, without averaging
or dividing it by the number of passes. The console timing blocks continue to
show each pass's actually measured compilation/preparation duration. The other
stage means include all passes, so adding `compilation_s` to them does not
reconstruct `mean_autotune_s`.
`mean_autotune_s` averages all passes; `mean_repeat_autotune_s` separates the
repeated selection cost from the first pass's cost. Neither is a fresh-process
compilation measurement. A stage that is not executed contributes zero.

The per-pass score is the event median for default, the original mean for
legacy, and the mean within configured quantile boundaries of the fastest measured attempt for new
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

## Real-device smoke

The smoke file covers both decorators and all three benchmark routes. NPU cache
and count controls are deliberately set for every case: all paths use hot cache,
but only current NPU uses the count override; legacy retains five warmup rows
and 30 measured rows.
Each case makes three decisions, verifies kernel output,
stable candidate order, CSV win counts and silent cached calls:

```bash
pytest third_party/ascend/unittest/autotune_ut/test_npu_autotune_smoke.py -q
```

CPU helper checks do not validate native event/profiler execution. Historical
policy/report unit tests in this branch still include assertions about removed
decorator arguments, warning policies and private APIs; those assertions need
migration before treating the full old suite as the current contract.

## Benchmark kernels with or without autotune

```python
from triton.backends.ascend import benchmark_npu

benchmark_npu(my_kernel, grid, x, y, n, BLOCK_SIZE=256)
```

For a fixed JIT kernel, this test-only wrapper measures initial compilation/launch,
then runs the current NPU pipeline with slow filtering disabled. It reuses the same
event calibration, shared active/warmup calculation, cache policy, quantile score,
quality checks, retries, profiler mode and timing/winner reports. The profiler target
name comes from the compiled kernel. Reports include actual compiled constants and
compiler metadata. Existing environment flags control output; this wrapper always
uses the current NPU benchmark for fixed kernels, regardless of BENCH_METHOD.

If the decorator chain already contains an autotuner, the wrapper only launches it;
that autotuner owns measurement, reporting and cache-hit behavior. No second benchmark
or duplicate report is added. Autotuner single-config behavior is unchanged.

The wrapper returns the original launch result, including LibEntry tuples. Fixed
kernels are benchmarked on every wrapper call; kernel arguments must tolerate repeated
execution. The repeated-autotune CSV experiment does not apply to fixed kernels.


## Plug into operation-level performance tests

Replace only the NPU performance function chosen by `auto_switch_platform`:

```python
from triton.backends.ascend import perf_npu

# Inside the existing platform selector:
if platform == "npu":
    perf_fn = perf_npu
```

Existing tests keep their interface:

```python
perf(lambda: apply_penalties(logits, token_freqs, pres_pens, freq_pens,
                             rep_pens, temps, blocksize))
```

`perf_npu(fn)` is the callable form of `benchmark_npu`, not a second measurement
implementation. It runs `fn()` once and observes completed tuning decisions in the
current Python context. If an autotuner measured candidates, it returns that call's
result without another benchmark or report. This works through Python wrappers
and with NPU, legacy and default test-branch tuners, independently of report flags.
It reports only the kernels measured by those tuners, not total operation latency.

If no tuner measured candidates (including cache hits and single-config bypasses),
it measures the callable with the current NPU policies and slow filtering disabled.
All repeated calls reuse compiled kernels and tuner caches. The return value is the
original operation result, not the benchmark score. Arguments must tolerate repeated
execution; the adapter does not clone or restore mutable operation inputs.

The final operation report lists each kernel position and name with its own
quantile score, ordinary mean, minimum and maximum duration in microseconds.
Repeated names remain separate positions. Warmup rows are excluded, and every
entry comes from the same retained profiler attempt; no extra measurements or
CSV reads are added. Aggregate operation latency is not printed as a kernel score.

Internally, each sample remains the **sum of device kernel durations for one
call** for quality checks and retaining the fastest attempt, excluding host work
and gaps between kernels. The printed quality status applies to those operation
samples, not separately to each kernel.
The callable must launch a fixed ordered sequence on one stream; row counts and
kernel-name patterns are checked before aggregation. It is not a wall-clock or
critical-path benchmark for concurrent streams. Only the per-call duration and
first kernel timestamp are aggregated, so quality runs on operation samples rather
than mixing individual kernels. Unknown compiled parameters are not inferred from
the callable. The timing report's compilation stage includes the initial operation
call, which can also perform allocations and host work.

This adapter targets the Triton Ascend **test branch**; no Mojo source or A5-specific
kernel code needs to be changed. Native A5 execution still requires device validation.


## Export retained measurements to an external CSV writer

```python
from triton.backends.ascend import capture_npu_reports, perf_npu

with capture_npu_reports(lambda name, measurement: export_csv(name, measurement)):
    perf_npu(lambda: operation(...))
```

This test-only context sends retained NPU measurements to the callback independently
of console-report flags. `measurement.kernel_rows` contains `(kernel_name, rows)`
pairs in launch order, with warmup removed. Rows retain available original profiler
columns; they are captured during the existing CSV read, only while this context is
active. Tuners export the selected candidate's retained attempt, not other candidates.
Plain operations export each fixed kernel position from their retained attempt.
Callbacks run without another acquisition. Capture/allocation and callback errors
propagate; CSV export errors are not silently suppressed as optional console output.
Outside the context, raw rows are not retained. Context restoration also runs on errors.
The context applies to the current thread/context, not launches dispatched to workers.

External writers must represent missing counters/shapes as unavailable. Summary mode
does not provide pipe-utilization metrics; use `TRITON_NPU_BENCH_PROFILER_MODE=full`
when those counters are required. Full mode still does not enable shape recording.
No host-latency measurement is added. For multi-kernel operation exports, a writer can
evaluate the existing quality checks per kernel on the retained samples; this reporting
check does not change the operation-level retry decision. Exporting a failed retained
attempt preserves its quality status and follows the existing fastest-score policy.
