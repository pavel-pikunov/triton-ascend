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

`runtime/_legacy_autotuner.py` copies release `runtime/autotuner.py` with only
three substitutions: the mode name and its two imports of the copied testing
module. Trailing whitespace on one original line was removed.
`_legacy_testing.py` copies release `testing.py` with only the two environment
overrides described below. Shared parsers, generators, UBTuner and compiler
are unchanged from the pinned release.

The original common profiler session, CV pruning/count calculation, warmup,
kernel-name filtering, CSV aggregation, synchronization, exceptions and cleanup
are preserved. L2 eviction in legacy mode still occurs before kernel hooks.
The new quality checks, retries, budget, slow filter, stage logs, winner report
and timing report are not applied to legacy. Added decorator arguments
`npu_bench_options`, `report_best_config` and `report_timing` are ignored there;
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
