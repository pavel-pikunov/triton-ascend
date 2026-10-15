# NPU quality CPU performance audit

Measured on 2026-10-15, Windows, Intel Core Ultra 7 258V (8 cores),
Python 3.11.9, NumPy 1.26.4, pandas 2.2.3 and SciPy 1.13.1.

The audit covers 100 seeded stress variants and eight additional distributions,
each with 24 candidates and either 250 or 5000 active rows: **216 cases**.
All five quality metrics run for every candidate, even after failure.
Dip and the inexpensive central-gap check both remain unchanged.

Each timing includes one pandas CSV read, name filtering, warmup exclusion,
score calculation, aggregates for every candidate, all quality checks, retry
bookkeeping and temporary-directory cleanup for **one profiler attempt**.
Filtering and budget calibration are disabled in this processing audit.
Input contains 32 columns and 50 warmup rows per candidate; the largest file
has 121200 rows and is approximately 49.8 MiB. CSV fixture generation and
validation are outside the timer. Native NPU launches, synchronization, CANN
analysis/export and final console printing were not run or timed.

Three serial repetitions per case, using the same verified, ready-made CSV.
Values below are medians of three; this measures processing with filesystem
cache reuse, not an uncached storage benchmark. Budgets apply per attempt,
not to the sum of multiple quality retries.

| Active rows per candidate | Cases within budget | Median across cases, s | Worst case median, s | Worst individual repetition, s |
| --- | ---: | ---: | ---: | ---: |
| 250 | 108/108 (1 s) | 0.080 | 0.117 | 0.251 |
| 5000 | 108/108 (2 s) | 0.642 | 1.586 | 1.631 |

Worst variant median for each distribution:

| Distribution | 24 x 250, s | 24 x 5000, s |
| --- | ---: | ---: |
| upper_outliers | 0.085 | 0.785 |
| lower_outliers | 0.086 | 0.866 |
| heavy_tail | 0.082 | 0.873 |
| burst | 0.078 | 0.844 |
| periodic | 0.081 | 0.900 |
| rising_drift | 0.081 | 0.929 |
| falling_drift | 0.092 | 0.802 |
| step_up | 0.117 | 1.045 |
| step_down | 0.078 | 0.617 |
| bimodal | 0.083 | 0.805 |
| warmup_leak | 0.077 | 0.735 |
| filter_recovery | 0.088 | 0.585 |
| winner_reanchoring | 0.090 | 0.591 |
| missing_profile | 0.084 | 0.641 |
| invalid_profile | 0.094 | 0.780 |
| stable_bias | 0.101 | 0.772 |
| between_batch_shift | 0.101 | 0.638 |
| retained_bad_attempt | 0.097 | 0.580 |
| filter_floor_outlier | 0.090 | 0.632 |
| quantization | 0.080 | 0.347 |
| gaussian | 0.100 | 0.640 |
| rounded | 0.096 | 0.450 |
| uniform | 0.097 | 0.634 |
| constant | 0.077 | 0.352 |
| linear | 0.088 | 0.505 |
| near_linear | 0.093 | 0.654 |
| collinear_wide | 0.089 | 1.586 |

The previous full-check Gaussian 24 x 5000 path took 23.382 s in one CPU
profile; the new median is 0.640 s. The former run included
dip (about 0.05 s); the improvement comes from removing Python
pair comparators and vectorizing report aggregates. This is a local CPU
comparison, not an NPU autotune speedup claim.

Qn and Theil-Sen retain their full-pair order statistics. Sampled slopes
suggest pivots only; exact counts certify the selected rank. Large inputs
use linear auxiliary memory, including collinear fallback blocks.

Validation:

- 33 reference, resource, aggregation and retry-control tests passed, including
  a two-separated-modes case caught by the inexpensive gap check.
- Qn, Theil-Sen and crossing counts matched direct all-pair references on
  308 datasets, including eight 5000-sample shapes, with SciPy 1.13.1 (Python 3.11)
  and 1.15.1 (Python 3.12).
- Both short-circuit and full-check modes produced identical winners, scores,
  active counts and profiler-attempt counts in all 100 stress models.
- New NPU: 80/100 expected choices in each mode. The 20 intentional red
  oracle cases per mode remain visible. Legacy: 14/85; default: 30/75.
- The local CPU runner loads byte-identical helper sources and normal installed
  NumPy/pandas/SciPy/Triton. Only device acquisition is replaced with generated
  data; no torch_npu backend or driver is substituted. Native Ascend integration
  was not collected because this machine has no installed torch_npu backend.

Run the opt-in performance cases on the normal installed-backend environment:

```bash
pytest third_party/ascend/unittest/autotune_ut/npu_quality_performance.py -q -n 0
```

Raw per-case times and overlapping component medians:
[npu_quality_performance.csv](npu_quality_performance.csv). `collector_total`
already includes CSV reading and aggregation; `quality_total` includes Qn,
Theil-Sen and dip. Do not add nested components to their parent totals.
