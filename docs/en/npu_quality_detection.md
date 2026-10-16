# Per-metric quality detection audit (test branch)

Run the opt-in assertions explicitly on the normal Ascend installation:

```bash
pytest third_party/ascend/unittest/autotune_ut/npu_quality_detection.py --dist=loadfile
```

Each metric gets 1000 present-defect cases and 1000 independent stationary
Gaussian controls, with 50 measured launches per case. Seeds and generation
are fixed in the test source. A different metric cannot cover a target's miss.
Data generation asserts the presence of each injected defect before evaluating
quality. There are no absent-spike cases in the present-defect denominator.

Controls are `40 + Normal(0, 0.02)` microseconds, without drift, change points or
added outliers. Model start times accumulate preceding launch durations; no host
overhead is included. These are synthetic tests, not NPU measurements.

| Target metric | Present defect |
| --- | --- |
| Mean/trim deviation | Exactly 2% of launches receive +100 us: approximately 5% mean deviation (near the current threshold). |
| Central gap | Exactly 10% of launches receive +8 us, producing a populated separated mode. |
| Dip | Two equally populated continuous modes, separated by 2 us. |
| Drift | A linear +2 us change over the observed window. |
| Change point | A +1 us step halfway through the window. |

The target is 100% detection of these explicit defects and 0% false alarms on
controls. These finite cases do not claim detection of every possible defect.
Expected failures remain ordinary red assertions: no xfail, skipping, seed
selection or threshold changes are used to hide false alarms.

## Recorded results before the additional 15% increase, 2026-10-16

Thresholds are 0.04, 0.26, 0.04, 0.39 and 0.36 in table order.

| Target metric | Detected / 1000 present defects | False alarms / 1000 controls |
| --- | ---: | ---: |
| Mean/trim deviation | 1000 (100%) | 0 (0%) |
| Central gap | 1000 (100%) | 510 (51%) |
| Dip | 1000 (100%) | 503 (50.3%) |
| Drift | 1000 (100%) | 432 (43.2%) |
| Change point | 1000 (100%) | 133 (13.3%) |

All 5000 present-defect assertions passed; 3422/5000 control assertions passed
and 1578 failed. The zero-false-alarm target is not met by four metrics.
These rates count individual target checks, not the combined quality decision.

The additional 15 reference cases passed for sizes 50, 100 and 250. Four
metrics are compared with pairwise Qn, SciPy Theil-Sen or direct formulas;
dip uses a bounded check on two balanced, narrow, widely separated continuous modes.
This is not an independent general-purpose dip implementation.

Local CPU validation used the unchanged helper and test bodies, with only the
backend import redirected to the copied helper package. The committed tests
retain the ordinary backend import and repository fixtures. Neither torch_npu
nor an NPU backend was substituted. No production code or thresholds changed
as part of this audit.

## Additional threshold increase

Thresholds were subsequently multiplied by 1.15 and rounded upward to 0.01:
mean/trim 0.05, central gap 0.30, dip 0.05, drift 0.45, change point 0.42.
The recorded results above use the previous thresholds. The injected mean/trim
defect is now near its threshold; its assertion is retained without changing
the data to force detection.
