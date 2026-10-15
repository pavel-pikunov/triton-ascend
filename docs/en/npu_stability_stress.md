# Synthetic NPU stability stress cases

This test-branch suite contains **20 scenarios x 5 variants = 100 models**,
run in both short-circuit and full-check modes: **200 NPU cases**.
It exercises the real NPU benchmark helpers, pandas collector and quality
algorithms with generated event estimates and profiler rows. It does not emulate
an NPU backend or validate native torch_npu timing, synchronization or L2 residency.
The ordinary installed-backend fixture is retained.

## Literature reviewed before implementation

Search date: 2026-10-14. Searches covered academic publisher, institutional and
arXiv sources for benchmark variability, warmup, autocorrelation and selection
bias. Included primary research and official methodological references; excluded
blogs, secondary summaries and hardware-specific noise claims without data.
This is a focused design review, not an exhaustive systematic survey.

- [Kalibera and Jones, 2013](https://kar.kent.ac.uk/33611/): repetition should target
  the level where uncertainty arises. More launches in one invocation cannot
  substitute for independent executions or remove systematic bias.
- [Mytkowicz et al., 2009](https://sape.inf.usi.ch/publications/asplos09.html):
  apparently innocuous setup changes can bias performance comparisons. Stable
  measurements alone do not establish a correct ranking.
- [Hoefler and Belli, 2015](https://htor.inf.ethz.ch/publications/index.php?pub=222):
  parallel performance claims need interpretable experimental conditions and
  statistical reporting, including the possibility of noise-driven improvements.
- [Barrett et al., 2017](https://arxiv.org/abs/1602.00602): VM measurements can have
  warmup, slowdown and multiple change points. We reuse these time-series shapes
  as synthetic stress models; VM results are not evidence of their NPU frequency.
- [Zhang, 2006, NIST](https://www.nist.gov/publications/calculation-uncertainty-mean-autocorrelated-measurements):
  autocorrelation invalidates uncertainty estimates that assume independent
  launches. Burst and periodic cases deliberately violate that assumption.
- [Cawley and Talbot, 2010](https://www.jmlr.org/beta/papers/v11/cawley10a.html):
  optimizing a noisy finite-sample criterion can produce selection bias. Applying
  this principle to the minimum score across retries is an inference; the paper
  studies model selection rather than NPU autotuning.

These sources motivate the models, not their numerical amplitudes. They do not
validate the current thresholds or prove that the new path beats legacy on hardware.

## Cases and independent reference

Each case has three labelled candidates whose clean device durations are
40, 42 and 120 us. The reference winner is the 40 us candidate, independently
of the contaminated samples, except for the retained-attempt contract below.
Successful noise cases return to clean data after
one through five attempts; variants also change severity or recovery timing.
Seeds and labels are fixed. Runs use the selected defaults:
hot cache, warmup=50, active>=250, central-half score, quality enabled,
five retries, 10 ms budget, 50 calibration launches, five filtering rounds,
factor=2 and delay=0.2 s. Only sleeping and device acquisition are replaced;
the requested pauses are still recorded and asserted.

| Scenario | Stress applied |
| --- | --- |
| `upper_outliers` | Sparse high-duration samples. |
| `lower_outliers` | Sparse low-duration samples. |
| `heavy_tail` | Right-skewed positive noise. |
| `burst` | One correlated block of slow launches. |
| `periodic` | Recurring slow launches. |
| `rising_drift` | Increasing duration through the acquisition. |
| `falling_drift` | Decreasing duration through the acquisition. |
| `step_up` | Abrupt slowdown midway through the acquisition. |
| `step_down` | Abrupt recovery midway through the acquisition. |
| `bimodal` | Alternating fast and slow modes. |
| `warmup_leak` | Initialization overhead extends into measured rows. |
| `filter_recovery` | Initially slow event estimates recover before exclusion. |
| `winner_reanchoring` | A new winner makes previously retained candidates suspect. |
| `missing_profile` | One through five missing profiler acquisitions. |
| `invalid_profile` | Truncation, NaN, zero, negative duration or duplicate timestamp. |
| `stable_bias` | Persistent candidate-specific timing bias. |
| `between_batch_shift` | Stable rows, changed load between candidate blocks. |
| `retained_bad_attempt` | A real lower score is retained despite failed quality and later slower samples. |
| `filter_floor_outlier` | An optimistic event minimum lowers the pruning threshold. |
| `quantization` | Timer resolution hides the 5% difference; acquisition order breaks the tie. |

Persistent bias, between-block shifts and coarse quantization are outside the
recovery guarantee: normal distinguishable measurements never become available
to the selection pass. More launches or retries cannot establish the reference
winner under those conditions. These are applicability limits, not regressions.
The optimistic-minimum scenario shows how history can remain decisive after
later measurements return to normal. Its event minimum is injected directly;
there are no individual event samples proving such an observation is realistic.
Coarse quantization is also imposed artificially, not native profiler precision.
In the persistent-bias and between-block-shift cases, B is correctly faster
under the supplied measurements. Red assertions require A in an unobserved
clean regime; they are not failures to detect within-sample instability.

Four scenarios require the independent winner even where current
protections cannot recover it. Their assertions remain ordinary failures:
no xfail, skip or fabricated exception. The retained-attempt scenario instead
requires B: its central-half score is about 21 us, below A's 40 us. It checks
that the first failed-quality attempt survives later clean measurements around
42 us. Quality detects the gap and requests a retry; failure does not establish
that the fast samples are wrong. Requiring A here contradicted the requested
minimum-score policy and has been corrected in all five variants.
Finite clean samples can themselves fail a heuristic quality threshold. Retry
addressing is checked against the actual metrics rather than assuming that every
clean candidate passes; all five metrics retain their definitions and thresholds.

All rows are generated in temporary directories and removed after each attempt;
no recorded datasets are committed. Reports retain aggregates, not raw rows.
Every case also checks one CSV read per attempted acquisition, shared active,
candidate order, targeted retries, warmup exclusion and cleanup where applicable.

## Run

The file is opt-in so known red cases do not enter ordinary smoke collection.
Run it explicitly on the usual installed Ascend test environment:

```bash
pytest third_party/ascend/unittest/autotune_ut/npu_stability_stress.py third_party/ascend/unittest/autotune_ut/npu_stability_comparison.py -q --dist=loadfile
```

`-vv` exposes all 360 scenario/variant/path/mode IDs. No native profiler sessions are
created by this synthetic suite; existing real-device smoke tests remain separate.

Local verification: Python 3.11 CPU helpers, byte-identical test assertions,
**80 passed / 20 failed in each NPU mode**, five failures in each remaining
applicability-limit scenario. Winners, scores, active counts and attempt counts
match between both modes in all 100 models.
Every failure was an independent-winner assertion; there were no acquisition
errors, skips or xfails in the completed CPU run. The normal repository fixture
could not be collected locally because torch_npu is absent. This result verifies
synthetic helper behavior, not native NPU execution.

The separate [CPU performance audit](npu_quality_performance.md) measures every
metric on all distributions, including full 24 x 5000 batches and CSV aggregates.

## Comparison with the original paths

`npu_stability_comparison.py` adds 85 legacy and 75 default-path cases. It uses
the unchanged legacy pandas collector and real `triton.testing.do_bench` with
synthetic event clocks; only device-interface/cache acquisition is replaced.
The ordinary backend fixture keeps real package imports. Legacy row-count
mismatch invokes the same event fallback as the original tuner; the full tuner,
native profiler and compilation are outside this comparison.

The same seeded distribution model is generated at each path's sample count.
Legacy uses warmup=5 and active=30. The default do_bench calculates its counts
from a clean estimate: A=2500, B=2380 and C=833 in this overhead-free model,
with its original 25/100 ms warmup/measurement budgets. These are not predicted
hardware counts. Cache residency, event error and profiler overhead are not modeled.
Scores are respectively the central-half mean, full mean and median.

Cells show successful checks out of five. A dash denotes an inapplicable
mechanism, not a passing test. No skip or xfail hides the ordinary red assertions.

| Scenario | New NPU | Legacy NPU | Default |
| --- | ---: | ---: | ---: |
| upper_outliers | 5/5 | 0/5 | 5/5 |
| lower_outliers | 5/5 | 5/5 | 5/5 |
| heavy_tail | 5/5 | 0/5 | 0/5 |
| burst | 5/5 | 0/5 | 5/5 |
| periodic | 5/5 | 0/5 | 5/5 |
| rising_drift | 5/5 | 0/5 | 0/5 |
| falling_drift | 5/5 | 0/5 | 0/5 |
| step_up | 5/5 | 0/5 | 0/5 |
| step_down | 5/5 | 0/5 | 0/5 |
| bimodal | 5/5 | 0/5 | 0/5 |
| warmup_leak | 5/5 | 0/5 | 5/5 |
| filter_recovery | 5/5 | — | — |
| winner_reanchoring | 5/5 | — | — |
| missing_profile | 5/5 | 0/5 | — |
| invalid_profile | 5/5 | 4/5 | — |
| stable_bias | 0/5 | 0/5 | 0/5 |
| between_batch_shift | 0/5 | 0/5 | 0/5 |
| retained_bad_attempt | 5/5 | 5/5 | 5/5 |
| filter_floor_outlier | 0/5 | — | — |
| quantization | 0/5 | 0/5 | 0/5 |
| All applicable checks | 80/100 | 14/85 | 30/75 |

The retained-attempt row checks each path's actual criterion: NPU/default
choose B, legacy chooses A because the full mean of B includes the slow tail.
It is not a clean-regime accuracy comparison. On the 70 common cases excluding
this contract check, results are 55/70, 5/70 and 25/70 respectively.
Legacy's invalid-profile checks evaluate selection: truncation falls back to
events, NaN produces an unusable score, and zero/negative durations or duplicate
timestamps are not rejected. Four passes do not imply strict input validation.

Noise affects only A and disappears on later profiler attempts; the original
paths use one attempt, while NPU can retry. This deliberately favors retries
and does not estimate real selection accuracy or establish an NPU speedup.
Recovery is tied to attempt number, not elapsed wall/device time: the model
does not allow a long default-path acquisition to recover within that attempt.
All 360 outcomes were collected on CPU with byte-identical helper sources and assertions;
156 ordinary failures were selection-oracle/unusable-score assertions, with
no collection errors or skips. Installed do_bench/statistics matched the
repository source by AST. Python 3.10/3.11 syntax and repository style checks
are separate from these synthetic results.
