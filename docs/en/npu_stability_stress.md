# Synthetic NPU stability stress cases

This test-branch suite contains **20 scenarios x 5 variants = 100 cases**.
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
of the contaminated samples. Successful noise cases return to clean data after
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
| `retained_bad_attempt` | Fast failed-quality attempt survives a slower good retry. |
| `filter_floor_outlier` | An optimistic event minimum lowers the pruning threshold. |
| `quantization` | Timer resolution hides the 5% difference; acquisition order breaks the tie. |

Persistent bias, between-block shifts and coarse quantization are outside the
recovery guarantee: normal distinguishable measurements never become available
to the selection pass. More launches or retries cannot establish the reference
winner under those conditions. These are applicability limits, not regressions.
The optimistic-minimum and retained-attempt scenarios also show how history can
remain decisive after later measurements return to normal.

The last five scenarios require the independent winner even where current
protections cannot recover it. Their assertions remain ordinary failures:
no xfail, skip or fabricated exception. In particular, retaining the fastest
attempt regardless of quality remains the requested production policy; a failed
oracle assertion documents its limit, not a changed implementation contract.
Finite clean samples can themselves fail a heuristic quality threshold. Retry
addressing is checked against the actual metrics rather than assuming that every
clean candidate passes; all statistical algorithms remain unchanged.

All rows are generated in temporary directories and removed after each attempt;
no recorded datasets are committed. Reports retain aggregates, not raw rows.
Every case also checks one CSV read per attempted acquisition, shared active,
candidate order, targeted retries, warmup exclusion and cleanup where applicable.

## Run

The file is opt-in so known red cases do not enter ordinary smoke collection.
Run it explicitly on the usual installed Ascend test environment:

```bash
pytest third_party/ascend/unittest/autotune_ut/npu_stability_stress.py -q --dist=loadfile
```

`-vv` exposes all 100 scenario/variant IDs. No native profiler sessions are
created by this synthetic suite; existing real-device smoke tests remain separate.

Local verification: Python 3.11 CPU helpers, byte-identical test assertions,
**75 passed / 25 failed**, five failures in each applicability-limit scenario.
Every failure was an independent-winner assertion; there were no acquisition
errors, skips or xfails in the completed CPU run. The normal repository fixture
could not be collected locally because torch_npu is absent. This result verifies
synthetic helper behavior, not native NPU execution.
