# Hyper-autotune synthetic benchmark

This benchmark simulates an environment that uses the real Ascend hyperparameter optimizer. It does **not** simulate the optimizer itself: `tools/hyper_autotune_simulation_benchmark.py` imports `HyperparameterAutotuner`, which lazily imports the real optional `optuna` package.

## Requirements

Run from the repository root with Python and the optional Optuna dependency installed. Ascend hardware, CANN, and a built Triton runtime are not required.

```bash
python3 -m pip install optuna
python3 tools/hyper_autotune_simulation_benchmark.py --trials 80 --repeats 5
```

## What it checks

- **Functionality:** the real `HyperparameterAutotuner` proposes vectors and converts them to `--hyper-max-parallel-parameters=...` flags.
- **Accuracy:** a nonlinear synthetic kernel is compared with a deterministic reference for both baseline and tuned vectors.
- **Performance:** the tuned vector must beat a deliberately poor baseline by at least `--min-speedup`.
- **Cache path:** the chosen vector/objective is stored in and retrieved from `HyperparameterAutotuneCache`.

Useful knobs:

```bash
python3 tools/hyper_autotune_simulation_benchmark.py \
  --trials 120 \
  --timeout-sec 60 \
  --seed 0 \
  --repeats 7 \
  --min-speedup 1.25
```
