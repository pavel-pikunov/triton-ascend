"""Optional autotune diagnostics, independent of benchmark and cache policies."""

import csv
import io
import os
import pprint
import warnings
from dataclasses import dataclass

REPORT_ENV = "TRITON_NPU_BENCH_REPORT_BEST_CONFIG"


@dataclass(frozen=True)
class NpuMeasurementReport:
    columns: tuple
    rows: tuple
    warmup: int
    active: int
    cache_mode: str
    mean_ms: float
    attempt: int = 1
    quality_failures: tuple = ()


def resolve_report_best_config(argument=None):
    if argument is not None:
        if not isinstance(argument, bool):
            raise ValueError("report_best_config must be a boolean or None")
        return argument
    value = os.getenv(REPORT_ENV, "0").strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise ValueError(f"Invalid {REPORT_ENV}: {value!r}")
    return value in {"1", "true"}


def report_safely(action):
    """Diagnostic failures must not affect measurement, retries or execution."""
    try:
        return action()
    except Exception as exc:
        try:
            warnings.warn(f"Autotune report unavailable: {exc}", RuntimeWarning, stacklevel=3)
        except Exception:
            # Even a warnings-as-errors policy must not change autotuning.
            pass


def print_best_config_report(function_name, configs, config, measurement=None):
    """Print all configuration fields and unabridged profiler CSV values."""
    number = next((i for i, candidate in enumerate(configs, 1) if candidate is config), None)
    stream = io.StringIO()
    stream.write(f"Triton autotuning result for {function_name}\n")
    stream.write(f"Selected config: {number}/{len(configs)} (1-based, before pruning)\n"
                 if number is not None else "Selected config: not in the original candidate list\n")
    parameters = dict(vars(config))
    ub_config = parameters.pop("ubtune_cfg", None)
    stream.write("Full config parameters:\n" + pprint.pformat(parameters, sort_dicts=False) + "\n")
    if ub_config is not None:
        stream.write("ubtune_cfg:\n" + pprint.pformat(ub_config, sort_dicts=False) + "\n")
    if measurement is None:
        stream.write("NPU profiler measurements unavailable for this selection; no extra profiling was run.\n")
    else:
        kernel_names = list(dict.fromkeys(row.get("Name", "") for row in measurement.rows))
        stream.write(f"Profiler kernel: {', '.join(kernel_names)}\n")
        stream.write(f"Mean duration: {measurement.mean_ms * 1000:.12g} us "
                     f"({measurement.mean_ms:.12g} ms)\n")
        stream.write(f"cache={measurement.cache_mode}, warmup={measurement.warmup}, "
                     f"active={measurement.active}, selected attempt={measurement.attempt}\n")
        if measurement.quality_failures:
            stream.write("Selected measurement failed quality checks: " + "; ".join(measurement.quality_failures) +
                         "\n")
        stream.write("Measured launches: all kernel_details.csv columns (warmup excluded)\n")
        writer = csv.DictWriter(stream, fieldnames=measurement.columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(measurement.rows)
    print(stream.getvalue(), end="")
