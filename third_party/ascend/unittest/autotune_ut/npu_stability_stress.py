"""Opt-in synthetic instability cases; oracle failures are deliberately ordinary red tests."""

import csv
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

SCENARIOS = ("upper_outliers", "lower_outliers", "heavy_tail", "burst", "periodic", "rising_drift", "falling_drift",
             "step_up", "step_down", "bimodal", "warmup_leak", "filter_recovery", "winner_reanchoring",
             "missing_profile", "invalid_profile", "stable_bias", "between_batch_shift", "retained_bad_attempt",
             "filter_floor_outlier", "quantization")
BASE_US = (40., 42., 120.)


def contaminated(values, scenario, variant, rng):
    n, strength = len(values), 6 + 2 * variant
    count = n // 10
    if scenario == "upper_outliers":
        values[rng.choice(n, count, replace=False)] += 10 * strength
    elif scenario == "lower_outliers":
        values[rng.choice(n, count, replace=False)] /= strength
    elif scenario == "heavy_tail":
        values += rng.lognormal(0, 1.5, n) * strength / 4
    elif scenario == "burst":
        begin = n * variant // 12
        values[begin:begin + n // 3] += 4 * strength
    elif scenario == "periodic":
        values[variant % 5::5] += 4 * strength
    elif scenario == "rising_drift":
        values += np.linspace(0, strength, n)
    elif scenario == "falling_drift":
        values += np.linspace(strength, 0, n)
    elif scenario == "step_up":
        values[n // 2:] += strength
    elif scenario == "step_down":
        values[:n // 2] += strength
    elif scenario == "bimodal":
        values[::2] += strength
    elif scenario == "warmup_leak":
        values[:n // 3] += 4 * strength
    return values


@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("variant", range(1, 6), ids=lambda value: f"v{value}")
def test_stability_reference(backend, monkeypatch, tmp_path, scenario, variant):
    policy = backend.policy
    options = policy.resolve_options()
    assert (options.cache_mode, options.warmup, options.active, options.lower_quantile, options.upper_quantile,
            options.quality_check, options.max_retries, options.measure_budget_ms, options.calibration_runs,
            options.filter_slow_configs, options.slow_config_rounds, options.slow_config_factor,
            options.slow_config_recheck_delay_s) == ("hot", 50, 250, .25, .75, True, 5, 10, 50, True, 5, 2, .2)
    order = [1, 0, 2] if scenario == "quantization" else [0, 1, 2]
    funcs = [lambda index=index: index for index in order]
    event_calls, pauses, batches, counts, first_means, reports, reads = [], [], [], [], {}, [], []
    acquired, quality_failures, attempt_means = [], [], {}
    estimates = [0, 0, 0]

    @contextmanager
    def timer(cold, scope):
        assert not cold and scope is None

        def measure(fn, count):
            index = fn()
            assert count == options.calibration_runs
            event_calls.append(index)
            estimates[index] += 1
            duration = BASE_US[index]
            if scenario in {"filter_recovery", "winner_reanchoring"}:
                if index == 0 and estimates[index] <= max(1, variant - 1):
                    duration = 200 + 20 * variant
                if scenario == "winner_reanchoring" and index == 1:
                    duration = 80
            elif scenario == "filter_floor_outlier" and index == 1:
                duration = 14 + variant
            return duration / 1000

        yield measure

    import pandas as pd
    read_csv = pd.read_csv

    def read(*args, **kwargs):
        reads.append(True)
        return read_csv(*args, **kwargs)

    def profile(selected, warmup, active, cold, directory, keep, names, **kwargs):
        labels = [fn() for fn in selected]
        batches.append(labels)
        counts.append(active)
        attempt = len(batches)
        assert labels == [index for index in order if index in labels]
        assert warmup == 50 and not cold and names == ["same"] * len(labels)
        assert "_active_counts" not in kwargs
        if scenario == "missing_profile" and attempt <= variant:
            return policy._collect_prof_result(directory, selected, warmup, active, "same", _return_samples=True)
        path = Path(directory) / "kernel_details.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Name", "Type", "Start Time(us)", "Duration(us)"))
            timestamp = 0
            for index in labels:
                rng = np.random.default_rng(100 * variant + index)
                values = BASE_US[index] + rng.permutation(np.linspace(-.08, .08, active))
                if attempt <= variant and index == 0:
                    values = contaminated(values, scenario, variant, rng)
                if scenario == "stable_bias" and index == 0:
                    values += 10 * variant
                if scenario == "between_batch_shift":
                    values *= (1.5 + .1 * variant) if index == 0 else 1
                if scenario == "retained_bad_attempt" and attempt == 1 and index == 1:
                    values *= .5
                    values[-active // 10:] += 200 * variant
                if scenario == "quantization":
                    quantum = (5, 6, 8, 10, 20)[variant - 1]
                    values = np.round(values / quantum) * quantum
                if scenario == "invalid_profile" and attempt == 1 and index == 0 and variant in (2, 3, 4):
                    values[active // 2] = (np.nan, 0., -1.)[variant - 2]
                first_means.setdefault(index, float(np.mean(values)))
                attempt_means[attempt, index] = float(np.mean(values))
                for launch, duration in enumerate(np.r_[np.full(warmup, 999.), values]):
                    timestamp += 1000
                    if scenario == "invalid_profile" and attempt == 1 and index == 0:
                        if variant == 1 and launch == warmup + active - 1:
                            continue
                        if variant == 5 and launch == warmup + active // 2:
                            timestamp -= 1000
                    writer.writerow(("same", "Kernel", timestamp, duration))
        records = policy._collect_prof_result(directory, selected, warmup, active, "same", cold,
                                              _return_samples=kwargs["_return_samples"], _capture_report=True,
                                              _quantiles=kwargs["_quantiles"])
        acquired.append(labels)
        return records

    evaluate_quality = backend.quality.evaluate_quality

    def evaluate(*samples):
        metrics, failures = evaluate_quality(*samples)
        quality_failures.append(bool(failures))
        return metrics, failures

    monkeypatch.setattr(policy, "event_timer", timer)
    monkeypatch.setattr(policy.time, "sleep", pauses.append)
    monkeypatch.setattr(pd, "read_csv", read)
    monkeypatch.setattr(backend.quality, "evaluate_quality", evaluate)
    costs = policy.bench_npu(profile, funcs, 5, 30, True, tmp_path, False, "same", options=options,
                             report_sink=reports.extend)
    scores = dict(zip(order, costs))
    winner = min(scores, key=scores.get)
    initial_mean_winner = min(first_means, key=first_means.get)
    assert counts and len(set(counts)) == 1 and counts[0] >= 250
    assert len(reads) == len(batches) - (variant if scenario == "missing_profile" else 0)
    assert len(reports) == 3 and all(report is None or report.samples is None for report in reports)
    for index, report in zip(order, reports):
        if report is not None:
            assert report.warmup == 50 and report.active == report.sample_count == counts[0]
            assert report.mean_ms * 1000 == pytest.approx(attempt_means[report.attempt, index])
    outcomes = iter(quality_failures)
    for previous, following in zip(acquired, acquired[1:]):
        assert following == [index for index in previous if next(outcomes)], "quality retries were not targeted"
    assert all(pause == .2 for pause in pauses)
    assert not list(tmp_path.iterdir()), "profiler acquisitions were not cleaned up"
    if scenario == "missing_profile":
        assert len(batches) >= variant + 1 and all(batch == batches[0] for batch in batches[:variant + 1])
    elif scenario == "invalid_profile":
        assert len(batches) >= 2 and batches[1] == batches[0]
    elif scenario == "winner_reanchoring":
        assert estimates[2] > 1, "new threshold did not revisit a previously retained candidate"
    elif scenario in SCENARIOS[:11]:
        assert len(batches) >= 2 and quality_failures[0], "contamination did not trigger a quality retry"
    assert winner == 0, (f"{scenario}/v{variant}: independent winner=0 (40 us), selected={winner}, "
                         f"scores_ms={scores}, first_mean_winner={initial_mean_winner}, batches={batches}")
