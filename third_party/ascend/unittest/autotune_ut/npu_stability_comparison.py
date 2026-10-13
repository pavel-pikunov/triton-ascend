"""Compare real legacy collection and do_bench control flow on synthetic timing inputs."""

import csv
import json
import math
from types import SimpleNamespace

import pytest

from npu_stability_stress import BASE_US, SCENARIOS, sample_values

FILTER_CASES = {"filter_recovery", "winner_reanchoring", "filter_floor_outlier"}
CASES = [
    (method, scenario)
    for method in ("npu_legacy", "triton")
    for scenario in SCENARIOS
    if scenario not in FILTER_CASES and (method != "triton" or scenario not in {"missing_profile", "invalid_profile"})
]


def event_scores(testing, monkeypatch, scenario, variant, order):
    """Run unchanged do_bench, replacing only interface/cache acquisition operations."""
    scores, counts = {}, {}
    for index in order:
        events, measured = [], []
        clock, timed = 0., False
        values = None

        class Event:

            def __init__(self, **kwargs):
                self.position = len(events)
                events.append(self)

            def record(self):
                nonlocal timed
                self.clock = clock
                timed = len(events) > 2 and self.position < 2 + (len(events) - 2) // 2

            def elapsed_time(self, end):
                return (end.clock - self.clock) / 1000

        def run():
            nonlocal clock, values
            duration = BASE_US[index]
            if timed:
                if values is None:
                    values = sample_values(scenario, variant, index, (len(events) - 2) // 2)
                duration = values[len(measured)]
                measured.append(duration)
            clock += duration

        # Keep the real driver and package imports; replace two device operations.
        driver_type = type(testing.runtime.driver.active)
        interface = SimpleNamespace(Event=Event, synchronize=lambda: None)
        cache = SimpleNamespace(zero_=lambda: None)
        monkeypatch.setattr(driver_type, "get_device_interface", lambda self: interface, raising=False)
        monkeypatch.setattr(driver_type, "get_empty_cache_for_benchmark", lambda self: cache, raising=False)
        scores[index] = testing.do_bench(run, quantiles=(.5, .2, .8))
        counts[index] = len(measured)
        assert counts[index] == max(1, int(100 / (BASE_US[index] / 1000)))
    return scores, counts


@pytest.mark.parametrize("method,scenario", CASES, ids=[f"{method}-{scenario}" for method, scenario in CASES])
@pytest.mark.parametrize("variant", range(1, 6), ids=lambda value: f"v{value}")
def test_baseline_reference(backend, monkeypatch, tmp_path, record_property, method, scenario, variant):
    order = [1, 0, 2] if scenario == "quantization" else [0, 1, 2]
    fallback = False
    if method == "triton":
        costs, counts = event_scores(backend.events, monkeypatch, scenario, variant, order)
    else:
        warmup, active = 5, 30
        if scenario != "missing_profile":
            with (tmp_path / "kernel_details.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("Name", "Type", "Start Time(us)", "Duration(us)"))
                timestamp = 0
                for index in order:
                    values = sample_values(scenario, variant, index, active)
                    for launch, duration in enumerate([999.] * warmup + values.tolist()):
                        timestamp += 1000
                        if scenario == "invalid_profile" and index == 0:
                            if variant == 1 and launch == warmup + active - 1:
                                continue
                            if variant == 5 and launch == warmup + active // 2:
                                timestamp -= 1000
                        writer.writerow(("same", "Kernel", timestamp, duration))
        funcs = [lambda index=index: index for index in order]
        try:
            result = backend.legacy._collect_prof_result(str(tmp_path), funcs, warmup, active, "same", False)
            costs, counts = dict(zip(order, result)), dict.fromkeys(order, active)
        except backend.legacy.ProfilerResultMismatchError:
            # The unchanged legacy tuner falls back to events on row-count mismatch.
            fallback = True
            costs, counts = event_scores(backend.events, monkeypatch, "clean", variant, order)
    scores = {index: score[0] if isinstance(score, list) else score for index, score in costs.items()}
    winner = min(costs, key=costs.get)
    # This case checks each path's actual criterion, not a fixed clean-regime oracle.
    expected = 1 if scenario == "retained_bad_attempt" and method == "triton" else 0
    for name, value in {
            "method": method, "scenario": scenario, "variant": variant, "winner": winner, "expected": expected,
            "scores_ms": json.dumps(scores), "active": json.dumps(counts), "profiler_attempts":
            int(method == "npu_legacy"), "event_fallback": fallback
    }.items():
        record_property(name, value)
    assert math.isfinite(scores[winner]) and scores[winner] > 0, f"unusable selected score: {scores}"
    assert winner == expected, f"{method}/{scenario}/v{variant}: expected winner={expected}, selected={winner}, {scores}"
