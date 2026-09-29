"""Calibrated retry heuristics on all samples; pairwise statistics cost O(N**2)."""

import numpy as np

THRESHOLDS = {
    "mean_trimmed_relative_deviation": 0.04,
    "central_gap_ratio": 0.25,
    "dip_statistic": 0.035,
    "normalized_drift": 0.35,
    "normalized_change_point": 0.35,
}


def central_50_mean(values):
    values = np.sort(values)
    kept = max(1, len(values) // 2)
    start = (len(values) - kept) // 2
    return float(np.mean(values[start:start + kept]))


def _ratio(numerator, denominator):
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    if abs(denominator) > 1e-15:
        return float(numerator / denominator)
    return 0. if abs(numerator) <= 1e-15 else float("inf")


def _lower_envelope(y, x):
    fit, knots, start = np.empty(len(y)), [0], 0
    fit[0] = y[0]
    while start < len(y) - 1:
        slopes = (y[start + 1:] - y[start]) / (x[start + 1:] - x[start])
        stop = start + 1 + int(np.argmin(slopes))
        slope = (y[stop] - y[start]) / (x[stop] - x[start])
        fit[start:stop + 1] = y[start] + (x[start:stop + 1] - x[start]) * slope
        knots.append(stop)
        start = stop
    return fit, np.asarray(knots)


def _dip(values):
    # Preserve the calibrated prototype's convention for small/discrete support.
    support, counts = np.unique(values, return_counts=True)
    if len(support) <= 4:
        return 0.
    probability = counts.astype(float) / len(values)
    empirical_cdf = np.cumsum(probability)
    x, cdf = support.copy(), empirical_cdf.copy()
    left_envelope, right_envelope, separation = [0.], [1.], 0.
    for _ in range(len(support) + 5):
        lower, lower_knots = _lower_envelope(cdf - probability, x)
        reflected, reflected_knots = _lower_envelope(1 - cdf[::-1], x[-1] - x[::-1])
        upper, upper_knots = 1 - reflected[::-1], len(cdf) - 1 - reflected_knots[::-1]
        lower_gaps = np.abs(upper[lower_knots] - lower[lower_knots])
        upper_gaps = np.abs(upper[upper_knots] - lower[upper_knots])
        lower_gap, upper_gap = np.max(lower_gaps), np.max(upper_gaps)
        if upper_gap > lower_gap:
            right = upper_knots[upper_gaps == upper_gap][-1]
            left, gap = lower_knots[lower_knots <= right][-1], upper_gap
        else:
            left = lower_knots[lower_gaps == lower_gap][0]
            right, gap = upper_knots[upper_knots >= left][0], lower_gap
        if gap <= separation or right == 0 or left == len(cdf):
            left_error = np.max(np.abs(empirical_cdf[:len(left_envelope)] - left_envelope))
            right_error = np.max(np.abs(empirical_cdf[-len(right_envelope) - 1:-1] - right_envelope))
            return float(.5 * max(left_error, right_error))
        separation = max(separation, np.max(np.abs(lower[:left + 1] - cdf[:left + 1])),
                         np.max(np.abs(upper[right:] - cdf[right:] + probability[right:])))
        left_envelope.extend(lower[1:left + 1])
        right_envelope[:0] = upper[right:-1].tolist()
        cdf, x, probability = cdf[left:right + 1], x[left:right + 1], probability[left:right + 1]
    raise RuntimeError("Dip statistic did not converge")


def evaluate_quality(times, durations):
    """Return named metrics and failure reasons for validated profiler samples."""
    order = np.argsort(times, kind="stable")
    times, values = np.asarray(times)[order], np.asarray(durations)[order]
    n = len(values)
    # Qn: scaled order statistic of all pairwise differences. Never cache pairs.
    ii, jj = np.triu_indices(n, k=1)
    differences = np.abs(values[jj] - values[ii])
    rank = (n // 2 + 1) * (n // 2) // 2
    scale = 2.2219 * float(np.partition(differences, rank - 1)[rank - 1])
    del differences
    # Theil-Sen drift over the full device-time span, normalized by Qn.
    slope = float(np.median((values[jj] - values[ii]) / (times[jj] - times[ii])))
    del ii, jj
    # Largest gap in the central 80% of sorted samples, in units of Qn.
    gaps = np.diff(np.sort(values))
    lo, hi = max(1, int(np.ceil(.1 * n))), min(n - 1, int(np.floor(.9 * n)))
    central_gaps = gaps if hi < lo else gaps[lo - 1:hi]
    # Normalized Pettitt rank statistic: detect a shift within the acquisition.
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    change = np.max(np.abs(2 * np.cumsum(ranks)[:-1] - np.arange(1, n) * (n + 1))) / (n * n // 4)
    metrics = {
        # Tail contamination: compare the mean with the central-half mean.
        "mean_trimmed_relative_deviation": abs(_ratio(np.mean(values), central_50_mean(values)) - 1),
        "central_gap_ratio": _ratio(np.max(central_gaps), scale),
        "dip_statistic": _dip(values),  # Multimodality statistic, not a p-value.
        "normalized_drift": _ratio(abs(slope) * (times[-1] - times[0]), scale),
        "normalized_change_point": float(change),
    }
    failures = [
        f"{name}={value:.8g} (limit={THRESHOLDS[name]:g})" for name, value in metrics.items()
        if not np.isfinite(value) or value > THRESHOLDS[name]
    ]
    return metrics, failures
