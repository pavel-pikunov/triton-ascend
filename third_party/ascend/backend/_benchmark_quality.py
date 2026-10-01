"""Calibrated retry heuristics on all samples, with linear auxiliary memory."""

from fractions import Fraction
from functools import cmp_to_key

import numpy as np
from scipy.stats import rankdata

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


def _qn(values):
    """Croux–Rousseeuw selection in the implicit sorted difference matrix.

    Each row contains sorted positive pair distances. Weighted row medians
    prune the matrix until only O(n) candidates remain; ties are counted on
    both sides of the pivot. The calibrated rank and factor are unchanged.
    """
    values = np.sort(values)
    n = len(values)
    rank = (n // 2 + 1) * (n // 2) // 2 - 1
    if n <= 256:
        i, j = np.triu_indices(n, 1)
        return 2.2219 * float(np.partition(values[j] - values[i], rank)[rank])
    rows = np.arange(n - 1)
    left, right = rows + 1, np.full(n - 1, n)
    discarded = 0
    while np.sum(right - left) > n:
        weights = right - left
        active = rows[weights > 0]
        midpoints = values[(left[active] + right[active] - 1) // 2] - values[active]
        order = np.argsort(midpoints)
        cumulative = np.cumsum(weights[active][order])
        pivot = midpoints[order[np.searchsorted(cumulative, (cumulative[-1] + 1) // 2)]]
        less, through = np.empty(n - 1, dtype=int), np.empty(n - 1, dtype=int)
        p = q = 1
        for i in rows:
            p, q = max(p, i + 1), max(q, i + 1)
            while p < n and values[p] - values[i] < pivot:
                p += 1
            while q < n and values[q] - values[i] <= pivot:
                q += 1
            less[i], through[i] = p, q
        below = int(np.sum(less - rows - 1))
        above = int(np.sum(through - rows - 1))
        if below <= rank < above:
            return 2.2219 * float(pivot)
        if rank < below:
            right = np.maximum(left, np.minimum(right, less))
        else:
            left = np.minimum(right, np.maximum(left, through))
        discarded = int(np.sum(left - rows - 1))
    candidates = np.concatenate([values[left[i]:right[i]] - values[i] for i in rows])
    rank -= discarded
    return 2.2219 * float(np.partition(candidates, rank)[rank])


def _slope_order(times, values, pivot, inclusive):
    """Order dual lines at a slope, breaking ties by acquisition order.

    Compare the actual pair slopes instead of y - pivot*x: the latter can
    lose differences when timestamps or durations have a large offset.
    """
    uncertain, rounded_tie = False, None
    tolerance = 8 * np.spacing(abs(pivot))

    def compare(i, j):
        nonlocal uncertain, rounded_tie
        if i == j:
            return 0
        a, b = (i, j) if i < j else (j, i)
        slope = (values[b] - values[a]) / (times[b] - times[a])
        # Rounded pair slopes can violate transitivity near a collinear fit.
        # In that case use exact streaming counts rather than a permutation.
        if 0 < abs(slope - pivot) <= tolerance:
            uncertain = True
        elif slope == pivot and pivot != 0 and np.isfinite(pivot) and not uncertain:
            dy = Fraction(float(values[b])) - Fraction(float(values[a]))
            dx = Fraction(float(times[b])) - Fraction(float(times[a]))
            if dy != Fraction(float(pivot)) * dx:
                # Multiple rounded ties can hide a crossing never compared
                # by the sort. A single sampled crossing is harmless.
                if rounded_tie is not None and rounded_tie != (a, b):
                    uncertain = True
                rounded_tie = (a, b)
        crossed = slope <= pivot if inclusive else slope < pivot
        return (1 if crossed else -1) * (1 if i < j else -1)

    return sorted(range(len(values)), key=cmp_to_key(compare)), uncertain


def _select_slope_direct(times, values, rank, pivot):
    """Linear-memory fallback for numerically ambiguous dual-line orderings.

    Count the actual rounded pair slopes, moving to an adjacent distinct
    slope until the requested rank is reached. This costs quadratic time per
    pass but never allocates pair matrices, including for collinear samples.
    """
    while True:
        below = through = 0
        previous, following = -np.inf, np.inf
        for i in range(len(values) - 1):
            slopes = (values[i + 1:] - values[i]) / (times[i + 1:] - times[i])
            less, greater = slopes[slopes < pivot], slopes[slopes > pivot]
            below += len(less)
            through += len(slopes) - len(greater)
            if len(less):
                previous = max(previous, np.max(less))
            if len(greater):
                following = min(following, np.min(greater))
        if below <= rank < through:
            return float(pivot)
        pivot = previous if rank < below else following


def _inversion_slopes(lower, upper, times, values, ranks=()):
    """Count crossings with merge sort; optionally report selected crossings.

    Sorted zero-based ranks select inversions uniformly without materializing
    them. Each merge reports a block of inversions against its remaining left
    run, so even sampling from O(n**2) crossings needs only O(n) space.
    """
    n = len(lower)
    positions = np.empty(n, dtype=int)
    positions[upper] = np.arange(n)
    source, target = list(lower), [0] * n
    slopes = np.empty(len(ranks))
    count = selected = 0
    width = 1
    while width < n:
        for start in range(0, n, 2 * width):
            middle, stop = min(start + width, n), min(start + 2 * width, n)
            i, j, out = start, middle, start
            while i < middle and j < stop:
                if positions[source[i]] < positions[source[j]]:
                    target[out] = source[i]
                    i += 1
                else:
                    size = middle - i
                    while selected < len(ranks) and ranks[selected] < count + size:
                        a, b = source[i + int(ranks[selected] - count)], source[j]
                        a, b = (a, b) if a < b else (b, a)
                        slopes[selected] = (values[b] - values[a]) / (times[b] - times[a])
                        selected += 1
                    count += size
                    target[out] = source[j]
                    j += 1
                out += 1
            target[out:stop] = source[i:middle] + source[j:stop]
        source, target = target, source
        width *= 2
    return count, slopes


def _select_slope(times, values, rank, rng):
    """Matoušek randomized interval contraction, followed by exact selection.

    Sampling chooses bounds only. Inversion counts certify every contraction;
    the returned value is an order statistic of all pairs, not an estimate.
    """
    n = len(values)
    lower, upper = list(range(n)), list(reversed(range(n)))
    below, count = 0, n * (n - 1) // 2
    while count > 8 * n:
        sample_ranks = np.sort(rng.integers(count, size=n))
        _, sample = _inversion_slopes(lower, upper, times, values, sample_ranks)
        sample.sort()
        center = (rank - below) * n // count
        radius = int(np.sqrt(n))
        pivots = np.unique(sample[np.clip([center - radius, center + radius], 0, n - 1)])
        for pivot in pivots:
            strict, uncertain = _slope_order(times, values, pivot, False)
            if uncertain:
                return _select_slope_direct(times, values, rank, sample[min(center, n - 1)])
            before, _ = _inversion_slopes(list(range(n)), strict, times, values)
            inclusive, uncertain = _slope_order(times, values, pivot, True)
            if uncertain:
                return _select_slope_direct(times, values, rank, sample[min(center, n - 1)])
            through, _ = _inversion_slopes(list(range(n)), inclusive, times, values)
            if before <= rank < through:
                return float(pivot)
            if rank < before:
                upper, count = strict, before - below
                break
            below, lower = through, inclusive
            count, _ = _inversion_slopes(lower, upper, times, values)
    _, candidates = _inversion_slopes(lower, upper, times, values, np.arange(count))
    rank -= below
    return float(np.partition(candidates, rank)[rank])


def _theil_sen(times, values):
    n = len(values)
    if n <= 256:
        i, j = np.triu_indices(n, 1)
        return float(np.median((values[j] - values[i]) / (times[j] - times[i])))
    pairs = n * (n - 1) // 2
    rng = np.random.default_rng(0)
    high = _select_slope(times, values, pairs // 2, rng)
    if pairs % 2:
        return high
    low = _select_slope(times, values, pairs // 2 - 1, rng)
    return float(np.mean([low, high]))


def evaluate_quality(times, durations):
    """Return named metrics and failure reasons for validated profiler samples."""
    order = np.argsort(times, kind="stable")
    times, values = np.asarray(times)[order], np.asarray(durations)[order]
    n = len(values)
    scale = _qn(values)
    # Theil-Sen drift over the full device-time span, normalized by Qn.
    slope = _theil_sen(times, values)
    # Largest gap in the central 80% of sorted samples, in units of Qn.
    gaps = np.diff(np.sort(values))
    lo, hi = max(1, int(np.ceil(.1 * n))), min(n - 1, int(np.floor(.9 * n)))
    central_gaps = gaps if hi < lo else gaps[lo - 1:hi]
    # Normalized Pettitt rank statistic: detect a shift within the acquisition.
    ranks = rankdata(values, method="average")
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
