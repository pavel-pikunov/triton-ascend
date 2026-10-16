"""Calibrated retry heuristics on all samples, with linear auxiliary memory."""

from fractions import Fraction
from math import gcd

import numpy as np
from scipy.stats import rankdata
from scipy.stats._stats import _kendall_dis

THRESHOLDS = {
    "mean_trimmed_relative_deviation": 0.05,
    "central_gap_ratio": 0.30,
    "dip_statistic": 0.05,
    "normalized_drift": 0.45,
    "normalized_change_point": 0.42,
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


def _integer_coordinates(values):
    """Represent binary floats on a shared integer grid, without rounding."""
    _, exponents = np.frexp(values)
    exponent = int(np.min(exponents[values != 0], initial=1024))
    scaled = np.ldexp(values, 53 - exponent)
    if exponent > 53 or np.max(np.abs(scaled)) >= 2**62:
        ratios = [float(value).as_integer_ratio() for value in values]
        denominator = max(d for _, d in ratios)
        integers = np.array([v * (denominator // d) for v, d in ratios], dtype=object)
        return integers - integers[0], denominator
    integers = scaled.astype(np.int64)
    return integers - integers[0], 1 << (53 - exponent)


def _exact_slope_counts(t, y, pivot, coordinates=None):
    """Resolve collinear groups at both round-to-even boundaries of a slope."""
    if coordinates is None:
        ty, dy = _integer_coordinates(y)
        tt, dt = _integer_coordinates(t)
    else:
        (ty, dy), (tt, dt) = coordinates
    if np.ptp(ty) > 2**53 or np.ptp(tt) > 2**53:
        return None
    point = Fraction(float(pivot))
    even = (np.asarray(pivot).view(np.uint64).item() & 1) == 0
    counts, previous, following = [], -np.inf, np.inf
    for neighbor, inclusive in ((-np.inf, not even), (np.inf, even)):
        cut = (point + Fraction(float(np.nextafter(pivot, neighbor)))) / 2
        a, b = cut.denominator * dt, cut.numerator * dy
        divisor = gcd(a, b)
        a, b = a // divisor, b // divisor
        left, right = ty.astype(float) * a, tt.astype(float) * b
        error = 8 * np.finfo(float).eps * (np.abs(left) + np.abs(right))
        if np.max(np.abs(left - right) + error) < 2**63:
            # Modular products recover the exact signed difference once it fits.
            residual = (ty.astype(np.uint64) * np.uint64(a % 2**64) - tt.astype(np.uint64) * np.uint64(b % 2**64)).view(
                np.int64)
        else:
            residual = ty.astype(object) * a - tt.astype(object) * b
        order = np.lexsort((-tt if inclusive else tt, residual))
        positions = np.empty(len(y), dtype=np.intp)
        positions[order] = np.arange(1, len(y) + 1)
        counts.append(int(_kendall_dis(np.arange(len(y), dtype=np.intp), positions)))
        slopes = np.diff(y[order]) / np.diff(t[order])
        previous = max(previous, np.max(slopes[slopes < pivot], initial=-np.inf))
        following = min(following, np.min(slopes[slopes > pivot], initial=np.inf))
    return counts[0], counts[1], previous, following


def _distance_counts(x, pivot, inclusive):
    rows = np.arange(len(x) - 1)
    boundary = np.maximum(rows + 1, np.searchsorted(x, x[:-1] + pivot, side='right' if inclusive else 'left'))
    # Correct rounded sums against the original rounded pair differences.
    for direction in (-1, 1):
        while True:
            index = np.clip(boundary - (direction < 0), 0, len(x) - 1)
            difference = x[index] - x[:-1]
            inside = difference <= pivot if inclusive else difference < pivot
            move = (boundary > rows + 1) & ~inside if direction < 0 else (boundary < len(x)) & inside
            if not np.any(move):
                break
            value = x[index[move]]
            boundary[move] = np.searchsorted(x, value, side='left' if direction < 0 else 'right')
            boundary = np.maximum(boundary, rows + 1)
    return boundary


def _select_pair_statistic(counts, low, high, rank, pairs, sample, median=False):
    """Sample pivots only; certify each contraction and selected full-pair rank."""
    fraction = rank / pairs
    trials = list(np.quantile(sample, [fraction, max(0, fraction - .1), min(1, fraction + .1)]))
    low_count, high_count = 0, pairs
    step = 0
    while True:
        step += 1
        fraction = (rank - low_count) / (high_count - low_count) if step % 8 else .5
        pivot = trials.pop(0) if trials else low + (high - low) * fraction
        if not low <= pivot <= high:
            continue
        below, through, previous, following = counts(pivot)
        if below <= rank < through:
            return float(np.mean([previous, pivot]) if median and pairs % 2 == 0 and below == rank else pivot)
        if rank < below:
            high, high_count = float(previous), below
        else:
            low, low_count = float(following), through
        if low == high:
            below, through, previous, following = counts(low)
            return float(np.mean([previous, low]) if median and pairs % 2 == 0 and below == rank else low)


def _qn(values):
    """Exact Croux-Rousseeuw distance quantile, without a pair matrix."""
    x = np.sort(values)
    n = len(x)
    rows = np.arange(n - 1)
    rank = (n // 2 + 1) * (n // 2) // 2 - 1

    if n <= 256:
        i, j = np.triu_indices(n, 1)
        return 2.2219 * float(np.partition(x[j] - x[i], rank)[rank])

    def count(pivot):
        left, right = _distance_counts(x, pivot, False), _distance_counts(x, pivot, True)
        previous = np.max(x[np.maximum(rows + 1, left - 1)] - x[:-1], where=left > rows + 1, initial=-np.inf)
        following = np.min(x[np.minimum(right, n - 1)] - x[:-1], where=right < n, initial=np.inf)
        return np.sum(left - rows - 1), np.sum(right - rows - 1), previous, following

    rng = np.random.default_rng(0)
    i, j = np.sort(rng.integers(n, size=(2, n)), axis=0)
    sample = x[j[i != j]] - x[i[i != j]]
    return 2.2219 * _select_pair_statistic(count, 0., float(x[-1] - x[0]), rank, n * (n - 1) // 2, sample)


def _slope_counts(t, y, pivot, coordinates=None):
    """Count pair slopes by dual-line inversions; correct ambiguous float ties."""
    n = len(y)
    residual = (y - y[0]) - pivot * (t - t[0])
    order = np.argsort(residual, kind="stable" if pivot == 0 else "quicksort")
    positions = np.empty(n, dtype=np.intp)
    positions[order] = np.arange(1, n + 1)
    below = int(_kendall_dis(np.arange(n, dtype=np.intp), positions))
    adjacent = (y[order[1:]] - y[order[:-1]]) / (t[order[1:]] - t[order[:-1]])
    previous = np.max(adjacent[adjacent < pivot], initial=-np.inf)
    following = np.min(adjacent[adjacent > pivot], initial=np.inf)
    if pivot == 0:
        _, counts = np.unique(y, return_counts=True)
        opposite = np.lexsort((-t, y))
        next_slopes = np.diff(y[opposite]) / np.diff(t[opposite])
        previous = max(previous, np.max(next_slopes[next_slopes < pivot], initial=-np.inf))
        following = min(following, np.min(next_slopes[next_slopes > pivot], initial=np.inf))
        return below, below + int(np.sum(counts * (counts - 1) // 2)), previous, following
    tolerance = 16 * np.finfo(float).eps * np.max(np.abs(y - y[0]) + abs(pivot) * np.abs(t - t[0]))
    splits = np.r_[0, np.flatnonzero(np.diff(residual[order]) > tolerance) + 1, n]
    if np.max(np.diff(splits)) > 8:
        exact = _exact_slope_counts(t, y, pivot, coordinates)
        if exact is not None:
            return exact
    equal = 0
    multiple = np.flatnonzero(np.diff(splits) > 1)
    if not len(multiple):
        return below, below, previous, following
    before_order, after_order = order.copy(), order.copy()
    for start, stop in zip(splits[multiple], splits[multiple + 1]):
        group = order[start:stop]
        before_order[start:stop] = np.sort(group)
        after_order[start:stop] = np.sort(group)[::-1]
        below -= int(_kendall_dis(np.arange(len(group), dtype=np.intp), group + 1))
        # Fixed-width blocks also keep the rare inexact-subtraction path linear in memory.
        for row in range(0, len(group) - 1, 16):
            a, b = group[row:row + 16, None], group[None, row + 1:]
            valid = np.arange(row + 1, len(group))[None, :] > np.arange(row, min(row + 16, len(group)))[:, None]
            slopes = y[b] - y[a]
            np.divide(slopes, t[b] - t[a], out=slopes, where=valid)
            less = (slopes < pivot) & valid
            count = np.count_nonzero(less)
            below += count
            equal += np.count_nonzero((slopes <= pivot) & valid) - count
            if previous < np.nextafter(pivot, -np.inf):
                previous = max(previous, np.max(np.where(less, slopes, -np.inf)))
            if following > np.nextafter(pivot, np.inf):
                following = min(following, np.min(np.where((slopes > pivot) & valid, slopes, np.inf)))
    for side_order in (before_order, after_order):
        slopes = np.diff(y[side_order]) / np.diff(t[side_order])
        previous = max(previous, np.max(slopes[slopes < pivot], initial=-np.inf))
        following = min(following, np.min(slopes[slopes > pivot], initial=np.inf))
    return below, below + equal, previous, following


def _theil_sen(t, y):
    """Select the full pairwise median in linear space, with certified counts.

    Sampling only suggests pivots. Inversion counts certify every bound and
    the returned rank, including both central ranks for an even pair count.
    """
    n = len(y)
    if n <= 256:
        i, j = np.triu_indices(n, 1)
        return float(np.median((y[j] - y[i]) / (t[j] - t[i])))
    if np.ptp(y) == 0:
        return 0.
    adjacent = np.diff(y) / np.diff(t)
    low, high = float(np.min(adjacent)), float(np.max(adjacent))
    rank, pairs = (n * (n - 1) // 2) // 2, n * (n - 1) // 2
    rng = np.random.default_rng(0)
    i, j = np.sort(rng.integers(n, size=(2, n)), axis=0)
    valid = i != j
    sample = (y[j[valid]] - y[i[valid]]) / (t[j[valid]] - t[i[valid]])
    coordinates = (_integer_coordinates(y), _integer_coordinates(t))
    return _select_pair_statistic(lambda pivot: _slope_counts(t, y, pivot, coordinates), low, high, rank, pairs, sample,
                                  median=True)


def evaluate_quality(times, durations, *, check_all=False):
    """Return retry metrics; check_all also evaluates checks after the first failure."""
    order = np.argsort(times, kind="stable")
    times, values = np.asarray(times)[order], np.asarray(durations)[order]
    n = len(values)
    scale = _qn(values)
    gaps = np.diff(np.sort(values))
    lo, hi = max(1, int(np.ceil(.1 * n))), min(n - 1, int(np.floor(.9 * n)))
    central_gaps = gaps if hi < lo else gaps[lo - 1:hi]
    ranks = rankdata(values, method="average")
    checks = (
        ("mean_trimmed_relative_deviation", lambda: abs(_ratio(np.mean(values), central_50_mean(values)) - 1)),
        ("central_gap_ratio", lambda: _ratio(np.max(central_gaps), scale)),
        ("normalized_change_point",
         lambda: float(np.max(np.abs(2 * np.cumsum(ranks)[:-1] - np.arange(1, n) * (n + 1))) / (n * n // 4))),
        ("dip_statistic", lambda: _dip(values)),
        ("normalized_drift", lambda: _ratio(abs(_theil_sen(times, values)) * (times[-1] - times[0]), scale)),
    )
    metrics = {}
    for name, calculate in checks:
        value = metrics[name] = calculate()
        if not np.isfinite(value) or value > THRESHOLDS[name]:
            if not check_all:
                break
    metrics = {name: metrics[name] for name in THRESHOLDS if name in metrics}
    failures = [
        f"{name}={value:.8g} (limit={THRESHOLDS[name]:g})" for name, value in metrics.items()
        if not np.isfinite(value) or value > THRESHOLDS[name]
    ]
    return metrics, failures
