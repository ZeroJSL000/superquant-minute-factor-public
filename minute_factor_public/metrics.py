"""Cross-sectional research metrics with explicit missing-value handling."""

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = 0.5 * (position + end - 1)
        position = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray, rank: bool = False) -> float:
    """Return a Pearson or Spearman correlation over finite paired observations."""
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3:
        return float("nan")
    x = np.asarray(left[mask], dtype=np.float64)
    y = np.asarray(right[mask], dtype=np.float64)
    if rank:
        x, y = _average_ranks(x), _average_ranks(y)
    x -= x.mean()
    y -= y.mean()
    scale = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / scale) if scale > 1e-12 else float("nan")


def standardize_by_day(signals: np.ndarray) -> np.ndarray:
    """Normalize each day's stock cross-section without looking at other days."""
    result = np.full(signals.shape, np.nan, dtype=np.float64)
    for day, row in enumerate(signals):
        valid = np.isfinite(row)
        if valid.sum() < 3:
            continue
        scale = row[valid].std()
        if scale > 1e-12:
            result[day, valid] = (row[valid] - row[valid].mean()) / scale
    return result


def daily_correlations(signals: np.ndarray, labels: np.ndarray, rank: bool = False) -> np.ndarray:
    if signals.shape != labels.shape or signals.ndim != 2:
        raise ValueError("signals and labels must share shape [day, stock]")
    return np.array([correlation(x, y, rank) for x, y in zip(signals, labels)])


def long_short_returns(signals: np.ndarray, labels: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    """Equal-weight top-minus-bottom next-day returns, before costs or trading rules."""
    if not 0.0 < fraction < 0.5:
        raise ValueError("fraction must be between zero and one half")
    result = np.full(len(signals), np.nan, dtype=np.float64)
    for day, (score, target) in enumerate(zip(signals, labels)):
        valid = np.isfinite(score) & np.isfinite(target)
        count = int(valid.sum())
        if count < 4:
            continue
        tail = max(1, int(count * fraction))
        order = np.argsort(score[valid], kind="mergesort")
        returns = target[valid]
        result[day] = returns[order[-tail:]].mean() - returns[order[:tail]].mean()
    return result


def summarize(signals: np.ndarray, labels: np.ndarray) -> dict[str, float | int | None]:
    ic = daily_correlations(signals, labels)
    rank_ic = daily_correlations(signals, labels, rank=True)
    spread = long_short_returns(signals, labels)

    def finite_mean(values: np.ndarray) -> float | None:
        valid = values[np.isfinite(values)]
        return float(valid.mean()) if len(valid) else None

    return {
        "days": int(len(signals)),
        "valid_ic_days": int(np.isfinite(ic).sum()),
        "mean_ic": finite_mean(ic),
        "mean_rank_ic": finite_mean(rank_ic),
        "mean_long_short_return": finite_mean(spread),
    }
