"""Reward-guided expression search and a redundancy-aware factor pool."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import MarketPanel
from .expression import Expr, closing_signal
from .metrics import correlation, daily_correlations, standardize_by_day
from .policy import MaskedPPO, sample_expression


@dataclass(frozen=True)
class FactorModel:
    """A portable checkpoint containing only expressions and fitted weights."""

    expressions: tuple[Expr, ...]
    weights: tuple[float, ...]
    feature_names: tuple[str, ...]
    chunk_minutes: int = 32
    search_method: str = "random"

    def __post_init__(self) -> None:
        if not self.expressions or len(self.expressions) != len(self.weights):
            raise ValueError("the factor pool and its weights must have equal nonzero lengths")
        if self.chunk_minutes < 1:
            raise ValueError("chunk_minutes must be positive")
        if not np.isfinite(self.weights).all():
            raise ValueError("all factor weights must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "expressions": [expression.to_dict() for expression in self.expressions],
            "weights": list(self.weights),
            "feature_names": list(self.feature_names),
            "chunk_minutes": self.chunk_minutes,
            "search_method": self.search_method,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FactorModel":
        if value.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        return cls(
            expressions=tuple(Expr.from_dict(item) for item in value["expressions"]),
            weights=tuple(float(item) for item in value["weights"]),
            feature_names=tuple(value["feature_names"]),
            chunk_minutes=int(value["chunk_minutes"]),
            search_method=value["search_method"],
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FactorModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _fit_ridge(signals: list[np.ndarray], labels: np.ndarray, penalty: float = 1e-3) -> tuple[float, ...]:
    features = np.stack([standardize_by_day(item) for item in signals], axis=-1)
    target = labels.reshape(-1).astype(np.float64)
    matrix = features.reshape(-1, features.shape[-1])
    valid = np.isfinite(target) & np.isfinite(matrix).all(axis=1)
    if valid.sum() <= len(signals):
        raise ValueError("too few finite training observations for pool fitting")
    x = matrix[valid]
    y = target[valid]
    coefficients = np.linalg.solve(x.T @ x + penalty * np.eye(x.shape[1]), x.T @ y)
    return tuple(float(item) for item in coefficients)


def discover(
    panel: MarketPanel,
    train_indices: np.ndarray,
    *,
    trials: int = 48,
    capacity: int = 5,
    mode: str = "pooled",
    search: str = "random",
    seed: int = 11,
    chunk_minutes: int = 32,
    max_correlation: float = 0.85,
) -> FactorModel:
    """Fit only on the supplied training days; validation and test remain unseen."""
    indices = np.asarray(train_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) < 3 or (indices < 0).any() or (indices >= panel.days - 1).any():
        raise ValueError("train_indices must select at least three labeled days")
    if trials < 1 or capacity < 1 or mode not in {"pooled", "single"}:
        raise ValueError("invalid trial count, capacity, or discovery mode")
    if search not in {"random", "ppo"}:
        raise ValueError("search must be random or ppo")
    if not 0.0 <= max_correlation < 1.0:
        raise ValueError("max_correlation must be in [0, 1)")
    if mode == "single":
        capacity = 1

    rng = np.random.default_rng(seed)
    policy = MaskedPPO(panel.feature_names, seed=seed) if search == "ppo" else None
    targets = panel.next_day_return[indices]
    chosen: list[Expr] = []
    chosen_signals: list[np.ndarray] = []
    chosen_scores: list[float] = []
    seen: set[str] = set()
    experiences: list[tuple[object, float]] = []

    for _ in range(trials):
        expression, trace = policy.sample() if policy else (
            sample_expression(panel.feature_names, rng), None
        )
        identity = json.dumps(expression.to_dict(), sort_keys=True)
        if identity in seen:
            reward = -0.05
        else:
            seen.add(identity)
            signals = closing_signal(panel, expression, indices, chunk_minutes)
            rank_ic = daily_correlations(signals, targets, rank=True)
            valid_ic = rank_ic[np.isfinite(rank_ic)]
            strength = abs(float(valid_ic.mean())) if len(valid_ic) else 0.0
            redundancy = max(
                (abs(correlation(signals.ravel(), old.ravel())) for old in chosen_signals),
                default=0.0,
            )
            if not np.isfinite(redundancy):
                redundancy = 1.0
            reward = strength - 0.15 * redundancy if len(valid_ic) else -0.05
            if len(valid_ic) and redundancy <= max_correlation:
                if len(chosen) < capacity:
                    chosen.append(expression)
                    chosen_signals.append(signals)
                    chosen_scores.append(reward)
                else:
                    worst = int(np.argmin(chosen_scores))
                    if reward > chosen_scores[worst]:
                        chosen[worst] = expression
                        chosen_signals[worst] = signals
                        chosen_scores[worst] = reward
        if policy:
            experiences.append((trace, reward))
            if len(experiences) >= 8:
                policy.update(experiences)
                experiences.clear()

    if policy and experiences:
        policy.update(experiences)
    if not chosen:
        raise ValueError("no candidate produced a valid training IC; increase trials or panel size")

    if mode == "single":
        ic = daily_correlations(chosen_signals[0], targets, rank=True)
        direction = float(np.nanmean(ic))
        weights = (1.0 if direction >= 0.0 else -1.0,)
    else:
        try:
            weights = _fit_ridge(chosen_signals, targets)
        except ValueError:
            weights = tuple(1.0 if np.nanmean(daily_correlations(s, targets, rank=True)) >= 0 else -1.0 for s in chosen_signals)

    return FactorModel(
        expressions=tuple(chosen),
        weights=weights,
        feature_names=panel.feature_names,
        chunk_minutes=chunk_minutes,
        search_method=search,
    )
