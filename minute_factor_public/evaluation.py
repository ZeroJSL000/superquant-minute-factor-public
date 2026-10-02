"""Evaluation that can reload a saved factor pool independently of discovery."""

import numpy as np

from .data import MarketPanel
from .discovery import FactorModel
from .expression import closing_signal
from .metrics import standardize_by_day, summarize


def model_scores(panel: MarketPanel, model: FactorModel, day_indices: np.ndarray) -> np.ndarray:
    if model.feature_names != panel.feature_names:
        raise ValueError("model and panel feature schemas differ")
    if len(model.expressions) == 0:
        raise ValueError("the saved model has no factors")
    output = np.zeros((len(day_indices), panel.stocks), dtype=np.float64)
    has_signal = np.zeros_like(output, dtype=bool)
    for expression, weight in zip(model.expressions, model.weights):
        raw = closing_signal(panel, expression, day_indices, model.chunk_minutes)
        normalized = standardize_by_day(raw)
        valid = np.isfinite(normalized)
        output += float(weight) * np.where(valid, normalized, 0.0)
        has_signal |= valid
    output[~has_signal] = np.nan
    return output


def evaluate_model(
    panel: MarketPanel, model: FactorModel, day_indices: np.ndarray
) -> dict[str, float | int | None]:
    """Score a saved model against labels on an explicitly selected time segment."""
    indices = np.asarray(day_indices, dtype=np.int64)
    if indices.ndim != 1 or (indices < 0).any() or (indices >= panel.days).any():
        raise ValueError("day_indices must be a valid one-dimensional selection")
    return summarize(model_scores(panel, model, indices), panel.next_day_return[indices])
