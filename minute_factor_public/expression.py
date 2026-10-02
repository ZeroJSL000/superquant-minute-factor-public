"""A small expression tree with causal, same-day minute operators."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import MarketPanel

UNARY = {"abs", "log1p_abs"}
BINARY = {"add", "sub", "mul", "div"}
ROLLING = {"mean", "std", "delta"}
PAIR_ROLLING = {"corr"}


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    name: str | None = None
    window: int | None = None

    def __post_init__(self) -> None:
        arity = 0 if self.op == "feature" else 1 if self.op in UNARY | ROLLING else 2
        if self.op not in {"feature"} | UNARY | BINARY | ROLLING | PAIR_ROLLING:
            raise ValueError(f"unknown expression operator: {self.op}")
        if len(self.args) != arity:
            raise ValueError(f"{self.op} expects {arity} operands")
        if self.op == "feature" and not self.name:
            raise ValueError("a feature expression needs a name")
        if self.op != "feature" and self.name is not None:
            raise ValueError("only feature expressions may have a name")
        if self.op in ROLLING | PAIR_ROLLING and (self.window is None or self.window < 2):
            raise ValueError("rolling operators need a window of at least two minutes")
        if self.op not in ROLLING | PAIR_ROLLING and self.window is not None:
            raise ValueError("this operator does not accept a window")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"op": self.op}
        if self.name is not None:
            result["name"] = self.name
        if self.window is not None:
            result["window"] = self.window
        if self.args:
            result["args"] = [arg.to_dict() for arg in self.args]
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Expr":
        return cls(
            op=value["op"],
            args=tuple(cls.from_dict(arg) for arg in value.get("args", [])),
            name=value.get("name"),
            window=value.get("window"),
        )


def _rolling(values: np.ndarray, window: int, op: str, chunk_minutes: int) -> np.ndarray:
    minutes, stocks = values.shape
    result = np.full((minutes, stocks), np.nan, dtype=np.float64)
    for start in range(window - 1, minutes, chunk_minutes):
        for minute in range(start, min(start + chunk_minutes, minutes)):
            history = values[minute - window + 1 : minute + 1]
            valid = np.isfinite(history).all(axis=0)
            if op == "mean":
                result[minute, valid] = history[:, valid].mean(axis=0)
            elif op == "std":
                result[minute, valid] = history[:, valid].std(axis=0)
    return result


def _correlation(left: np.ndarray, right: np.ndarray, window: int, chunk_minutes: int) -> np.ndarray:
    minutes, stocks = left.shape
    result = np.full((minutes, stocks), np.nan, dtype=np.float64)
    for start in range(window - 1, minutes, chunk_minutes):
        for minute in range(start, min(start + chunk_minutes, minutes)):
            x = left[minute - window + 1 : minute + 1]
            y = right[minute - window + 1 : minute + 1]
            valid = np.isfinite(x).all(axis=0) & np.isfinite(y).all(axis=0)
            if not valid.any():
                continue
            xc = x[:, valid] - x[:, valid].mean(axis=0)
            yc = y[:, valid] - y[:, valid].mean(axis=0)
            denominator = np.sqrt((xc * xc).sum(axis=0) * (yc * yc).sum(axis=0))
            correlations = np.divide(
                (xc * yc).sum(axis=0),
                denominator,
                out=np.full(valid.sum(), np.nan),
                where=denominator > 1e-12,
            )
            result[minute, valid] = correlations
    return result


def evaluate_day(
    expression: Expr, day_features: np.ndarray, feature_names: tuple[str, ...], chunk_minutes: int = 32
) -> np.ndarray:
    """Evaluate one expression on one day without reading a future minute or day."""
    if chunk_minutes < 1:
        raise ValueError("chunk_minutes must be positive")
    if day_features.ndim != 3 or day_features.shape[2] != len(feature_names):
        raise ValueError("day_features must have shape [minute, stock, feature]")
    cache: dict[Expr, np.ndarray] = {}

    def visit(node: Expr) -> np.ndarray:
        if node in cache:
            return cache[node]
        if node.op == "feature":
            if node.name not in feature_names:
                raise ValueError(f"feature is missing from panel: {node.name}")
            result = day_features[:, :, feature_names.index(node.name)].astype(np.float64)
        else:
            arguments = tuple(visit(arg) for arg in node.args)
            if node.op == "abs":
                result = np.abs(arguments[0])
            elif node.op == "log1p_abs":
                result = np.log1p(np.abs(arguments[0]))
            elif node.op == "add":
                result = arguments[0] + arguments[1]
            elif node.op == "sub":
                result = arguments[0] - arguments[1]
            elif node.op == "mul":
                result = arguments[0] * arguments[1]
            elif node.op == "div":
                result = np.divide(
                    arguments[0], arguments[1],
                    out=np.full_like(arguments[0], np.nan),
                    where=np.abs(arguments[1]) > 1e-12,
                )
            elif node.op in {"mean", "std"}:
                result = _rolling(arguments[0], node.window, node.op, chunk_minutes)
            elif node.op == "delta":
                result = np.full_like(arguments[0], np.nan)
                result[node.window :] = arguments[0][node.window :] - arguments[0][: -node.window]
            elif node.op == "corr":
                result = _correlation(arguments[0], arguments[1], node.window, chunk_minutes)
            else:
                raise AssertionError("an unvalidated operator reached evaluation")
        cache[node] = result
        return result

    return visit(expression)


def closing_signal(
    panel: MarketPanel, expression: Expr, day_indices: np.ndarray, chunk_minutes: int = 32
) -> np.ndarray:
    """Take the final minute of each day's expression as a daily cross-section."""
    signals = np.full((len(day_indices), panel.stocks), np.nan, dtype=np.float64)
    for output_index, day_index in enumerate(day_indices):
        minute_values = evaluate_day(expression, panel.day(int(day_index)), panel.feature_names, chunk_minutes)
        signals[output_index] = minute_values[-1]
    return signals
