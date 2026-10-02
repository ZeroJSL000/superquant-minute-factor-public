"""Public, synthetic-data minute-factor research example."""

from .data import MarketPanel, synthetic_market
from .discovery import FactorModel, discover
from .evaluation import evaluate_model

__all__ = ["MarketPanel", "synthetic_market", "FactorModel", "discover", "evaluate_model"]
