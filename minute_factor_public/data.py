"""Synthetic market panels with an explicit prediction-time boundary."""

from dataclasses import dataclass

import numpy as np

FEATURES = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class MarketPanel:
    """CPU-resident features [day, minute, stock, feature] and next-day labels."""

    features: np.ndarray
    next_day_return: np.ndarray
    feature_names: tuple[str, ...] = FEATURES

    def __post_init__(self) -> None:
        if self.features.ndim != 4:
            raise ValueError("features must have shape [day, minute, stock, feature]")
        days, minutes, stocks, channels = self.features.shape
        if days < 3 or minutes < 3 or stocks < 4:
            raise ValueError("the panel needs at least 3 days, 3 minutes, and 4 stocks")
        if channels != len(self.feature_names) or len(set(self.feature_names)) != channels:
            raise ValueError("feature_names must uniquely identify every feature channel")
        if self.next_day_return.shape != (days, stocks):
            raise ValueError("next_day_return must have shape [day, stock]")

    @property
    def days(self) -> int:
        return self.features.shape[0]

    @property
    def stocks(self) -> int:
        return self.features.shape[2]

    def day(self, index: int) -> np.ndarray:
        """Return one day only; expression windows never span adjacent days."""
        return self.features[index]


def synthetic_market(
    days: int = 84, minutes: int = 32, stocks: int = 24, seed: int = 7
) -> MarketPanel:
    """Generate fictional OHLCV values; no source market observations are used."""
    if days < 3 or minutes < 3 or stocks < 4:
        raise ValueError("days, minutes, or stocks are too small")
    rng = np.random.default_rng(seed)
    values = np.empty((days, minutes, stocks, len(FEATURES)), dtype=np.float32)
    previous_close = rng.uniform(20.0, 120.0, size=stocks)
    latent_drift = np.zeros(stocks, dtype=np.float64)

    for day in range(days):
        latent_drift = 0.4 * latent_drift + rng.normal(0.0, 0.007, size=stocks)
        for minute in range(minutes):
            open_price = previous_close.copy()
            noise = rng.normal(0.0, 0.003, size=stocks)
            close_price = open_price * np.exp(latent_drift / minutes + noise)
            spread = np.abs(rng.normal(0.001, 0.0005, size=stocks))
            high = np.maximum(open_price, close_price) * (1.0 + spread)
            low = np.minimum(open_price, close_price) * (1.0 - spread)
            volume = rng.lognormal(mean=8.0, sigma=0.4, size=stocks)
            values[day, minute] = np.stack(
                (open_price, high, low, close_price, volume), axis=-1
            )
            previous_close = close_price

    daily_close = values[:, -1, :, FEATURES.index("close")].astype(np.float64)
    labels = np.full((days, stocks), np.nan, dtype=np.float32)
    labels[:-1] = (daily_close[1:] / daily_close[:-1] - 1.0).astype(np.float32)
    return MarketPanel(values, labels)
