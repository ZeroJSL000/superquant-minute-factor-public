"""Checks for causal windows, held-out isolation, and portable checkpoints."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from minute_factor_public.data import MarketPanel, synthetic_market
from minute_factor_public.discovery import FactorModel, discover
from minute_factor_public.evaluation import evaluate_model
from minute_factor_public.expression import Expr, evaluate_day
from minute_factor_public.metrics import correlation
from minute_factor_public.splits import chronological_split, walk_forward_splits


class ResearchBoundaryTests(unittest.TestCase):
    def test_next_day_label_uses_only_later_close(self) -> None:
        panel = synthetic_market(days=12, minutes=8, stocks=6, seed=4)
        close = panel.features[:, -1, :, panel.feature_names.index("close")].astype(np.float64)
        expected = close[1:] / close[:-1] - 1.0
        np.testing.assert_allclose(panel.next_day_return[:-1], expected, rtol=1e-5, atol=1e-7)
        self.assertTrue(np.isnan(panel.next_day_return[-1]).all())

    def test_rolling_operator_cannot_read_future_or_previous_day(self) -> None:
        panel = synthetic_market(days=12, minutes=9, stocks=6, seed=4)
        expression = Expr("mean", (Expr("feature", name="close"),), window=3)
        original = evaluate_day(expression, panel.day(1), panel.feature_names, chunk_minutes=2)
        changed = panel.day(1).copy()
        changed[6:, :, panel.feature_names.index("close")] *= 10.0
        modified = evaluate_day(expression, changed, panel.feature_names, chunk_minutes=2)
        np.testing.assert_allclose(original[:6], modified[:6], equal_nan=True)
        self.assertTrue(np.isnan(original[:2]).all())
        self.assertTrue(np.isnan(evaluate_day(expression, panel.day(2), panel.feature_names)[:2]).all())

    def test_rank_ic_handles_ties_and_missing_values(self) -> None:
        x = np.array([1.0, 1.0, 2.0, 3.0, np.nan])
        y = np.array([2.0, 1.0, 4.0, 5.0, 100.0])
        self.assertTrue(np.isfinite(correlation(x, y, rank=True)))
        self.assertTrue(np.isnan(correlation(np.ones(5), np.arange(5), rank=True)))

    def test_saved_model_and_training_ignore_held_out_labels(self) -> None:
        panel = synthetic_market(days=30, minutes=10, stocks=8, seed=2)
        train, validation, test = chronological_split(panel.days)
        first = discover(panel, train, trials=12, capacity=2, seed=3, chunk_minutes=4)
        changed_labels = panel.next_day_return.copy()
        changed_labels[validation] = 100.0
        changed_labels[test] = -100.0
        changed = MarketPanel(panel.features, changed_labels, panel.feature_names)
        second = discover(changed, train, trials=12, capacity=2, seed=3, chunk_minutes=4)
        self.assertEqual(first.to_dict(), second.to_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            first.save(path)
            restored = FactorModel.load(path)
            self.assertEqual(first.to_dict(), restored.to_dict())
            self.assertEqual(evaluate_model(panel, first, test), evaluate_model(panel, restored, test))

    def test_walk_forward_periods_do_not_overlap(self) -> None:
        for train, validation, test in walk_forward_splits(days=85, year_days=12):
            self.assertLess(train[-1], validation[0])
            self.assertLess(validation[-1], test[0])
            self.assertLess(test[-1], 84)


if __name__ == "__main__":
    unittest.main()
