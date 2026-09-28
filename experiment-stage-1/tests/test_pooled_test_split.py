"""Regression: pooled test fits must respect the caller's evaluation masks."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config, models


class PooledTestSplitTests(unittest.TestCase):
    def test_test_fit_uses_train_plus_validation_and_scores_test(self):
        frame = pd.DataFrame({"station_id": [1, 1, 1],
                              "date": pd.to_datetime(["2025-01-01", "2025-12-01", "2026-05-01"]),
                              "x": [1., 2., 3.]})
        captured = []
        def fake_fit(spec, factory, train, evaluation, *args):
            captured.append((train["date"].tolist(), evaluation["date"].tolist()))
            return pd.DataFrame(), []
        spec = models.RunSpec("E2_TEST", algorithm="gbr", training_strategy="global",
                              include_station_id=True, parameter_profile="e2", prefer_gpu=False)
        with patch.object(models, "assemble_feature_columns", return_value=["x"]), \
             patch.object(models, "_fit_partition", side_effect=fake_fit):
            models.run_holdout(spec, frame, [1], evaluation="test")
            models.run_holdout(spec, frame, [1], evaluation="validation")
        self.assertEqual(captured[0], (frame["date"].iloc[:2].tolist(), frame["date"].iloc[2:].tolist()))
        self.assertEqual(captured[1], (frame["date"].iloc[:1].tolist(), frame["date"].iloc[1:2].tolist()))


if __name__ == "__main__":
    unittest.main()
