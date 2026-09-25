"""Regression coverage for common-support comparisons and fitted model metadata."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import artifacts, metrics


class ReportingRegressions(unittest.TestCase):
    def setUp(self):
        self.predictions = pd.DataFrame({
            "station_id": [1, 1, 1, 1, 2],
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03",
                                    "2025-01-04", "2025-01-01"]),
            "horizon": [1] * 5, "y_true": [10., 20., 30., 40., 50.],
            "y_pred": [12., 21., np.inf, 41., np.nan],
        })
        self.source = self.predictions[["station_id", "date"]].copy()
        self.source["pm25"] = [8., np.nan, 31., np.inf, 51.]

    def test_comparison_has_identical_finite_support_and_preserves_input(self):
        original = self.predictions.copy(deep=True)
        paired = metrics.paired_persistence_reports(self.predictions, self.source.iloc[::-1])
        self.assertEqual(paired["n"], 1)
        for name in ("model", "persistence"):
            self.assertEqual(paired[name]["overall"]["micro"]["n"], 1)
            self.assertEqual(paired[name]["overall"]["micro"]["rmse"], 2.)
        pd.testing.assert_frame_equal(self.predictions, original)

    def test_empty_common_support_is_reported_without_error(self):
        self.source["pm25"] = np.nan
        paired = metrics.paired_persistence_reports(self.predictions, self.source)
        self.assertEqual(paired["n"], 0)
        self.assertEqual(paired["model"]["overall"]["micro"]["n"], 0)
        self.assertTrue(np.isnan(paired["persistence"]["overall"]["micro"]["rmse"]))

    def test_empty_prediction_frame_has_zero_support(self):
        empty = pd.DataFrame(columns=self.predictions.columns)
        paired = metrics.paired_persistence_reports(empty, self.source)
        self.assertEqual(paired["n"], 0)

    def test_horizons_share_current_pm_but_exclude_missing_targets(self):
        predictions = pd.concat([self.predictions.iloc[:1], self.predictions.iloc[:1]],
                                ignore_index=True)
        predictions.loc[1, ["horizon", "y_true", "y_pred"]] = [2, 16., 10.]
        missing_target = predictions.iloc[:1].assign(horizon=3, y_true=np.nan)
        paired = metrics.paired_persistence_reports(
            pd.concat([predictions, missing_target]), self.source)
        self.assertEqual(paired["n"], 2)
        self.assertAlmostEqual(paired["model"]["overall"]["micro"]["rmse"], np.sqrt(20))
        self.assertAlmostEqual(paired["persistence"]["overall"]["micro"]["rmse"], np.sqrt(34))

    def test_duplicate_keys_cannot_multiply_comparison_rows(self):
        for predictions, source in (
            (pd.concat([self.predictions, self.predictions.iloc[:1]]), self.source),
            (self.predictions, pd.concat([self.source, self.source.iloc[:1]])),
        ):
            with self.subTest(prediction_rows=len(predictions), source_rows=len(source)):
                with self.assertRaises(ValueError):
                    metrics.paired_persistence_reports(predictions, source)

    def _saved_config(self, root, estimator=None, prefer_gpu=True, supplied=None):
        records = ([] if estimator is None else [{"role": "main", "group": "global",
                    "horizon": 1, "estimator": estimator}])
        with patch.object(artifacts, "detect_device", return_value={"device": "cuda"}) as detect, \
             patch.object(artifacts, "dataset_manifest", return_value={}), \
             patch.object(artifacts, "save_models"):
            artifacts.save_run(root, "reporting", supplied or {}, self.source, [1, 2],
                               pd.DataFrame(), metrics.build_reports(self.predictions.iloc[:1]),
                               self.predictions, [], model_records=records, prefer_gpu=prefer_gpu)
            detect.assert_called_once_with(prefer_gpu=prefer_gpu)
        return json.loads((root / "config.json").read_text())

    def test_cpu_preference_and_gbr_actual_device_override_cuda_availability(self):
        for preference in (True, False):
            with tempfile.TemporaryDirectory() as folder:
                saved = self._saved_config(Path(folder), GradientBoostingRegressor(), preference)
                self.assertEqual(saved["device"], "cpu")
                self.assertEqual(saved["actual_device"], "cpu")
                self.assertEqual(saved["requested_device"], "gpu" if preference else "cpu")

    def test_lightgbm_cpu_fallback_is_reported_as_cpu(self):
        from lightgbm import LGBMRegressor
        with tempfile.TemporaryDirectory() as folder:
            saved = self._saved_config(Path(folder), LGBMRegressor(device_type="cpu"))
            self.assertEqual(saved["actual_device"], "cpu")

    def test_empty_records_preserve_supplied_device_and_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            saved = self._saved_config(Path(folder), supplied={"device": "cpu",
                                       "evaluation_splits": ["cv", "validation", "test"]})
            self.assertEqual(saved["actual_device"], "cpu")
            self.assertEqual(saved["evaluation_splits"], ["cv", "validation", "test"])

    def test_xgb_refit_saves_selection_without_evaluation_history(self):
        estimator = XGBRegressor(n_estimators=2, device="cpu", n_jobs=1).fit(
            np.arange(12).reshape(-1, 1), np.arange(12))
        estimator.selection_metadata_ = {"policy": "internal", "selected_n_estimators": 2}
        # Requested parameters may disagree with the already fitted booster.
        estimator.device = "cuda"
        record = {"role": "main", "group": "global", "horizon": 1, "estimator": estimator}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifacts.save_models(root, [record], "validation", ["x"])
            item = json.loads((root / "model/validation/manifest.json").read_text())["models"][0]
            self.assertEqual(item["actual_device"], "cpu")
            self.assertEqual(item["selection_metadata"], estimator.selection_metadata_)
            self.assertNotIn("training_history_file", item)


if __name__ == "__main__":
    unittest.main()
