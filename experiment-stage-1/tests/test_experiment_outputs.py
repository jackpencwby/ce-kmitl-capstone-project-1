"""Exercise saved models and separate test reporting for the shared runner."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import artifacts, config, features, models


_original_xgb_params_for_horizon = config.xgb_params_for_horizon


def _fast_xgb_params(horizon, n_estimators):
    return {**_original_xgb_params_for_horizon(horizon), "n_estimators": n_estimators}


class ExperimentOutputsTests(unittest.TestCase):
    def test_xgb_models_use_each_horizons_selected_parameters(self):
        for horizon in config.FORECAST_HORIZONS:
            expected = config.xgb_params_for_horizon(horizon)
            for factory in (models.make_xgb, models.make_baseline_xgb):
                with self.subTest(horizon=horizon, factory=factory.__name__):
                    actual = factory(prefer_gpu=False, horizon=horizon).get_params()
                    for name, value in expected.items():
                        self.assertEqual(actual[name], value)
                    self.assertEqual(actual["device"], "cpu")

    def test_shared_e1_residual_fit_and_test_reuse(self):
        dates = pd.date_range("2025-01-01", periods=300)
        frame = pd.DataFrame([
            {"station_id": sid, "date": date, "x": float(i % 13),
             "target_t1": float(sid + np.sin(i / 15))}
            for sid in (1, 2) for i, date in enumerate(dates)
        ])
        spec = models.RunSpec("E1_RESIDUAL", training_strategy="global_local_tree",
                              include_station_id=True, baseline_xgb=True,
                              prefer_gpu=False)
        with patch.object(config, "VALIDATION_START", "2025-09-01"), \
             patch.object(config, "TEST_START", "2025-10-01"), \
             patch.object(config, "FORECAST_HORIZONS", (1,)), \
             patch.object(config, "xgb_params_for_horizon",
                          side_effect=lambda h: _fast_xgb_params(h, 10)), \
             patch.object(config, "BASELINE_EARLY_STOPPING_ROUNDS", 2), \
             patch.object(config.MLP_PARAMS, "min_station_rows", 2), \
             patch.object(config, "baseline_feature_list", return_value=["x"]):
            fitted = models.run_holdout(spec, frame, [1, 2])
            test = models.predict_e1_test_from_validation_models(
                spec, frame, [1, 2], fitted["model_records"])
        self.assertTrue(any(r["role"] == "local_residual"
                            for r in fitted["model_records"]))
        self.assertEqual(set(test["reports"]["by_station"].station_id), {1, 2})

    def test_oof_xgb_uses_training_window_for_internal_early_stopping(self):
        dates = pd.date_range("2025-01-01", periods=200)
        X = pd.DataFrame({"x": np.arange(200, dtype=float)})
        y = pd.Series(np.sin(np.arange(200) / 10))
        future = pd.DataFrame({"x": [201.0]})
        with patch.object(config, "xgb_params_for_horizon",
                          side_effect=lambda h: _fast_xgb_params(h, 12)), \
             patch.object(config, "BASELINE_EARLY_STOPPING_ROUNDS", 3):
            prediction, fitted = models._fit_predict_estimator(
                models.make_baseline_xgb, False, X, y, future,
                train_dates=pd.Series(dates), horizon=1)
        self.assertEqual(len(prediction), 1)
        self.assertIn("validation_1", fitted.evals_result())

    def test_e1_test_reuses_global_regional_and_residual_models(self):
        import torch

        class Constant:
            def __init__(self, value):
                self.value = value

            def predict(self, features):
                return np.full(len(features), self.value)

        frame = pd.DataFrame([
            {"station_id": sid, "date": date, "x": (np.nan if sid == 1 else 2.0),
             "region_id": region, "target_t1": 5.0}
            for sid, region in ((1, "North"), (2, "South"))
            for date in pd.to_datetime(["2025-01-01", "2025-01-03"])
        ])
        mlp = torch.nn.Linear(3, 1)
        with torch.no_grad():
            mlp.weight.zero_()
            mlp.bias.fill_(2.0)
        mlp.eval()
        cases = [
            (models.RunSpec("g", training_strategy="global", include_station_id=True,
                            baseline_xgb=True),
             [{"role": "main", "group": "global", "horizon": 1,
               "estimator": Constant(3.0)}], [3.0, 3.0]),
            (models.RunSpec("r", training_strategy="regional", include_station_id=True,
                            include_region_id=True, baseline_xgb=True),
             [{"role": "main", "group": group, "horizon": 1,
               "estimator": Constant(value)}
              for group, value in (("North", 3.0), ("South", 4.0))], [3.0, 4.0]),
            (models.RunSpec("t", training_strategy="global_local_tree",
                            include_station_id=True, baseline_xgb=True),
             [{"role": "main", "group": "global", "horizon": 1,
               "estimator": Constant(3.0)},
              {"role": "local_residual", "group": "1", "horizon": 1,
               "estimator": Constant(1.0)}], [4.0, 3.0]),
            (models.RunSpec("m", training_strategy="global_local_mlp",
                            include_station_id=True, baseline_xgb=True),
             [{"role": "main", "group": "global", "horizon": 1,
               "estimator": Constant(3.0)},
              {"role": "local_residual_mlp", "group": "1", "horizon": 1,
               "estimator": mlp, "feature_columns": ["x", "station_id_code", "global_pred"],
               "mean": np.zeros(3), "std": np.ones(3)}], [5.0, 3.0]),
        ]
        with patch.object(config, "VALIDATION_START", "2025-01-02"), \
             patch.object(config, "TEST_START", "2025-01-03"), \
             patch.object(config, "FORECAST_HORIZONS", (1,)), \
             patch.object(config, "baseline_feature_list", return_value=["x"]):
            for spec, records, expected in cases:
                with self.subTest(strategy=spec.training_strategy):
                    result = models.predict_e1_test_from_validation_models(
                        spec, frame, [1, 2], records)
                    actual = result["predictions"].sort_values("station_id").y_pred.tolist()
                    self.assertEqual(actual, expected)

    def test_exact_targets_do_not_cross_missing_dates_or_segments(self):
        frame = pd.DataFrame({
            "station_id": [1, 1, 1, 1],
            "segment_id": [1, 1, 2, 2],
            "date": pd.to_datetime(["2025-01-01", "2025-01-03",
                                    "2025-01-04", "2025-01-05"]),
            "pm25": [10.0, 20.0, 30.0, 40.0],
        })
        result = features.add_exact_day_targets(frame, [1, 2])
        self.assertTrue(np.isnan(result.loc[0, "target_t1"]))
        self.assertEqual(result.loc[0, "target_t2"], 20.0)
        self.assertTrue(np.isnan(result.loc[1, "target_t1"]))
        self.assertEqual(result.loc[2, "target_t1"], 40.0)

    def test_fixed_local_baseline_uses_validation_early_stopping_and_reuses_model(self):
        dates = pd.date_range("2025-01-01", periods=55)
        frame = pd.DataFrame([
            {"station_id": sid, "date": date, "x": float(i % 9),
             **{f"target_t{h}": float(i + sid + h) for h in range(1, 8)}}
            for sid in (1, 2) for i, date in enumerate(dates)
        ])
        spec = models.RunSpec("fixed", training_strategy="local",
                              baseline_xgb=True, prefer_gpu=False)
        with patch.object(config, "VALIDATION_START", "2025-01-31"), \
             patch.object(config, "TEST_START", "2025-02-15"), \
             patch.object(config, "xgb_params_for_horizon",
                          side_effect=lambda h: _fast_xgb_params(h, 12)), \
             patch.object(config, "BASELINE_EARLY_STOPPING_ROUNDS", 3), \
             patch.object(config, "baseline_feature_list", return_value=["x"]):
            validation = models.run_holdout(spec, frame, [1, 2])
            test = models.predict_local_test_from_validation_models(
                spec, frame, validation["model_records"])
        self.assertEqual(len(validation["model_records"]), 14)
        self.assertTrue(all(r["estimator"].best_iteration < 12
                            for r in validation["model_records"]))
        self.assertEqual(set(test["reports"]["by_station"].station_id), {1, 2})
        self.assertEqual(len(test["model_records"]), 0)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifacts.save_models(root, validation["model_records"], "validation",
                                  validation["feature_list"])
            manifest = (root / "model" / "validation" / "manifest.json").read_text()
            self.assertIn("best_iteration", manifest)
            self.assertTrue(list((root / "model" / "validation").glob("*training_history.csv")))

    def test_shared_test_path_for_local_global_regional_and_multi(self):
        dates = pd.date_range("2025-01-01", periods=40)
        frame = pd.DataFrame([
            {"station_id": sid, "date": date, "x": float(i),
             "region_id": "North" if sid == 1 else "South",
             **{f"target_t{h}": float(i + sid + h) for h in range(1, 8)}}
            for sid in (1, 2) for i, date in enumerate(dates)
        ])
        factory = lambda prefer_gpu: XGBRegressor(n_estimators=2, max_depth=2,
                                                   n_jobs=1, tree_method="hist")
        specs = [
            models.RunSpec("local", training_strategy="local", prefer_gpu=False),
            models.RunSpec("global", training_strategy="global",
                           include_station_id=True, prefer_gpu=False),
            models.RunSpec("regional", training_strategy="regional",
                           include_region_id=True, prefer_gpu=False),
            models.RunSpec("multi", training_strategy="local",
                           forecast_strategy="multi", prefer_gpu=False),
        ]
        with patch.object(config, "VALIDATION_START", "2025-01-21"), \
             patch.object(config, "TEST_START", "2025-01-31"), \
             patch.object(config, "baseline_feature_list", return_value=["x"]), \
             patch.dict(models.ESTIMATOR_FACTORIES, {"xgboost": factory}):
            for spec in specs:
                with self.subTest(strategy=spec.training_strategy,
                                  forecast=spec.forecast_strategy):
                    result = models.run_holdout(spec, frame, [1, 2], "test")
                    self.assertEqual(set(result["reports"]["by_station"].station_id),
                                     {1, 2})
                    self.assertTrue(result["model_records"])
                    self.assertEqual(result["predictions"].date.min(), pd.Timestamp("2025-01-31"))
                    self.assertEqual(result["predictions"].date.max(), pd.Timestamp("2025-02-09"))

    def test_residual_models_and_per_station_test_metrics(self):
        dates = pd.date_range("2025-01-01", "2025-11-10", freq="D")
        rows = []
        for sid in (1, 2):
            for i, date in enumerate(dates):
                value = sid * 2 + np.sin(i / 10)
                row = {"station_id": sid, "date": date, "x": float(i % 31)}
                row.update({f"target_t{h}": value + h * 0.1 for h in range(1, 8)})
                rows.append(row)
        frame = pd.DataFrame(rows)
        spec = models.RunSpec("test", training_strategy="global_local_tree",
                              include_station_id=True, prefer_gpu=False)
        factory = lambda prefer_gpu: XGBRegressor(n_estimators=2, max_depth=2,
                                                   n_jobs=1, tree_method="hist")
        with patch.object(config, "VALIDATION_START", "2025-09-01"), \
             patch.object(config, "TEST_START", "2025-10-01"), \
             patch.object(config, "baseline_feature_list", return_value=["x"]), \
             patch.object(config.MLP_PARAMS, "min_station_rows", 2), \
             patch.dict(models.ESTIMATOR_FACTORIES, {"xgboost": factory}):
            result = models.run_holdout(spec, frame, [1, 2], evaluation="test")
        self.assertEqual(set(result["reports"]["by_station"].station_id), {1, 2})
        self.assertEqual(len(result["reports"]["by_horizon"]), 7)
        self.assertEqual(len([r for r in result["model_records"]
                              if r["role"] == "main" and r["group"] == "global"]), 7)
        self.assertTrue(any(r["role"] == "local_residual"
                            for r in result["model_records"]))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifacts.save_models(root, result["model_records"], "test",
                                  result["feature_list"])
            self.assertTrue((root / "model" / "test" / "manifest.json").exists())
            self.assertTrue(list((root / "model" / "test").glob("*.json")))
            with patch.object(config, "LOCAL_MASTER_CSV", root / "missing.csv"):
                artifacts.save_run(
                    run_dir=root, run_id="test", run_config={}, df=frame,
                    stations=[1, 2], folds_manifest=pd.DataFrame(),
                    reports=result["reports"], predictions=result["predictions"],
                    feature_list=result["feature_list"],
                    split="test", model_records=[])
            self.assertTrue((root / "metrics_by_station_test.csv").exists())
            self.assertTrue((root / "metrics_overall_test.json").exists())


if __name__ == "__main__":
    unittest.main()
