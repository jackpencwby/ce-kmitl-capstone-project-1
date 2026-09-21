"""Notebook result schema coverage for all E1 strategies."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config, notebook_outputs, models, runner, artifacts, data


class NotebookOutputTests(unittest.TestCase):
    def test_shared_e1_runner_writes_notebook_tree(self):
        dates = pd.date_range("2025-01-01", periods=40)
        frame = pd.DataFrame([
            {"station_id": sid, "date": date, "pm25": float(i + sid),
             "x": float(i % 7), "target_t1": float(i + sid + 1)}
            for sid in (1, 2) for i, date in enumerate(dates)
        ])
        spec = models.RunSpec("E1_SYNTHETIC", training_strategy="global",
                              include_station_id=True, baseline_xgb=True,
                              prefer_gpu=False)
        args = SimpleNamespace(train=True, source="local", stations=["all"],
                               max_stations=None, prefer_cpu=True, check_gcs=False,
                               upload_gcs=False)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "master.csv"
            source.write_text("synthetic dataset", encoding="utf-8")
            (root / "run").mkdir()
            with patch.object(config, "FORECAST_HORIZONS", (1,)), \
                 patch.object(config, "VALIDATION_START", "2025-01-21"), \
                 patch.object(config, "TEST_START", "2025-01-31"), \
                 patch.object(config, "LOCAL_MASTER_CSV", source), \
                 patch.object(config, "xgb_params_for_horizon",
                              side_effect=lambda h: {"n_estimators": 8,
                                                     "max_depth": 2}), \
                 patch.object(config, "BASELINE_EARLY_STOPPING_ROUNDS", 2), \
                 patch.object(config, "baseline_feature_list", return_value=["x"]), \
                 patch.object(runner, "prepare_dataframe", return_value=frame), \
                 patch.object(data, "resolve_stations", return_value=[1, 2]), \
                 patch.object(artifacts, "make_run_dir", return_value=root / "run"):
                self.assertEqual(runner.execute(spec, args), 0)
            self.assertTrue((root / "run" / "horizon_01" / "test" / "1" / "metrics.csv").exists())
            self.assertTrue((root / "run" / "horizon_01" / "model.ubj").exists())
            manifest = json.loads((root / "run" / "horizon_01" / "model_manifest.json").read_text())
            self.assertTrue(all((root / "run" / "horizon_01" / item["file"]).resolve().exists()
                                for item in manifest["models"]))

    def test_metrics_use_only_rows_paired_with_persistence(self):
        frame = pd.DataFrame({
            "actual_pm25": [1.0, 2.0, 3.0],
            "predicted_pm25": [2.0, 3.0, 4.0],
            "persistence_baseline": [1.0, float("nan"), 5.0],
        })
        result = notebook_outputs.metrics_for(frame)
        self.assertEqual(result["n_samples"], 3)
        self.assertEqual(result["comparison_n_samples"], 2)
        self.assertEqual(result["rmse"], 1.0)
        self.assertEqual(result["xgb_comparison_rmse"], 1.0)

    def test_horizon_and_station_files_match_notebook_columns(self):
        dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"])
        frame = pd.DataFrame([
            {"station_id": sid, "date": date, "pm25": float(sid + i),
             "x": float(i), "target_t1": float(sid + i + 1)}
            for sid in (1, 2) for i, date in enumerate(dates)
        ])
        prediction = frame[["station_id", "date", "target_t1"]].rename(
            columns={"target_t1": "y_true"})
        prediction["horizon"] = 1
        prediction["y_pred"] = prediction.y_true + 0.5
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text(json.dumps({
                "fixed_params": {"max_depth": 8}, "training_strategy": "local"}))
            (root / "dataset_manifest.json").write_text(
                json.dumps({"stations_used": ["1", "2"]}))
            for split in ("validation", "test"):
                model_dir = root / "model" / split
                model_dir.mkdir(parents=True)
                (model_dir / "manifest.json").write_text(json.dumps({
                    "models": [{"role": "main", "group": "1", "horizon": 1,
                                "file": "main__1__h1.json"}]}))
            with patch.object(config, "FORECAST_HORIZONS", (1,)):
                notebook_outputs.export(root, frame,
                                        {"validation": prediction, "test": prediction}, ["x"])
            horizon = root / "horizon_01"
            self.assertEqual(pd.read_csv(horizon / "test" / "predictions.csv").columns.tolist(),
                             notebook_outputs.PREDICTION_COLUMNS)
            self.assertEqual(pd.read_csv(horizon / "test" / "metrics.csv").columns.tolist(),
                             ["horizon", "split"] + notebook_outputs.METRIC_COLUMNS)
            self.assertTrue((horizon / "test" / "1" / "time_series.png").is_file())
            self.assertTrue((horizon / "station_summary.csv").is_file())
            self.assertTrue((root / "horizon_summary.csv").is_file())
            self.assertTrue((root / "all_station_summary.csv").is_file())
            self.assertTrue((root / "completion.json").is_file())
            model_manifest = json.loads((horizon / "model_manifest.json").read_text())
            self.assertEqual(len(model_manifest["models"]), 2)
            self.assertFalse((horizon / "model.ubj").exists())


if __name__ == "__main__":
    unittest.main()
