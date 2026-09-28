import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import models, config_e2
from common.training_progress import GBRProgress, summarize, write_atomic


class TrainingProgressTests(unittest.TestCase):
    def test_atomic_write_retries_windows_reader_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            original = Path.replace
            calls = []
            def replace(source, target):
                calls.append(target)
                if len(calls) == 1:
                    raise PermissionError("reader holds file")
                return original(source, target)
            with patch.object(Path, "replace", replace), patch("common.training_progress.time.sleep"):
                write_atomic(path, {"trees_done": 1})
            self.assertEqual(json.loads(path.read_text()), {"trees_done": 1})
            self.assertEqual(len(calls), 2)

    def test_best_effort_progress_write_never_stops_training(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            with patch.object(Path, "replace", side_effect=PermissionError("locked")), \
                 patch("common.training_progress.time.sleep"):
                self.assertFalse(write_atomic(path, {"trees_done": 1}, best_effort=True))
            self.assertFalse(path.exists())

    def test_counts_eta_and_checkpoint_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("common.training_progress.time.monotonic", side_effect=[100, 120]):
                monitor = GBRProgress(root, 1, [1, 2], 100, 10)
                with contextlib.redirect_stdout(io.StringIO()) as log:
                    self.assertFalse(monitor(9, None, {}))
            row = json.loads((root / "h1_progress.json").read_text())
            self.assertEqual(row["training_percent"], 10)
            self.assertEqual(row["estimated_training_seconds_remaining"], 180)
            self.assertIn("overall training 5.0%", log.getvalue())
            (root / "h2.joblib").touch()
            self.assertEqual(summarize(root, [1, 2], 100)["training_percent"], 55)
            self.assertEqual(summarize(root, [1, 2, 3], 100)["horizons"][2]["state"], "unknown_or_pending")

    def test_monitor_does_not_change_fitted_predictions(self):
        from functools import partial
        x = pd.DataFrame({"x": [float(i) for i in range(60)]})
        y = pd.Series([float(i % 7) for i in range(60)])
        params = {**config_e2.params_for("gbr"), "n_estimators": 4}
        with tempfile.TemporaryDirectory() as directory, patch.object(config_e2, "params_for", return_value=params):
            factory = partial(models.make_gbr, parameter_profile="e2")
            baseline, _ = models._fit_predict_estimator(factory, False, x, y, x)
            monitor = GBRProgress(Path(directory), 1, [1], 4, 2)
            actual, _ = models._fit_predict_estimator(factory, False, x, y, x, gbr_monitor=monitor)
            self.assertEqual(baseline.tolist(), actual.tolist())
            row = json.loads((Path(directory) / "h1_progress.json").read_text())
            self.assertEqual(row["trees_done"], 4)
            self.assertEqual(row["state"], "predicting")


if __name__ == "__main__":
    unittest.main()
