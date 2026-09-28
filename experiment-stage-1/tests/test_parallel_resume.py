"""Verify spawned horizon workers match sequential results and reuse checkpoints."""
import sys
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config, models
from resume_e2_3_test import train_horizon, positive_workers


class ParallelResumeTests(unittest.TestCase):
    def test_parallel_matches_serial_and_reuses_checkpoint(self):
        dates = pd.date_range("2026-03-01", "2026-04-20")
        cols = models.assemble_feature_columns(models.RunSpec("test"))
        frame = pd.DataFrame({column: [float(i % 11) for i in range(len(dates))] for column in cols})
        frame["station_id"] = 1
        frame["date"] = dates
        for h in (1, 2):
            frame[f"target_t{h}"] = [float((i + h) % 13) for i in range(len(dates))]
        spec = models.RunSpec("E2_TEST", algorithm="gbr", training_strategy="global",
                              include_station_id=True, parameter_profile="e2", prefer_gpu=False)
        features = cols + ["station_id_code"]
        with tempfile.TemporaryDirectory() as directory:
            serial, parallel = Path(directory) / "serial", Path(directory) / "parallel"
            serial.mkdir()
            parallel.mkdir()
            for h in (1, 2):
                train_horizon(h, spec, frame, [1], features, serial)
            with ProcessPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(train_horizon, h, spec, frame, [1], features, parallel) for h in (1, 2)]
                self.assertEqual(sorted(f.result() for f in futures), [1, 2])
            for h in (1, 2):
                a, b = joblib.load(serial / f"h{h}.joblib"), joblib.load(parallel / f"h{h}.joblib")
                pd.testing.assert_frame_equal(a["predictions"], b["predictions"])
                self.assertGreaterEqual(b["predictions"]["date"].min(), pd.Timestamp(config.TEST_START))
            with patch.object(models, "run_holdout", side_effect=AssertionError("checkpoint was refitted")):
                self.assertEqual(train_horizon(1, spec, frame, [1], features, parallel), 1)
        self.assertEqual(list(config.FORECAST_HORIZONS), list(range(1, 8)))

    def test_worker_count_must_be_positive(self):
        import argparse
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_workers("0")


if __name__ == "__main__":
    unittest.main()
