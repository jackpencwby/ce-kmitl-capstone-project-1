"""Regression coverage for the September 19 per-station export."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config, data, features, models


class DatasetTests(unittest.TestCase):
    def client(self):
        client = Mock()
        prefix = "clean-data/preprocess-09-19/"
        objects = {
            prefix + "1003_Nakhon Nayok Weather Observing Station/daily_dataset.csv":
                b"station_id,date,co_mean_ugm3\n1003,2026-01-02,123\n",
            prefix + "1_Chiang Mai/daily_dataset.csv":
                b"date,co_mean_ugm3,station_id\n2026-01-01,456,1\n",
        }
        client.get_paginator.return_value.paginate.return_value = [
            {"Contents": [{"Key": key}]} for key in
            [*objects, prefix + "all_stations_daily.csv", prefix + "summary.csv"]
        ]
        client.get_object.side_effect = lambda **kw: {"Body": io.BytesIO(objects[kw["Key"]])}
        return client

    def test_gcs_load_combines_paginated_station_files_without_summary_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder) / "master.csv"
            with patch.object(data, "load_env", return_value={"GCS_BUCKET": "test"}), \
                 patch.object(data, "_s3_client", return_value=self.client()), \
                 patch.object(config, "LOCAL_MASTER_CSV", local):
                frame = data.load_master("gcs")
                self.assertEqual(frame.station_id.tolist(), [1, 1003])
                self.assertEqual(frame.co_mean_ugm3.tolist(), [456, 123])
                self.assertEqual(len(pd.read_csv(local)), 2)

    def test_failed_download_preserves_existing_cache(self):
        client = self.client()
        client.get_object.side_effect = [
            {"Body": io.BytesIO(b"station_id,date\n1003,2026-01-02\n")},
            IOError("interrupted download"),
        ]
        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder) / "master.csv"
            local.write_text("existing cache", encoding="utf-8")
            with patch.object(data, "load_env", return_value={"GCS_BUCKET": "test"}), \
                 patch.object(data, "_s3_client", return_value=client):
                with self.assertRaises(IOError):
                    data.download_master_from_gcs(local)
            self.assertEqual(local.read_text(), "existing cache")
            self.assertEqual(list(Path(folder).iterdir()), [local])

    def test_connection_check_accepts_station_layout(self):
        client = self.client()

        def head_object(**kwargs):
            if not kwargs["Key"].endswith("/daily_dataset.csv"):
                raise RuntimeError("No combined CSV exists")
            return {}

        client.head_object.side_effect = head_object
        with patch.object(data, "load_env", return_value={"GCS_BUCKET": "test"}), \
             patch.object(data, "_s3_client", return_value=client):
            self.assertTrue(data.check_gcs_connection())

    def test_empty_prefix_does_not_create_a_cache(self):
        client = self.client()
        client.get_paginator.return_value.paginate.return_value = [{}]
        with tempfile.TemporaryDirectory() as folder:
            local = Path(folder) / "master.csv"
            with patch.object(data, "load_env", return_value={"GCS_BUCKET": "test"}), \
                 patch.object(data, "_s3_client", return_value=client):
                with self.assertRaises(FileNotFoundError):
                    data.download_master_from_gcs(local)
            self.assertFalse(local.exists())

    def test_co_aod_are_model_inputs_and_missing_values_stay_causal(self):
        frame = pd.DataFrame({c: [1.0] * 40 for c in config.baseline_feature_list()
                              if not c.endswith("_missing")})
        frame["station_id"] = 1
        frame["date"] = pd.date_range("2026-01-01", periods=40)
        frame["pm25"] = 10.0
        frame["co_mean_ugm3"] = 123.0
        frame["co_8h_max_ugm3"] = np.nan
        frame["aod500_mean"] = np.nan
        frame.loc[39, "aod500_mean"] = 0.8
        result = features.build_base_features(frame)
        columns = models.assemble_feature_columns(models.RunSpec("test"))
        for col in ("co_mean_ugm3", "co_8h_max_ugm3", "aod500_mean"):
            self.assertIn(col, columns)
            self.assertIn(col + "_missing", columns)
        self.assertTrue(pd.isna(result.loc[38, "aod500_mean"]))
        self.assertEqual(result.loc[38, "aod500_mean_missing"], 1)
        self.assertEqual(result.loc[39, "aod500_mean"], 0.8)
        self.assertEqual(result.loc[39, "aod500_mean_missing"], 0)
        self.assertEqual(result.loc[38, "co_mean_ugm3"], 123)
        self.assertTrue(pd.isna(result.loc[38, "co_8h_max_ugm3"]))
        self.assertIn("pm25", columns)
        self.assertEqual(result.loc[38, "pm25"], 10.0)
        self.assertTrue(pd.isna(frame.loc[38, "aod500_mean"]))
        with self.assertRaises(KeyError):
            features.build_base_features(frame.drop(columns="co_mean_ugm3"))


if __name__ == "__main__":
    unittest.main()
