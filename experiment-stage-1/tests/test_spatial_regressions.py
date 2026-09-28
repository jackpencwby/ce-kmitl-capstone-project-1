"""Small causal fixtures for E4 neighbor weighting and fallback behavior."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config, features


class SpatialRegressionTests(unittest.TestCase):
    spatial = config.SpatialConfig(neighbor_lags=(1, 3))

    def fixture(self):
        frame = pd.DataFrame([
            {"station_id": sid, "date": day, "segment_id": 1, "pm25": pm,
             "wind_direction_avg": 0.0, "wind_speed_avg": 2.0}
            for sid, pm in [(1, 20.), (2, 10.), (3, 90.)]
            for day in pd.date_range("2025-01-01", periods=5)
        ])
        neighbors = pd.DataFrame({"station_id": [1, 1], "neighbor_id": [2, 3],
                                  "distance_km": [1., 3.], "bearing_deg": [0., 180.]})
        return frame, neighbors

    def target_result(self, frame, neighbors, mode="wind"):
        result = features.add_neighbor_features(frame, neighbors, mode, self.spatial)
        return result[result.station_id == 1].set_index("date")

    def test_missing_own_pm_does_not_remove_target_or_dates(self):
        frame, neighbors = self.fixture()
        frame.loc[frame.station_id == 1, "pm25"] = np.nan
        for mode, expected in [("unweighted", 50.), ("distance", 110. / 3.), ("wind", 10.)]:
            with self.subTest(mode=mode):
                result = self.target_result(frame, neighbors, mode)
                self.assertAlmostEqual(result.loc["2025-01-02", "neighbor_pm_lag_1"], expected)
        frame.loc[:, "pm25"] = np.nan
        result = self.target_result(frame, neighbors)
        self.assertEqual(len(result), 5)
        self.assertTrue(result.neighbor_pm_lag_1.isna().all())

    def test_no_neighbors_keeps_schema_and_missing_features(self):
        coords = pd.DataFrame({"station_id": [1], "lat": [14.], "long": [100.]})
        neighbors = features.build_neighbor_table(coords)
        self.assertEqual(list(neighbors), ["station_id", "neighbor_id", "distance_km", "bearing_deg"])
        frame, _ = self.fixture()
        for mode in ["unweighted", "distance", "wind"]:
            with self.subTest(mode=mode):
                result = self.target_result(frame[frame.station_id == 1], neighbors, mode)
                self.assertTrue(result.neighbor_pm_lag_1.isna().all())
                if mode == "wind":
                    self.assertEqual(result.loc["2025-01-02", "neighbor_wind_fallback_lag_1"], 1.)

    def test_calm_missing_wind_and_effective_zero_weights_use_distance(self):
        for cause in ["calm", "negative_speed", "missing_speed", "missing_direction",
                      "missing_upwind_pm", "crosswind_roundoff"]:
            with self.subTest(cause=cause):
                frame, neighbors = self.fixture()
                target = frame.station_id == 1
                if cause in ["calm", "negative_speed", "missing_speed"]:
                    frame.loc[target, "wind_speed_avg"] = {"calm": 0., "negative_speed": -1.,
                                                           "missing_speed": np.nan}[cause]
                elif cause == "missing_direction":
                    frame.loc[target, "wind_direction_avg"] = np.nan
                elif cause == "missing_upwind_pm":
                    frame.loc[(frame.station_id == 2) & (frame.date == pd.Timestamp("2025-01-01")),
                              "pm25"] = np.nan
                else:
                    neighbors["bearing_deg"] = [90., 270.]
                wind = self.target_result(frame, neighbors)
                distance = self.target_result(frame, neighbors, "distance")
                self.assertAlmostEqual(wind.loc["2025-01-02", "neighbor_pm_lag_1"],
                                       distance.loc["2025-01-02", "neighbor_pm_lag_1"])
                self.assertEqual(wind.loc["2025-01-02", "neighbor_wind_fallback_lag_1"], 1.)

    def test_fallback_flags_are_lagged_and_wind_only(self):
        frame, neighbors = self.fixture()
        frame.loc[(frame.station_id == 1) & (frame.date == pd.Timestamp("2025-01-02")),
                  "wind_speed_avg"] = 0.
        wind = self.target_result(frame, neighbors)
        self.assertTrue(np.isnan(wind.loc["2025-01-01", "neighbor_wind_fallback_lag_1"]))
        self.assertEqual(wind.loc["2025-01-02", "neighbor_wind_fallback_lag_1"], 0.)
        self.assertEqual(wind.loc["2025-01-03", "neighbor_wind_fallback_lag_1"], 1.)
        self.assertEqual(wind.loc["2025-01-05", "neighbor_wind_fallback_lag_3"], 1.)
        expected = ["neighbor_pm_lag_1", "neighbor_pm_lag_3"]
        self.assertEqual(features.neighbor_feature_cols(self.spatial), expected)
        for mode in ["distance", "unweighted"]:
            result = self.target_result(frame, neighbors, mode)
            self.assertEqual(features.neighbor_feature_cols(self.spatial, mode=mode), expected)
            self.assertFalse(any("fallback" in col for col in result))
        self.assertEqual(features.neighbor_feature_cols(self.spatial, mode="wind"),
                         expected + ["neighbor_wind_fallback_lag_1", "neighbor_wind_fallback_lag_3"])

    def test_flags_obey_exact_dates_segments_and_causality(self):
        frame, neighbors = self.fixture()
        frame.loc[frame.date == pd.Timestamp("2025-01-01"), "wind_speed_avg"] = 0.
        frame = frame[frame.date != pd.Timestamp("2025-01-02")].copy()
        frame.loc[frame.date >= pd.Timestamp("2025-01-04"), "segment_id"] = 2
        before = self.target_result(frame, neighbors)
        self.assertTrue(np.isnan(before.loc["2025-01-03", "neighbor_wind_fallback_lag_1"]))
        self.assertTrue(np.isnan(before.loc["2025-01-04", "neighbor_wind_fallback_lag_3"]))
        self.assertTrue(np.isnan(before.loc["2025-01-04", "neighbor_wind_fallback_lag_1"]))
        frame.loc[frame.date >= pd.Timestamp("2025-01-04"), ["pm25", "wind_direction_avg", "wind_speed_avg"]] = 999.
        after = self.target_result(frame, neighbors)
        cols = features.neighbor_feature_cols(self.spatial, mode="wind")
        pd.testing.assert_frame_equal(before.loc[:"2025-01-04", cols], after.loc[:"2025-01-04", cols])


if __name__ == "__main__":
    unittest.main()
