"""Unit tests for spatial wind/distance helpers (Experimental_Plan.md 9).

The plan explicitly requires unit tests for the four cardinal directions and
for the wind-convention handling. Run from the repo root:

  .venv/Scripts/python.exe -m unittest discover -s experiment-stage-1/tests -v
"""

from __future__ import annotations

import os
import sys
import unittest

# Make the experiment-stage-1 directory importable as the package root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import config, features  # noqa: E402


class TestBearing(unittest.TestCase):
    def test_cardinal_bearings(self):
        # From origin (0,0): north, east, south, west neighbours.
        self.assertAlmostEqual(features.initial_bearing_deg(0, 0, 1, 0), 0.0, places=1)     # N
        self.assertAlmostEqual(features.initial_bearing_deg(0, 0, 0, 1), 90.0, places=1)    # E
        self.assertAlmostEqual(features.initial_bearing_deg(0, 0, -1, 0), 180.0, places=1)  # S
        self.assertAlmostEqual(features.initial_bearing_deg(0, 0, 0, -1), 270.0, places=1)  # W

    def test_haversine_known_distance(self):
        # One degree of latitude is ~111 km.
        d = features.haversine_km(0, 0, 1, 0)
        self.assertTrue(110 < d < 112, f"expected ~111 km, got {d:.2f}")

    def test_haversine_symmetry_and_zero(self):
        self.assertAlmostEqual(features.haversine_km(13.7, 100.5, 13.7, 100.5), 0.0, places=6)
        a = features.haversine_km(13.7, 100.5, 18.8, 98.9)
        b = features.haversine_km(18.8, 98.9, 13.7, 100.5)
        self.assertAlmostEqual(a, b, places=6)


class TestAngularDifference(unittest.TestCase):
    def test_wrap(self):
        self.assertAlmostEqual(features.angular_difference_deg(10, 350), 20.0, places=6)
        self.assertAlmostEqual(features.angular_difference_deg(0, 180), 180.0, places=6)
        self.assertAlmostEqual(features.angular_difference_deg(90, 90), 0.0, places=6)


class TestWindWeighting(unittest.TestCase):
    """A neighbour that is directly upwind should carry more weight than one
    that is crosswind, given equal distance. Wind is 'wind from' convention:
    wind_from=0 means air arrives from the north, so the upwind neighbour is
    to the north (bearing 0)."""

    def _tiny_dataset(self):
        # Target station 1 at origin; neighbour 2 to the north, neighbour 3
        # to the east. Two consecutive days so lag_1 is defined on day 2.
        dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
        rows = []
        # station 1 (target): wind_from = 0 (from the north)
        for d in dates:
            rows.append({config.STATION_ID_COL: 1, config.DATE_COL: d,
                         config.LAT_COL: 0.0, config.LON_COL: 0.0,
                         config.TARGET_COL: 10.0, config.WIND_FROM_COL: 0.0})
        # neighbour 2 to the north with HIGH pm; neighbour 3 to the east LOW
        for d in dates:
            rows.append({config.STATION_ID_COL: 2, config.DATE_COL: d,
                         config.LAT_COL: 1.0, config.LON_COL: 0.0,
                         config.TARGET_COL: 100.0, config.WIND_FROM_COL: 0.0})
            rows.append({config.STATION_ID_COL: 3, config.DATE_COL: d,
                         config.LAT_COL: 0.0, config.LON_COL: 1.0,
                         config.TARGET_COL: 20.0, config.WIND_FROM_COL: 0.0})
        return pd.DataFrame(rows)

    def test_upwind_neighbor_dominates(self):
        df = self._tiny_dataset()
        coords = df[[config.STATION_ID_COL, config.LAT_COL, config.LON_COL]].groupby(
            config.STATION_ID_COL, as_index=False).first()
        nbr = features.build_neighbor_table(coords, config.SpatialConfig(
            max_radius_km=500, max_neighbors=5, epsilon_km=1.0, wind_power=2.0,
            neighbor_lags=(1,)))
        wind = features.add_neighbor_features(df, nbr, mode="wind",
                                              spatial=config.SpatialConfig(
                                                  max_radius_km=500, max_neighbors=5,
                                                  epsilon_km=1.0, wind_power=2.0,
                                                  neighbor_lags=(1,)))
        unweighted = features.add_neighbor_features(df, nbr, mode="unweighted",
                                                    spatial=config.SpatialConfig(
                                                        max_radius_km=500, max_neighbors=5,
                                                        epsilon_km=1.0, wind_power=2.0,
                                                        neighbor_lags=(1,)))
        # Day 2, target station 1: lag_1 aggregate of neighbours from day 1.
        w_val = wind[(wind[config.STATION_ID_COL] == 1)]["neighbor_pm_lag_1"].dropna().iloc[0]
        u_val = unweighted[(unweighted[config.STATION_ID_COL] == 1)]["neighbor_pm_lag_1"].dropna().iloc[0]
        # Upwind (north, pm=100) should pull the wind aggregate above the plain mean (60).
        self.assertGreater(w_val, u_val)
        self.assertGreater(w_val, 60.0)


if __name__ == "__main__":
    unittest.main()
