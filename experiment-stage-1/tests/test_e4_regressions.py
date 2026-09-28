"""Regressions for test routing and calendar-based spatial features."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config, features, models


class E4RegressionTests(unittest.TestCase):
    def test_holdout_partitions_respect_requested_split(self):
        frame = pd.DataFrame([
            {'station_id': sid, 'region_id': region, 'date': pd.Timestamp(date)}
            for sid, region in [(1, 'North'), (2, 'South'), (3, 'North')]
            for date in ['2025-01-01', '2025-01-10', '2025-01-20']
        ]).sample(frac=1, random_state=42)
        # Preserve non-contiguous indexes and exclude station 3.
        for strategy in ['local', 'global', 'regional']:
            for evaluation in ['validation', 'test']:
                with self.subTest(strategy=strategy, evaluation=evaluation):
                    captured = []

                    def capture(spec, factory, train, val, cols, records, group):
                        captured.append((train.copy(), val.copy()))
                        return pd.DataFrame(), []

                    with patch.object(config, 'VALIDATION_START', '2025-01-10'), \
                         patch.object(config, 'TEST_START', '2025-01-20'), \
                         patch.object(models, '_fit_partition', side_effect=capture), \
                         patch.object(models, '_finalize', return_value={}):
                        models.run_holdout(models.RunSpec('regression', training_strategy=strategy),
                                           frame, [1, 2], evaluation=evaluation)
                    self.assertTrue(captured)
                    expected_train = {'2025-01-01'}
                    if evaluation == 'test':
                        expected_train.add('2025-01-10')
                    for train, val in captured:
                        self.assertEqual(set(train.date.dt.strftime('%Y-%m-%d')), expected_train)
                        self.assertEqual(set(val.date.dt.strftime('%Y-%m-%d')),
                                         {'2025-01-20' if evaluation == 'test' else '2025-01-10'})
                        self.assertNotIn(3, train.station_id.values)
                        self.assertNotIn(3, val.station_id.values)

    def test_spatial_lags_use_exact_dates_and_do_not_cross_segments(self):
        frame = pd.DataFrame([
            {'station_id': sid, 'date': pd.Timestamp(date), 'segment_id': segment,
             'pm25': value, 'wind_direction_avg': 0.0}
            for sid in [1, 2]
            for date, segment, value in [('2025-01-01', 1, 10.),
                                         ('2025-01-03', 1, 30.),
                                         ('2025-01-04', 2, 40.),
                                         ('2025-01-05', 2, 50.)]
        ]).sample(frac=1, random_state=42)
        neighbors = pd.DataFrame({'station_id': [1, 2], 'neighbor_id': [2, 1],
                                  'distance_km': [1., 1.], 'bearing_deg': [0., 0.]})
        for mode in ['unweighted', 'distance', 'wind']:
            for use_segments in [True, False]:
                with self.subTest(mode=mode, segments=use_segments):
                    source = frame if use_segments else frame.drop(columns='segment_id')
                    result = features.add_neighbor_features(source, neighbors, mode=mode,
                        spatial=config.SpatialConfig(neighbor_lags=(1, 2, 3, 7)))
                    self.assertEqual(len(result), len(source))
                    self.assertEqual(result[['station_id', 'date']].values.tolist(),
                                     source[['station_id', 'date']].values.tolist())
                    for sid in [1, 2]:
                        s = result[result.station_id == sid].set_index('date')
                        self.assertTrue(np.isnan(s.loc['2025-01-03', 'neighbor_pm_lag_1']))
                        self.assertEqual(s.loc['2025-01-03', 'neighbor_pm_lag_2'], 10.)
                        self.assertEqual(s.loc['2025-01-05', 'neighbor_pm_lag_1'], 40.)
                        self.assertTrue(s.neighbor_pm_lag_7.isna().all())
                        if use_segments:
                            self.assertTrue(np.isnan(s.loc['2025-01-04', 'neighbor_pm_lag_1']))
                            self.assertTrue(np.isnan(s.loc['2025-01-05', 'neighbor_pm_lag_2']))
                        else:
                            self.assertEqual(s.loc['2025-01-04', 'neighbor_pm_lag_1'], 30.)


if __name__ == '__main__':
    unittest.main()
