"""Stage 2 global + station residual forecasting contracts."""
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jack_run_stage2 as stage2


class MeanRegressor(RegressorMixin, BaseEstimator):
    fits = []

    def fit(self, X, y):
        self.mean_ = np.mean(y)
        self.feature_importances_ = np.zeros(X.shape[1])
        self.fits.append((X.copy(), y.copy()))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class ResidualTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.concat([
            pd.DataFrame({'station_id': sid, 'date': pd.date_range('2024-01-01', periods=450),
                          'x': np.arange(450, dtype=float), 'region_id': 1})
            for sid in [1, 2]
        ], ignore_index=True)
        for h in range(1, 8):
            self.frame[f'target_t{h}'] = self.frame.x + self.frame.station_id * 100 + h
        for target, name, value in [
            (stage2.config, 'VALIDATION_START', '2025-01-01'),
            (stage2.config, 'TEST_START', '2025-02-01'),
            (stage2.config, 'baseline_feature_list', lambda: ['x']),
            (stage2, 'make_cpu_lgbm', lambda *a, **kw: MeanRegressor()),
        ]:
            p = patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        MeanRegressor.fits.clear()

    def run_experiment(self, experiment, split='validation', frame=None):
        return stage2.run_pooled(stage2.make_spec(experiment),
                                self.frame if frame is None else frame, [1, 2], {}, split)

    def test_station_residuals_are_added_and_models_are_saved(self):
        for experiment in [15, 16]:
            with self.subTest(experiment=experiment):
                result = self.run_experiment(experiment)
                records = result['model_records']
                local = [r for r in records if r['role'].startswith('local_residual')]
                self.assertEqual({r['group'] for r in local}, {'1', '2'})
                self.assertEqual(len(local), 14 if experiment == 15 else 2)
                pred = result['predictions']
                self.assertEqual(set(pred.horizon), set(range(1, 8)))
                station_mean = pred.groupby('station_id').y_pred.mean()
                self.assertAlmostEqual(station_mean[2] - station_mean[1], 100.)

    def test_holdout_labels_never_affect_predictions(self):
        for experiment in [15, 16]:
            for split, boundary in [('validation', '2025-01-01'), ('test', '2025-02-01')]:
                with self.subTest(experiment=experiment, split=split):
                    original = self.run_experiment(experiment, split)
                    changed = self.frame.copy()
                    for h in range(1, 8):
                        changed.loc[changed.date + pd.Timedelta(days=h) >= boundary,
                                    f'target_t{h}'] += 1e6
                    other = self.run_experiment(experiment, split, changed)
                    np.testing.assert_allclose(original['predictions'].y_pred,
                                               other['predictions'].y_pred)

    def test_short_training_history_uses_global_only(self):
        frame = self.frame[self.frame.date >= '2024-10-01'].copy()
        for experiment in [15, 16]:
            result = self.run_experiment(experiment, frame=frame)
            self.assertFalse(any(r['role'].startswith('local_residual')
                                 for r in result['model_records']))
            self.assertTrue(np.isfinite(result['predictions'].y_pred).all())

    def test_multi_residual_targets_use_past_only_oof_predictions(self):
        self.run_experiment(16)
        # For this ramp, an in-sample residual has mean zero per horizon;
        # a forward OOF residual has positive mean because the past mean lags.
        locals_ = [(X, y) for X, y in MeanRegressor.fits
                   if X.x.min() >= 180 and X.x.max() < 366]
        # One horizon-stacked multi-head fit per station (previously 7 wrapper estimators each).
        self.assertEqual(len(locals_), 2)
        self.assertTrue(all(float(y.mean()) > 0 for _, y in locals_))

    def test_multi_output_excludes_incomplete_target_rows(self):
        frame = self.frame.copy()
        frame.loc[frame.date == '2025-01-05', 'target_t7'] = np.nan
        result = self.run_experiment(16, frame=frame)
        self.assertFalse((result['predictions'].date == '2025-01-05').any())

    def test_experiment16_wide_features_without_fragmentation_warning(self):
        columns = [f'feature_{i}' for i in range(150)]
        wide = pd.DataFrame({name: self.frame.x + i for i, name in enumerate(columns)},
                            index=self.frame.index)
        frame = pd.concat([self.frame, wide], axis=1)
        frame.index = np.arange(len(frame)) * 3
        for split in ['validation', 'test']:
            with self.subTest(split=split):
                expected = self.run_experiment(16, split)['predictions']
                with patch.object(stage2.config, 'baseline_feature_list', lambda: ['x'] + columns), \
                     patch.object(stage2.models, '_run_residual',
                                  side_effect=AssertionError('Experiment 16 must use its multi-output path')), \
                     warnings.catch_warnings():
                    warnings.simplefilter('error', pd.errors.PerformanceWarning)
                    result = self.run_experiment(16, split, frame)
                pd.testing.assert_frame_equal(expected.reset_index(drop=True),
                                              result['predictions'].reset_index(drop=True))
                self.assertEqual(sum(r['role'] == 'local_residual_multi_output'
                                     for r in result['model_records']), 2)
        for X, _ in MeanRegressor.fits:
            if 'feature_149' in X:
                np.testing.assert_array_equal(X.feature_149, X.x + 149)

    def test_real_lightgbm_and_model_serialization(self):
        import joblib
        from lightgbm import LGBMRegressor

        def factory(*args, **kwargs):
            return LGBMRegressor(n_estimators=2, num_leaves=3, min_child_samples=5,
                                 n_jobs=1, verbosity=-1, device_type='cpu')

        with patch.object(stage2, 'make_cpu_lgbm', factory):
            for experiment in [1, 2, 3, 4, 15, 16]:
                with self.subTest(experiment=experiment):
                    result = self.run_experiment(experiment)
                    self.assertTrue(np.isfinite(result['predictions'].y_pred).all())
                    with tempfile.TemporaryDirectory() as directory:
                        stage2.artifacts.save_models(Path(directory), result['model_records'],
                                                     'validation', result['feature_list'])
                        files = list(Path(directory).rglob('*.joblib'))
                        self.assertEqual(len(files), len(result['model_records']))
                        for file in files:
                            self.assertTrue(hasattr(joblib.load(file), 'predict'))


if __name__ == '__main__':
    unittest.main()
