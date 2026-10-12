"""Behavioral checks for XGBoost global + local tree correction."""
import sys
import tempfile
import json
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_xgb_residual as residual
REAL_XGB = residual.base._xgb


class MeanModel:
    def fit(self, X, y, **kwargs):
        self.mean = np.asarray(y).mean(axis=0)
        self.feature_importances_ = np.zeros(X.shape[1])
        return self

    def predict(self, X):
        return np.broadcast_to(self.mean, (len(X),) + np.asarray(self.mean).shape).copy()


class XGBResidualTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.concat([
            pd.DataFrame({'station_id': sid, 'date': pd.date_range('2024-01-01', periods=450),
                          'x': np.arange(450, dtype=float)}) for sid in [1, 2]
        ], ignore_index=True)
        for h in range(1, 8):
            self.frame[f'target_t{h}'] = self.frame.x + 100 * self.frame.station_id + h
        for obj, name, value in [
            (residual.base.config, 'VALIDATION_START', '2025-01-01'),
            (residual.base.config, 'TEST_START', '2025-02-01'),
            (residual.base.config, 'baseline_feature_list', lambda: ['x']),
            (residual.base, '_xgb', lambda *a, **kw: MeanModel()),
        ]:
            p = patch.object(obj, name, value)
            p.start()
            self.addCleanup(p.stop)

    def evaluate(self, experiment, frame=None, split='validation', rounds=None):
        spec = residual.base.make_spec(experiment, 42)
        context = residual.make_context(self.frame if frame is None else frame, [1, 2], spec, split)
        return residual.fit_evaluate(context, {}, 3, None, 42, {'total': 0}, True,
                                     selected_rounds=rounds)

    def test_global_plus_station_correction_and_native_heads(self):
        for experiment, count in [(13, 21), (14, 3)]:
            with self.subTest(experiment=experiment):
                result = self.evaluate(experiment)
                self.assertEqual(len(result['model_records']), count)
                means = result['predictions'].groupby('station_id').y_pred.mean()
                self.assertAlmostEqual(means[2] - means[1], 100.)
                self.assertTrue(all('local_residual' in r['role'] or r['group'] == 'global'
                                    for r in result['model_records']))

    def test_heldout_labels_cannot_change_fitted_models(self):
        for experiment in [13, 14]:
            selected = self.evaluate(experiment)['selected_rounds']
            for split, cutoff in [('validation', '2025-01-01'), ('test', '2025-02-01')]:
                rounds = selected if split == 'test' else None
                expected = self.evaluate(experiment, split=split, rounds=rounds)
                changed = self.frame.copy()
                for h in range(1, 8):
                    changed.loc[changed.date + np.timedelta64(h, 'D') >= cutoff,
                                f'target_t{h}'] += 1e6
                actual = self.evaluate(experiment, changed, split, rounds)
                np.testing.assert_allclose(expected['predictions'].y_pred, actual['predictions'].y_pred)

    def test_short_history_falls_back_to_global(self):
        frame = self.frame[self.frame.date >= '2024-10-01']
        for experiment, count in [(13, 7), (14, 1)]:
            result = self.evaluate(experiment, frame)
            self.assertEqual(len(result['model_records']), count)

    def test_wide_features_have_no_fragmentation(self):
        names = [f'x{i}' for i in range(140)]
        frame = pd.concat([self.frame, pd.DataFrame({n: self.frame.x for n in names})], axis=1)
        frame.index = np.arange(len(frame)) * 3
        with patch.object(residual.base.config, 'baseline_feature_list', lambda: ['x'] + names), \
             warnings.catch_warnings():
            warnings.simplefilter('error', pd.errors.PerformanceWarning)
            for experiment in [13, 14]:
                self.evaluate(experiment, frame)

    def test_real_xgboost_scalar_and_vector_models_roundtrip(self):
        import xgboost as xgb

        def cpu_factory(*args, **kwargs):
            return REAL_XGB(*args, **kwargs).set_params(device='cpu', n_jobs=1, max_depth=2)

        with patch.object(residual.base, '_xgb', cpu_factory):
            for experiment in [13, 14]:
                with self.subTest(experiment=experiment):
                    result = self.evaluate(experiment)
                    self.assertTrue(np.isfinite(result['predictions'].y_pred).all())
                    with tempfile.TemporaryDirectory() as directory:
                        residual.base.artifacts.save_models(Path(directory), result['model_records'],
                                                            'validation', ['x', 'station_id_code'])
                        files = list(Path(directory).rglob('*.json'))
                        model_files = [f for f in files if '__h' in f.name]
                        self.assertEqual(len(model_files), len(result['model_records']))
                        for record in result['model_records']:
                            name = f"{record['role']}__{record['group']}__h{record['horizon']}.json"
                            loaded = xgb.XGBRegressor()
                            loaded.load_model(Path(directory) / 'model' / 'validation' / name)
                            loaded.set_params(device='cpu', n_jobs=1)
                            X = pd.DataFrame({'x': [10., 20.], 'station_id_code': [0, 1]})
                            # JSON roundtrips of vector base scores can differ by
                            # a few float32 ULPs (observed max absolute error 1.6e-5).
                            np.testing.assert_allclose(loaded.predict(X), record['estimator'].predict(X),
                                                       rtol=1e-6, atol=1e-5)
                            if experiment == 14:
                                self.assertEqual(loaded.predict(X).shape, (2, 7))
                                self.assertEqual(record['estimator'].get_params()['multi_strategy'],
                                                 'multi_output_tree')

    def test_early_stopping_only_sees_purged_training_tail(self):
        calls = []

        class RecordingModel(MeanModel):
            best_iteration = 1

            def fit(self, X, y, **kwargs):
                if 'eval_set' in kwargs:
                    stop_X, _ = kwargs['eval_set'][0]
                    self_outer.assertLess(X.x.max() + 7, stop_X.x.min())
                    self_outer.assertLess(stop_X.x.max() + 7, 366)
                    calls.append(len(X))
                return super().fit(X, y, **kwargs)

        self_outer = self
        with patch.object(residual.base, '_xgb', lambda *a, **kw: RecordingModel()):
            for experiment in [13, 14]:
                context = residual.make_context(self.frame, [1, 2],
                                                residual.base.make_spec(experiment, 42), 'validation')
                result = residual.fit_evaluate(context, {}, 5, 2, 42, {'total': 0}, True)
                self.assertTrue(all(v <= 5 for v in result['selected_rounds'].values()))
        self.assertTrue(calls)

    def test_hierarchical_search_uses_combined_residual_evaluator(self):
        base = residual.base
        context = residual.make_context(self.frame, [1, 2], base.make_spec(14, 42), 'validation')
        args = base.parse_args(['--experiment', '14', '--development',
                                '--max-phase1-candidates', '1', '--max-phase2-candidates', '1',
                                '--max-phase3-candidates', '1'])
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(base, '_synchronize_cuda'), \
             patch.object(base, 'fit_evaluate', side_effect=AssertionError('Global-only evaluator used')):
            _, summary, trials, _, _ = base.hierarchical_search(context, args, Path(directory))
        self.assertEqual(len(trials), 7)
        self.assertTrue(all(row['status'] == 'ok' for row in trials))
        self.assertGreater(summary['total_models_trained'], 7)

    def test_cli_pipeline_saves_both_splits_and_seed(self):
        base = residual.base
        for experiment in [13, 14]:
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(residual, 'cpu_preflight', return_value={'requested_device': 'cpu'}), \
                 patch.object(base.runner, 'prepare_dataframe', return_value=self.frame), \
                 patch.object(base.data, 'resolve_stations', return_value=[1, 2]), \
                 patch.object(base, '_artifact_dir', return_value=Path(directory)), \
                 patch.object(base, '_synchronize_cuda'), \
                 patch.object(base, 'MAX_BOOSTING_ROUNDS', 2), \
                 patch.object(base, 'PHASE1_BUDGETS', (2,)), \
                 patch.object(base, 'PHASE2_BUDGETS', (2, 2)), \
                 patch.object(base, 'PHASE3_BUDGETS', (2, 2)):
                self.assertEqual(residual.main(['--experiment', str(experiment), '--train',
                    '--device', 'cpu', '--seed', '17', '--max-phase1-candidates', '1',
                    '--max-phase2-candidates', '1', '--max-phase3-candidates', '1']), 0)
                root = Path(directory)
                saved = json.loads((root / 'config.json').read_text())
                self.assertEqual(saved['seed'], 17)
                for split in ['validation', 'test']:
                    self.assertTrue((root / f'residual_oof_folds_{split}.csv').exists())
                    self.assertTrue(list((root / 'model' / split).glob('local_residual*')))
                self.assertTrue((root / 'metrics_overall_test.json').exists())


if __name__ == '__main__':
    unittest.main()
