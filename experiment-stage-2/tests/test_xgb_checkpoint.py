"""Resume must preserve predictions and avoid repeating completed model fits."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_xgb_residual as residual


class CheckpointTests(unittest.TestCase):
    def summary(self):
        return {'score': 1.5, 'best_iterations': {'global:1': 2}, 'best_scores': {},
                'selected_rounds': {'global:1': 3},
                'rmse_by_horizon': {f'rmse_t{h}': 1.5 for h in range(1, 8)}}

    def test_cli_resume_reuses_run_directory_and_finishes_both_splits(self):
        self.checkpoint_class()
        frame = pd.DataFrame({'station_id': 1, 'date': pd.date_range('2024-01-01', periods=450),
                              'x': np.arange(450, dtype=float)})
        for h in range(1, 8):
            frame[f'target_t{h}'] = frame.x + h
        base = residual.base
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(base.config, 'VALIDATION_START', '2025-01-01'), \
             patch.object(base.config, 'TEST_START', '2025-02-01'), \
             patch.object(base.config, 'baseline_feature_list', lambda: ['x']), \
             patch.object(base.runner, 'prepare_dataframe', return_value=frame), \
             patch.object(base.data, 'resolve_stations', return_value=[1]), \
             patch.object(base, '_artifact_dir', return_value=Path(directory)), \
             patch.object(base, '_synchronize_cuda'), \
             patch.object(base, 'MAX_BOOSTING_ROUNDS', 2), \
             patch.object(base, 'PHASE1_BUDGETS', (2,)), \
             patch.object(base, 'PHASE2_BUDGETS', (2, 2)), \
             patch.object(base, 'PHASE3_BUDGETS', (2, 2)):
            args = ['--experiment', '14', '--train', '--device', 'cpu', '--checkpoint',
                    '--max-phase1-candidates', '1', '--max-phase2-candidates', '1',
                    '--max-phase3-candidates', '1']
            real_fit = residual._fit
            calls = []

            def interrupt(*a):
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                calls.append(1)
                return real_fit(*a)

            with patch.object(residual, '_fit', interrupt), self.assertRaises(KeyboardInterrupt):
                residual.main(args)
            root = Path(directory)
            self.assertTrue((root / 'checkpoint/manifest.json').exists())
            self.assertEqual(residual.main(args + ['--resume', directory]), 0)
            for split in ('validation', 'test'):
                suffix = '' if split == 'validation' else '_test'
                self.assertTrue((root / f'metrics_overall{suffix}.json').exists())
                self.assertTrue(list((root / 'model' / split).glob('*.json')))
            with patch.object(residual, '_fit', side_effect=AssertionError('finished run retrained')):
                self.assertEqual(residual.main(args + ['--resume', directory]), 0)

    def checkpoint_class(self):
        self.assertTrue(hasattr(residual, 'RunCheckpoint'),
                        'Residual runner needs a persistent resume checkpoint')
        return residual.RunCheckpoint

    def test_interrupt_then_resume_preserves_predictions_and_skips_fits(self):
        cls = self.checkpoint_class()
        frame = pd.DataFrame({'station_id': 1, 'date': pd.date_range('2024-01-01', periods=450),
                              'x': np.arange(450, dtype=float)})
        for h in range(1, 8):
            frame[f'target_t{h}'] = frame.x + h
        with patch.object(residual.base.config, 'VALIDATION_START', '2025-01-01'), \
             patch.object(residual.base.config, 'TEST_START', '2025-02-01'), \
             patch.object(residual.base.config, 'baseline_feature_list', lambda: ['x']), \
             tempfile.TemporaryDirectory() as directory:
            for experiment in (13, 14):
                spec = residual.base.make_spec(experiment, 42)
                context = residual.make_context(frame, [1], spec, 'validation')
                context['device'] = 'cpu'
                params = {'n_jobs': 1, 'max_depth': 2}
                baseline_count = {'total': 0}
                baseline = residual.fit_evaluate(context, params, 2, None, 42, baseline_count, True)
                location = Path(directory) / str(experiment)
                real_fit = residual._fit
                completed = []
                stop_after = 6 if experiment == 13 else 2

                def interrupt(*args):
                    if len(completed) == stop_after:
                        raise KeyboardInterrupt('simulated preemption')
                    result = real_fit(*args)
                    completed.append(result)
                    return result

                with cls(location, {'experiment': experiment}) as checkpoint:
                    context['checkpoint'] = checkpoint
                    with patch.object(residual, '_fit', interrupt), self.assertRaises(KeyboardInterrupt):
                        residual.fit_evaluate(context, params, 2, None, 42, {'total': 0}, True)
                resumed_count = {'total': 0}
                with cls(location, {'experiment': experiment}, resume=True) as checkpoint:
                    context['checkpoint'] = checkpoint
                    resumed = residual.fit_evaluate(context, params, 2, None, 42, resumed_count, True)
                    self.assertEqual(resumed_count['total'], baseline_count['total'] - stop_after)
                    np.testing.assert_allclose(resumed['predictions'].y_pred,
                                               baseline['predictions'].y_pred, rtol=1e-6, atol=1e-5)
                    self.assertEqual(resumed['selected_rounds'], baseline['selected_rounds'])
                    self.assertEqual(len(resumed['model_records']), len(baseline['model_records']))

    def test_manifest_mismatch_and_missing_resume_are_rejected(self):
        cls = self.checkpoint_class()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                with cls(root, {'seed': 42}, resume=True):
                    pass
            with cls(root, {'seed': 42}):
                pass
            with self.assertRaises(ValueError):
                with cls(root, {'seed': 43}, resume=True):
                    pass
            with self.assertRaises(ValueError):
                with cls(root, {'seed': 42}):
                    pass

    def test_concurrent_writer_is_rejected(self):
        cls = self.checkpoint_class()
        with tempfile.TemporaryDirectory() as directory:
            with cls(Path(directory), {'seed': 42}):
                with self.assertRaises(RuntimeError):
                    with cls(Path(directory), {'seed': 42}, resume=True):
                        pass

    def test_completed_candidate_summary_survives_restart(self):
        cls = self.checkpoint_class()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with cls(root, {}) as checkpoint:
                scope = checkpoint.scope({'params': {'max_depth': 2}})
                scope.save_summary(self.summary())
            with cls(root, {}, resume=True) as checkpoint:
                self.assertEqual(checkpoint.scope({'params': {'max_depth': 2}}).load_summary()['score'], 1.5)
                self.assertIsNone(checkpoint.scope({'params': {'max_depth': 3}}).load_summary())

    def test_uncommitted_model_is_retrained_and_corruption_is_rejected(self):
        cls = self.checkpoint_class()
        import xgboost as xgb
        X = np.arange(24, dtype=float).reshape(12, 2)
        calls = []

        def train():
            calls.append(1)
            return xgb.XGBRegressor(n_estimators=2, n_jobs=1).fit(X, X[:, 0]), 2

        with tempfile.TemporaryDirectory() as directory:
            with cls(Path(directory), {}) as checkpoint:
                scope = checkpoint.scope({'candidate': 1})
                scope.fit('global:1', train, 'cpu')
                marker = next((scope.root / 'models').glob('*.json'))
                marker.unlink()  # model exists but its atomic completion marker never committed
                scope.fit('global:1', train, 'cpu')
                self.assertEqual(len(calls), 2)
                next((scope.root / 'models').glob('*.ubj')).write_bytes(b'truncated')
                with self.assertRaisesRegex(ValueError, 'corrupt'):
                    scope.fit('global:1', train, 'cpu')
                self.assertEqual(len(calls), 2)

    def test_identity_changes_with_data_and_seed_but_not_resume_path(self):
        self.checkpoint_class()
        args = residual.base.parse_args(['--experiment', '13', '--train', '--checkpoint'])
        frame = pd.DataFrame({'x': [1., 2.], 'target': [3., 4.]})
        expected = residual.run_identity(frame, [1], args, 'cuda:0')
        args.resume = Path('somewhere')
        self.assertEqual(residual.run_identity(frame, [1], args, 'cuda:0'), expected)
        self.assertNotEqual(residual.run_identity(frame + 1, [1], args, 'cuda:0'), expected)
        args.seed += 1
        self.assertNotEqual(residual.run_identity(frame, [1], args, 'cuda:0'), expected)

    def test_corrupt_candidate_aborts_search_instead_of_changing_ranking(self):
        cls = self.checkpoint_class()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with cls(root, {}) as checkpoint:
                scope = checkpoint.scope({'candidate': 1})
                scope.save_summary(self.summary())
                context = {'candidate_evaluator': lambda *a, **kw: scope.load_summary()}
                args = residual.base.parse_args(['--experiment', '13'])
                for broken in ('{broken', '{}', 'null', '{"score": "bad"}'):
                    (scope.root / 'summary.json').write_text(broken)
                    with self.assertRaises(residual.base.CheckpointError):
                        residual.base._run_round(1, 1, [{}], 2, context, args, root, [], {'total': 0}, {})


if __name__ == '__main__':
    unittest.main()
