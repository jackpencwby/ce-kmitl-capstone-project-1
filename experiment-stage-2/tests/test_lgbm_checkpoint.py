"""Interruptions must not repeat committed LightGBM fits or change predictions."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jack_run_stage2 as runner


class ResumeTests(unittest.TestCase):
    def checkpoint_class(self):
        self.assertTrue(hasattr(runner, 'LGBMCheckpoint'), 'LightGBM checkpoint support is missing')
        return runner.LGBMCheckpoint

    def test_interrupted_fits_resume_with_identical_predictions(self):
        cls = self.checkpoint_class()
        frame = pd.DataFrame({'station_id': 1, 'date': pd.date_range('2024-01-01', periods=450),
                              'x': np.sin(np.arange(450) / 25)})
        for h in range(1, 8):
            frame[f'target_t{h}'] = frame.x + h
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner.config, 'VALIDATION_START', '2025-01-01'), \
             patch.object(runner.config, 'TEST_START', '2025-02-01'), \
             patch.object(runner.config, 'baseline_feature_list', lambda: ['x']):
            for experiment in (15, 16):
                spec = runner.make_spec(experiment, 42, 'cpu')
                params = {'num_leaves': 4, 'min_child_samples': 5}
                baseline = runner.run_pooled(spec, frame, [1], params, 'validation', budget=3)
                real = runner._fit_residual_lgbm
                calls = []
                def interrupted(*a, **kw):
                    if len(calls) == 2:
                        raise KeyboardInterrupt('preemption')
                    calls.append(1)
                    return real(*a, **kw)
                path = Path(directory) / str(experiment)
                with cls(path, {'experiment': experiment}) as checkpoint:
                    with patch.object(runner, '_fit_residual_lgbm', interrupted), self.assertRaises(KeyboardInterrupt):
                        runner.run_pooled(spec, frame, [1], params, 'validation', budget=3, checkpoint=checkpoint)
                self.assertEqual(len(list((path / 'checkpoint').rglob('*.txt'))), 2)
                with cls(path, {'experiment': experiment}, resume=True) as checkpoint:
                    resumed = runner.run_pooled(spec, frame, [1], params, 'validation', budget=3, checkpoint=checkpoint)
                    np.testing.assert_allclose(baseline['predictions'].y_pred, resumed['predictions'].y_pred, rtol=0, atol=1e-12)
                    self.assertEqual(baseline['fit_rounds'], resumed['fit_rounds'])
                    with patch.object(runner, '_fit_residual_lgbm', side_effect=AssertionError('fit repeated')):
                        reused = runner.run_pooled(spec, frame, [1], params, 'validation', budget=3, checkpoint=checkpoint)
                    np.testing.assert_allclose(resumed['predictions'].y_pred, reused['predictions'].y_pred)
                    # Candidate cache survives deletion of its redundant model files.
                    scored = runner.run_pooled(spec, frame, [1], params, 'validation', budget=3,
                                               checkpoint=checkpoint, cache_summary=True)
                    with patch.object(runner, '_fit_residual_lgbm', side_effect=AssertionError('candidate repeated')):
                        cached = runner.run_pooled(spec, frame, [1], params, 'validation', budget=3,
                                                   checkpoint=checkpoint, cache_summary=True)
                    self.assertEqual(runner.primary_score(scored), runner.primary_score(cached))

    def test_cli_resumes_final_exports_and_completed_run(self):
        self.checkpoint_class()
        frame = pd.DataFrame({'station_id': 1, 'date': pd.date_range('2024-01-01', periods=450),
                              'x': np.arange(450, dtype=float), 'pm25': np.arange(450, dtype=float)})
        for h in range(1, 8):
            frame[f'target_t{h}'] = frame.x + h
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner.config, 'VALIDATION_START', '2025-01-01'), \
             patch.object(runner.config, 'TEST_START', '2025-02-01'), \
             patch.object(runner.config, 'baseline_feature_list', lambda: ['x']), \
             patch.object(runner.runner, 'prepare_dataframe', return_value=frame), \
             patch.object(runner.data, 'resolve_stations', return_value=[1]), \
             patch.object(runner, 'PHASE_BUDGETS', {1: (2,), 2: (2,), 3: (2,)}):
            args = ['--experiment', '16', '--train', '--device', 'cpu', '--checkpoint',
                    '--run-dir', directory, '--max-phase1-candidates', '1',
                    '--max-phase2-candidates', '1', '--max-phase3-candidates', '1']
            real = runner.save_split
            def interrupt_test(*a, **kw):
                if a[-1] == 'test':
                    raise KeyboardInterrupt()
                return real(*a, **kw)
            with patch.object(runner, 'save_split', interrupt_test), self.assertRaises(KeyboardInterrupt):
                runner.main(args)
            self.assertFalse((Path(directory) / 'checkpoint/completed.json').exists())
            with patch.object(runner, '_fit_residual_lgbm', side_effect=AssertionError('completed model retrained')):
                self.assertEqual(runner.main(args + ['--resume', directory]), 0)
                self.assertEqual(runner.main(args + ['--resume', directory]), 0)
            for split in ('validation', 'test'):
                self.assertTrue(list((Path(directory) / 'model' / split).glob('*.joblib')))
            self.assertTrue((Path(directory) / 'checkpoint/completed.json').exists())
            from xgb_checkpoint import CheckpointError
            marker = Path(directory) / 'checkpoint/completed.json'
            valid = marker.read_text()
            marker.write_text('{broken')
            with self.assertRaises(CheckpointError):
                runner.main(args + ['--resume', directory])
            marker.write_text(valid)
            (Path(directory) / 'metrics_overall_test.json').write_text('{}')
            with self.assertRaises(CheckpointError):
                runner.main(args + ['--resume', directory])

    def test_corrupt_checkpoint_aborts_instead_of_skipping_candidate(self):
        self.checkpoint_class()
        from xgb_checkpoint import CheckpointError
        args = runner.parse_args(['--experiment', '15'])
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(runner, 'run_pooled', side_effect=CheckpointError('corrupt')):
            with self.assertRaises(CheckpointError):
                runner._run_round(1, 1, [{}], 3, runner.make_spec(15), pd.DataFrame(), [1], args,
                                  Path(directory), [], {'models': 0})

    def test_native_model_corruption_and_uncommitted_model(self):
        cls = self.checkpoint_class()
        import lightgbm as lgb
        from xgb_checkpoint import CheckpointError
        X = np.arange(200, dtype=float).reshape(100, 2)
        calls = []
        def train():
            calls.append(1)
            return lgb.LGBMRegressor(n_estimators=2, n_jobs=1, verbosity=-1).fit(X, X[:, 0]), 2
        with tempfile.TemporaryDirectory() as directory, cls(Path(directory), {}) as checkpoint:
            scope = checkpoint.scope({'candidate': 1})
            scope.fit('global', train, 'cpu')
            next((scope.root / 'models').glob('*.json')).unlink()
            scope.fit('global', train, 'cpu')
            self.assertEqual(len(calls), 2)
            next((scope.root / 'models').glob('*.txt')).write_bytes(b'corrupt')
            with self.assertRaises(CheckpointError):
                scope.fit('global', train, 'cpu')
            self.assertEqual(len(calls), 2)

    def test_resume_rejects_changed_data_and_options(self):
        cls = self.checkpoint_class()
        from lgbm_checkpoint import lightgbm_identity
        args = runner.parse_args(['--experiment', '15', '--train', '--checkpoint'])
        frame = pd.DataFrame({'x': [1., 2.]})
        identity = lightgbm_identity(frame, [1], args)
        with tempfile.TemporaryDirectory() as directory:
            with cls(directory, identity):
                pass
            args.resume = Path(directory)
            args.run_dir = Path(directory)
            with cls(directory, lightgbm_identity(frame, [1], args), resume=True):
                pass
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                with cls(directory, lightgbm_identity(frame + 1, [1], args), resume=True):
                    pass
            args.seed += 1
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                with cls(directory, lightgbm_identity(frame, [1], args), resume=True):
                    pass


if __name__ == '__main__':
    unittest.main()
