"""Device selection must preserve Jack's scalar and horizon-stacked design."""
import sys
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jack_run_stage2 as stage2


class DeviceTests(unittest.TestCase):
    def test_gpu_cli_and_run_identity(self):
        for experiment in (15, 16):
            args = stage2.parse_args(['--experiment', str(experiment), '--device', 'gpu'])
            spec = stage2.make_spec(experiment, device=args.device)
            self.assertTrue(spec.prefer_gpu)
            self.assertIn('LGBM_GPU', spec.run_id)
            self.assertEqual(spec.training_strategy, 'global_local_tree')

    def test_opencl_ids_reach_spec(self):
        args = stage2.parse_args(['--experiment', '16', '--device', 'gpu',
                                  '--gpu-platform-id', '1', '--gpu-device-id', '0'])
        spec = stage2.make_spec(16, device=args.device,
                               gpu_platform_id=args.gpu_platform_id, gpu_device_id=args.gpu_device_id)
        self.assertEqual((spec.gpu_platform_id, spec.gpu_device_id), (1, 0))

    def test_explicit_device_survives_factory(self):
        for device in ('cpu', 'gpu'):
            estimator = stage2.make_cpu_lgbm({'device_type': device, 'n_estimators': 2})
            self.assertEqual(estimator.get_params()['device_type'], device)

    def test_failed_gpu_probe_is_actionable_and_never_falls_back(self):
        with patch('lightgbm.LGBMRegressor.fit', side_effect=RuntimeError('GPU unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'gpu'):
                stage2.require_lightgbm('gpu')

    @unittest.skipUnless(os.environ.get('RUN_LIGHTGBM_GPU_TESTS') == '1', 'Opt-in real GPU integration')
    def test_real_gpu_residual_models_and_reload(self):
        import joblib
        import numpy as np
        import pandas as pd
        from test_residual import ResidualTests

        real_factory = stage2.make_cpu_lgbm
        fixture = ResidualTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)

        def gpu_factory(params, **kwargs):
            return real_factory(params, **kwargs).set_params(n_jobs=2)

        with patch.object(stage2, 'make_cpu_lgbm', gpu_factory):
            for experiment in (15, 16):
                spec = stage2.make_spec(experiment, device='gpu',
                                       gpu_platform_id=int(os.environ.get('LIGHTGBM_GPU_PLATFORM_ID', '-1')),
                                       gpu_device_id=0)
                selected = None
                for split in ('validation', 'test'):
                    with self.subTest(experiment=experiment, split=split):
                        result = stage2.run_pooled(spec, fixture.frame, [1, 2],
                                                  {'num_leaves': 3, 'min_child_samples': 5},
                                                  split, budget=5, early_stopping_rounds=2,
                                                  fit_rounds=selected)
                        selected = result['fit_rounds']
                        records = result['model_records']
                        self.assertEqual(len(records), 21 if experiment == 15 else 3)
                        self.assertTrue(np.isfinite(result['predictions'].y_pred).all())
                        for record in records:
                            model = record['estimator']
                            self.assertEqual(model.booster_.params['device_type'], 'gpu')
                            if experiment == 16:
                                self.assertIn(stage2.HORIZON_FEATURE, model.feature_name_)
                        with tempfile.TemporaryDirectory() as directory:
                            stage2.artifacts.save_models(Path(directory), records, split, result['feature_list'])
                            paths = list(Path(directory).rglob('*.joblib'))
                            self.assertEqual(len(paths), len(records))
                            for path in paths:
                                loaded = joblib.load(path)
                                self.assertEqual(loaded.booster_.params['device_type'], 'gpu')
                                original = next(r['estimator'] for r in records
                                                if path.stem == f"{r['role']}__{r['group']}__h{r['horizon']}")
                                X = pd.DataFrame(np.ones((2, len(loaded.feature_name_))),
                                                 columns=loaded.feature_name_)
                                np.testing.assert_allclose(loaded.predict(X), original.predict(X))


if __name__ == '__main__':
    unittest.main()
