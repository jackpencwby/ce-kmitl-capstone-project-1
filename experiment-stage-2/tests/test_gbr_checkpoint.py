import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class CheckpointTests(unittest.TestCase):
    def cache_type(self):
        self.assertIsNotNone(importlib.util.find_spec('gbr_checkpoint'), 'GBR checkpoint feature is missing')
        from gbr_checkpoint import Checkpoint
        return Checkpoint

    def test_completed_fit_reused_and_predictions_equal(self):
        Cache = self.cache_type()
        X = np.arange(80.).reshape(40, 2); y = np.sin(X[:, 0])
        count = []
        def train():
            count.append(1)
            return GradientBoostingRegressor(n_estimators=3, random_state=7).fit(X, y)
        with tempfile.TemporaryDirectory() as d:
            with Cache(d, {'seed': 7}) as c:
                expected = c.fit('h1', train).predict(X)
            with Cache(d, {'seed': 7}, resume=True) as c:
                actual = c.fit('h1', train).predict(X)
            self.assertEqual(len(count), 1)
            np.testing.assert_array_equal(actual, expected)

    def test_changed_identity_corruption_and_duplicate_writer_rejected(self):
        Cache = self.cache_type()
        with tempfile.TemporaryDirectory() as d:
            with Cache(d, {'seed': 7}) as c:
                c.fit('h1', lambda: {'done': True})
                with self.assertRaises(Exception):
                    with Cache(d, {'seed': 7}, resume=True): pass
            with self.assertRaises(ValueError):
                with Cache(d, {'seed': 8}, resume=True): pass
            model = next((Path(d)/'checkpoint/models').glob('*.joblib'))
            model.write_bytes(b'broken')
            with Cache(d, {'seed': 7}, resume=True) as c:
                with self.assertRaises(ValueError): c.fit('h1', lambda: None)

    def test_unfinished_fit_retries_and_saved_trial_skips(self):
        Cache = self.cache_type()
        with tempfile.TemporaryDirectory() as d:
            with Cache(d, {'seed': 7}) as c:
                with self.assertRaises(RuntimeError): c.fit('bad', lambda: (_ for _ in ()).throw(RuntimeError()))
                c.save_trial('trial1', {'score': 1.25})
            with Cache(d, {'seed': 7}, resume=True) as c:
                self.assertEqual(c.fit('bad', lambda: 42), 42)
                self.assertEqual(c.load_trial('trial1'), {'score': 1.25})

if __name__ == '__main__': unittest.main()

class RunnerResumeTests(unittest.TestCase):
    def test_all_model_types_resume_without_refitting(self):
        import run_stage2_gradient_boosting as r
        self.assertTrue(hasattr(r, '_fit_model'), 'Runner checkpoint integration is missing')
        from gbr_checkpoint import Checkpoint
        from unittest.mock import patch
        X=np.arange(80.).reshape(40,2)
        with tempfile.TemporaryDirectory() as d:
            with Checkpoint(d, {'test':1}) as c:
                for joint in (False, True):
                    y=np.tile(np.sin(X[:,0,None]),(1,7)) if joint else np.sin(X[:,0])
                    m=r._model({'max_depth':2},joint,42,3)
                    first=r._fit_model(m,X,y,c).predict(X)
                    other=r._model({'max_depth':2},joint,42,3)
                    with patch.object(type(other),'fit',side_effect=AssertionError('unexpected refit')):
                        second=r._fit_model(other,X,y,c).predict(X)
                    np.testing.assert_array_equal(first,second)
    def test_signal_defers_exit_until_safe_boundary(self):
        import run_stage2_gradient_boosting as r
        self.assertTrue(hasattr(r, '_request_stop'), 'Safe signal handling is missing')
        r._request_stop(10,None)
        try:
            with self.assertRaises(SystemExit) as caught: r._check_stop()
            self.assertEqual(caught.exception.code,75)
        finally: r._STOP_REQUESTED=False

class EndToEndResumeTests(unittest.TestCase):
    def test_all_six_resume_export_and_completed_run(self):
        import pandas as pd
        import run_stage2_gradient_boosting as r
        from unittest.mock import patch
        from contextlib import ExitStack
        from gbr_checkpoint import Checkpoint
        frame=pd.DataFrame({'station_id':1,'date':pd.date_range('2024-01-01',periods=450),'x':np.arange(450.),'pm25':np.arange(450.),'region_id':1})
        for h in range(1,8): frame[f'target_t{h}']=frame.x+h
        with ExitStack() as stack:
            stack.enter_context(patch.object(r.config,'VALIDATION_START','2025-01-01'))
            stack.enter_context(patch.object(r.config,'TEST_START','2025-02-01'))
            stack.enter_context(patch.object(r.models,'assemble_feature_columns',return_value=['x']))
            stack.enter_context(patch.object(r.models,'_add_categorical_codes',side_effect=lambda f,*a:(f,['x'])))
            stack.enter_context(patch.object(r.runner,'prepare_dataframe',return_value=frame))
            for exp in (5,6,11,12,17,18):
                with self.subTest(exp=exp), tempfile.TemporaryDirectory() as d:
                    args=['--experiment',str(exp),'--train','--source','local','--checkpoint','--run-dir',d,'--max-estimators','2','--max-phase1-candidates','1','--max-phase2-candidates','1','--max-phase3-candidates','1']
                    original=r._write_result
                    def stop_at_test(*a,**kw):
                        if a[-1]=='test': raise SystemExit(75)
                        return original(*a,**kw)
                    with patch.object(r,'_write_result',side_effect=stop_at_test):
                        with self.assertRaises(SystemExit): r.main(args)
                    with patch.object(GradientBoostingRegressor,'fit',side_effect=AssertionError('refit')), patch.object(r.JointMultiOutputGradientBoostingRegressor,'fit',side_effect=AssertionError('refit')), patch.object(r,'cpu_preflight',return_value={}):
                        self.assertEqual(r.main(args+['--resume',d]),0)
                        self.assertEqual(r.main(args+['--resume',d]),0)
                    self.assertTrue((Path(d)/'checkpoint/completed.json').exists())
                    self.assertTrue(list((Path(d)/'models/test').glob('*.joblib')))

class CorruptionTests(unittest.TestCase):
    def test_malformed_marker_aborts_and_train_error_remains_train_error(self):
        from gbr_checkpoint import Checkpoint, CheckpointError
        for content in ('null','[]','{}'):
            with tempfile.TemporaryDirectory() as d:
                with Checkpoint(d,{'x':1}) as c:
                    c.fit('x',lambda:42)
                    next((Path(d)/'checkpoint/models').glob('*.json')).write_text(content)
                    with self.assertRaises(CheckpointError): c.fit('x',lambda:43)
                    with self.assertRaises(ValueError) as caught:
                        c.fit('y',lambda: (_ for _ in ()).throw(ValueError('training error')))
                    self.assertNotIsInstance(caught.exception,CheckpointError)
