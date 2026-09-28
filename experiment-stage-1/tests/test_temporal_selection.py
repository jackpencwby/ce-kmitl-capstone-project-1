from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import training


class TemporalSelectionTests(unittest.TestCase):
    def test_selection_uses_only_train_tail_then_refits_full_training(self):
        fits = []

        class Estimator:
            def __init__(self):
                self.params = {'n_estimators':20}
                self.best_iteration = 2
            def get_params(self):
                return self.params.copy()
            def set_params(self, **kwargs):
                self.params.update(kwargs)
                return self
            def fit(self, X, y, **kwargs):
                fits.append((X.index.tolist(), y.to_list(), kwargs, self.params.copy()))
                return self
            def predict(self, X):
                return np.zeros(len(X))

        dates = pd.Series(pd.date_range('2025-01-01',periods=100))
        X = pd.DataFrame({'x':np.arange(100)})
        y = pd.Series(np.arange(100))
        fitted = training.fit_temporal_selection(lambda prefer_gpu:Estimator(),False,
            'xgboost',X,y,dates,horizon=7,patience=2)
        self.assertEqual(len(fits),2)
        train_idx, _, kwargs, _ = fits[0]
        eval_X, eval_y = kwargs['eval_set'][-1]
        self.assertLess(dates.loc[train_idx].max()+pd.Timedelta(days=7), dates.loc[eval_X.index].min())
        self.assertEqual(fits[1][0],list(range(100)))
        self.assertEqual(fits[1][3]['n_estimators'],3)
        self.assertIsNone(fits[1][3]['early_stopping_rounds'])
        self.assertEqual(fitted.selection_metadata_['selected_n_estimators'],3)

    def test_small_windows_record_fixed_budget_fallback(self):
        from xgboost import XGBRegressor
        X=pd.DataFrame({'x':np.arange(12,dtype=float)})
        y=pd.Series(np.arange(12,dtype=float))
        model=training.fit_temporal_selection(
            lambda prefer_gpu:XGBRegressor(n_estimators=3,n_jobs=1),False,'xgboost',
            X,y,pd.Series(pd.date_range('2025-01-01',periods=12)),horizon=1,patience=2)
        self.assertEqual(model.selection_metadata_['status'],'insufficient_history')
        self.assertEqual(model.get_params()['n_estimators'],3)

    def test_real_xgb_and_lgbm_refits_are_predictable(self):
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        X=pd.DataFrame({'x':np.arange(110,dtype=float)})
        y=pd.Series(np.sin(np.arange(110)/8))
        dates=pd.Series(pd.date_range('2025-01-01',periods=110))
        for algorithm,factory in [
            ('xgboost',lambda prefer_gpu:XGBRegressor(n_estimators=12,n_jobs=1,max_depth=2)),
            ('lightgbm',lambda prefer_gpu:LGBMRegressor(n_estimators=12,n_jobs=1,max_depth=2,verbosity=-1))]:
            with self.subTest(algorithm=algorithm):
                model=training.fit_temporal_selection(factory,False,algorithm,X,y,dates,7,2)
                self.assertEqual(model.selection_metadata_['status'],'selected_and_refit')
                self.assertTrue(1<=model.selection_metadata_['selected_n_estimators']<=12)
                self.assertTrue(np.isfinite(model.predict(X)).all())


if __name__=='__main__':
    unittest.main()
