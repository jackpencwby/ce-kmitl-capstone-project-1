"""Behavioral regressions for complete cohorts, temporal CV and tree selection."""
from pathlib import Path
import sys
import unittest
import tempfile
import json
import runpy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config, models, runner, features, data


class ZeroEstimator:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


def sample_frame():
    d = pd.DataFrame({'station_id': 1, 'date': pd.date_range('2025-01-01', periods=90),
                      'x': np.arange(90, dtype=float), 'pm25': 10., 'segment_id': 1})
    for h in range(1, 8):
        d[f'target_t{h}'] = d.x + h
    return d


class E23CorrectnessTests(unittest.TestCase):
    def test_no_eligible_stations_reports_actionable_error(self):
        d = sample_frame()
        args = SimpleNamespace(prefer_cpu=True, check_gcs=False, source='local',
            stations=['all'], max_stations=None, train=True, upload_gcs=False)
        with patch.object(runner, 'prepare_dataframe', return_value=d), \
             patch.object(data, 'resolve_stations', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'No eligible stations'):
                runner.execute(models.RunSpec('E2'), args)

    def test_missing_early_origins_cannot_move_training_label_boundary(self):
        d = sample_frame()
        d.loc[d.date.between('2025-02-01', '2025-02-07') |
              d.date.between('2025-03-01', '2025-03-07'), 'target_t7'] = np.nan
        fitted_labels = []

        class Recorder(ZeroEstimator):
            def fit(self, X, y):
                fitted_labels.append(float(y.max()))
                return self

        with patch.object(config, 'VALIDATION_START', '2025-02-01'), \
             patch.object(config, 'TEST_START', '2025-03-01'), \
             patch.object(config, 'baseline_feature_list', return_value=['x']), \
             patch.dict(models.ESTIMATOR_FACTORIES, {'xgboost': lambda prefer_gpu: Recorder()}):
            for split, start in [('validation', '2025-02-01'), ('test', '2025-03-01')]:
                for strategy in ['direct', 'multi']:
                    with self.subTest(split=split, strategy=strategy):
                        fitted_labels.clear()
                        models.run_holdout(models.RunSpec('E3', forecast_strategy=strategy,
                            complete_target_evaluation=True), d, [1], evaluation=split)
                        self.assertEqual(len(fitted_labels), 7)
                        self.assertLess(max(fitted_labels),
                            (pd.Timestamp(start)-pd.Timestamp('2025-01-01')).days)

    def test_e3_common_support_requires_complete_training_vectors(self):
        d = sample_frame()
        d.loc[d.date < '2025-02-01', 'target_t7'] = np.nan
        with patch.object(config, 'VALIDATION_START', '2025-02-01'), \
             patch.object(config, 'TEST_START', '2025-03-01'), \
             patch.object(config, 'baseline_feature_list', return_value=['x']), \
             patch.dict(models.ESTIMATOR_FACTORIES, {'xgboost': lambda prefer_gpu: ZeroEstimator()}):
            for strategy in ['direct', 'multi']:
                result = models.run_holdout(models.RunSpec('E3', forecast_strategy=strategy,
                    complete_target_evaluation=True), d, [1])
                self.assertTrue(result['predictions'].empty)

    def test_empty_validation_still_saves_artifacts(self):
        from common import artifacts
        d = sample_frame()
        d.loc[d.date < '2025-02-01', 'target_t7'] = np.nan
        args = SimpleNamespace(prefer_cpu=True, check_gcs=False, source='local',
            stations=['all'], max_stations=None, train=True, upload_gcs=False)
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(runner, 'prepare_dataframe', return_value=d), \
             patch.object(config, 'VALIDATION_START', '2025-02-01'), \
             patch.object(config, 'TEST_START', '2025-03-01'), \
             patch.object(config, 'baseline_feature_list', return_value=['x']), \
             patch.object(artifacts, 'make_run_dir', return_value=Path(folder)), \
             patch.dict(models.ESTIMATOR_FACTORIES, {'xgboost': lambda prefer_gpu: ZeroEstimator()}):
            self.assertEqual(runner.execute(models.RunSpec('E3', forecast_strategy='multi',
                complete_target_evaluation=True), args), 0)
            report = json.loads((Path(folder)/'persistence_comparison_validation.json').read_text())
            self.assertEqual(report['n'], 0)
            self.assertTrue((Path(folder)/'config.json').exists())

    def test_entrypoints_enable_corrected_screening_policy(self):
        root=Path(__file__).resolve().parents[1]
        for name in ['E2_1','E2_2','E2_3','E3_1','E3_2','E4_1','E4_2','E4_3','E4_4']:
            with self.subTest(entrypoint=name), patch.object(sys,'argv',[name]), \
                 patch.object(runner,'execute',return_value=0) as execute:
                runpy.run_path(str(root/(name+'.py')))['main']()
                spec=execute.call_args.args[0]
                self.assertTrue(spec.walk_forward)
                self.assertEqual(spec.complete_target_evaluation,name.startswith('E3'))
                self.assertEqual(spec.selection_patience, None if name=='E2_3' else 100)

    def test_runner_saves_real_cv_paired_reports_and_refitted_models(self):
        from xgboost import XGBRegressor
        from common import artifacts
        d=pd.DataFrame({'station_id':1,'date':pd.date_range('2025-01-01',periods=200),
                        'x':np.arange(200,dtype=float),'pm25':10.})
        for h in [1,2]:
            d[f'target_t{h}']=10.+np.sin((d.x+h)/8)
        d.loc[175,'pm25']=np.nan
        args=SimpleNamespace(prefer_cpu=True,check_gcs=False,source='local',stations=['all'],
                             max_stations=None,train=True,upload_gcs=False)
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(runner,'prepare_dataframe',return_value=d), \
             patch.object(config,'VALIDATION_START','2025-04-15'), \
             patch.object(config,'TEST_START','2025-06-01'), \
             patch.object(config,'FORECAST_HORIZONS',(1,2)), \
             patch.object(config,'baseline_feature_list',return_value=['x']), \
             patch.object(artifacts,'make_run_dir',return_value=Path(folder)), \
             patch.dict(models.ESTIMATOR_FACTORIES,{'xgboost':lambda prefer_gpu:XGBRegressor(
                 n_estimators=8,max_depth=2,n_jobs=1,device='cpu')}):
            runner.execute(models.RunSpec('E2_SMOKE',walk_forward=True,selection_patience=2),args)
            p=Path(folder)
            cv=pd.read_parquet(p/'predictions_cv.parquet')
            self.assertEqual(set(cv.fold),{1,2,3,4})
            self.assertEqual(len(pd.read_csv(p/'cv_fold_metrics.csv')),4)
            selections = pd.read_csv(p/'cv_model_selection.csv')
            self.assertEqual(set(selections.fold), {1,2,3,4})
            self.assertTrue((selections.selected_n_estimators <= 8).all())
            self.assertTrue((cv.date+pd.to_timedelta(cv.horizon,unit='D')<pd.Timestamp('2025-06-01')).all())
            c=json.loads((p/'config.json').read_text())
            self.assertEqual(c['actual_device'],'cpu')
            self.assertEqual(c['evaluation_splits'],['cv','validation','test'])
            pair=json.loads((p/'persistence_comparison_test.json').read_text())
            self.assertEqual(pair['model']['micro']['n'],pair['persistence']['micro']['n'])
            t=pd.read_parquet(p/'predictions_test.parquet')
            self.assertEqual(pair['n'],len(t)-2)
            self.assertTrue((t.date>=pd.Timestamp('2025-06-01')).all())
            manifest=json.loads((p/'model/test/manifest.json').read_text())
            self.assertTrue(all(m['selection_metadata']['status']=='selected_and_refit'
                                for m in manifest['models']))

    def test_e3_complete_target_support_is_equal_in_validation_and_test(self):
        d = sample_frame()
        d.loc[[42, 76], 'target_t7'] = np.nan
        with patch.object(config, 'VALIDATION_START', '2025-02-01'), \
             patch.object(config, 'TEST_START', '2025-03-01'), \
             patch.object(config, 'baseline_feature_list', return_value=['x']), \
             patch.dict(models.ESTIMATOR_FACTORIES, {'xgboost': lambda prefer_gpu: ZeroEstimator()}):
            for split in ['validation', 'test']:
                outputs = []
                for strategy in ['direct', 'multi']:
                    spec = models.RunSpec('E3', forecast_strategy=strategy,
                                          complete_target_evaluation=True)
                    p = models.run_holdout(spec, d, [1], evaluation=split)['predictions']
                    outputs.append(p[['station_id', 'date', 'horizon', 'y_true']])
                    self.assertFalse(p.date.isin(d.loc[[42, 76], 'date']).any())
                    self.assertFalse(p.y_true.isna().any())
                    self.assertEqual(p.groupby('horizon').size().nunique(), 1)
                pd.testing.assert_frame_equal(outputs[0], outputs[1])

    def test_walk_forward_actually_refits_and_purges_each_fold_labels(self):
        d = sample_frame()
        fit_max = []

        class Recorder(ZeroEstimator):
            def fit(self, X, y):
                fit_max.append(float(X.x.max()))
                return self

        with patch.object(config, 'VALIDATION_START', '2025-02-01'), \
             patch.object(config, 'TEST_START', '2025-03-01'), \
             patch.object(config, 'FORECAST_HORIZONS', (1,)), \
             patch.object(config, 'baseline_feature_list', return_value=['x']), \
             patch.dict(models.ESTIMATOR_FACTORIES, {'xgboost': lambda prefer_gpu: Recorder()}):
            cv = models.run_walk_forward(models.RunSpec('E2'), d, [1])
        self.assertEqual(len(fit_max), 4)
        self.assertEqual(len(set(fit_max)), 4)
        self.assertEqual(set(cv['predictions'].fold), {1, 2, 3, 4})
        for row in cv['fold_metrics'].to_dict('records'):
            p = cv['predictions'].loc[cv['predictions'].fold == row['fold']]
            self.assertTrue((p.date + pd.to_timedelta(p.horizon, unit='D')
                             < pd.Timestamp(row['valid_end_exclusive'])).all())
            self.assertLess(pd.Timestamp('2025-01-01') + pd.Timedelta(days=fit_max[row['fold']-1]+1),
                            p.date.min())

    def test_lgbm_row_sampling_is_enabled(self):
        est = models.make_lgbm(prefer_gpu=False)
        self.assertEqual(est.get_params()['subsample'], .8)
        self.assertGreater(est.get_params()['subsample_freq'], 0)

    def test_spatial_uses_observed_sources_before_history_filter(self):
        d = pd.DataFrame([{'station_id':sid,'date':date,'segment_id':1,'pm25':pm,
                           'lat':lat,'long':100.,'xgboost_history_valid':not(sid==2 and i==0)}
                          for sid,pm,lat in [(1,20.,14.),(2,10.,14.1),(3,90.,13.9)]
                          for i,date in enumerate(pd.date_range('2025-01-01',periods=3))])
        with patch.object(data, 'load_master', return_value=d), \
             patch.object(features, 'build_base_features', side_effect=lambda x:x):
            result = runner.prepare_dataframe(models.RunSpec('E4',spatial_mode='unweighted'),'local')
        value = result.loc[(result.station_id==1)&(result.date==pd.Timestamp('2025-01-02')),
                           'neighbor_pm_lag_1'].item()
        self.assertEqual(value,50.)
        self.assertEqual(len(result),8)


if __name__ == '__main__':
    unittest.main()
