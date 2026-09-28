"""Verify E2 routing with pooled fits and isolation from legacy recipes."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import config, config_e2, models


class E2ParametersTests(unittest.TestCase):
    def test_factories_keep_legacy_recipes(self):
        for algorithm, factory in models.ESTIMATOR_FACTORIES.items():
            with self.subTest(algorithm=algorithm):
                legacy = factory(prefer_gpu=False)
                e2 = factory(prefer_gpu=False, parameter_profile="e2")
                if algorithm == "gbr":
                    legacy, e2 = legacy.steps[-1][1], e2.steps[-1][1]
                expected = (config.xgb_params_for_horizon(1) if algorithm == "xgboost"
                            else config.LGBM_PARAMS.as_dict() if algorithm == "lightgbm"
                            else config.GBR_PARAMS.as_dict())
                for key, value in expected.items():
                    self.assertEqual(legacy.get_params()[key], value)
                for key, value in config_e2.params_for(algorithm).items():
                    self.assertEqual(e2.get_params()[key], value)

    def test_global_e2_fits_seven_shared_models_for_each_algorithm(self):
        dates = pd.date_range("2025-01-01", periods=60)
        frame = pd.DataFrame([
            {"station_id": sid, "date": date, "x": float(i % 9),
             **{f"target_t{h}": float(i + sid + h) for h in range(1, 8)}}
            for sid in (1, 2) for i, date in enumerate(dates)
        ])
        original = config_e2.params_for
        def small_params(algorithm):
            return {**original(algorithm), "n_estimators": 2}
        with patch.object(config, "VALIDATION_START", "2025-02-10"), \
             patch.object(config, "TEST_START", "2025-03-01"), \
             patch.object(config, "baseline_feature_list", return_value=["x"]), \
             patch.object(config_e2, "params_for", side_effect=small_params), \
             patch.object(config, "xgb_params_for_horizon", side_effect=AssertionError("E1 recipe accessed")):
            for algorithm in ("xgboost", "lightgbm", "gbr"):
                with self.subTest(algorithm=algorithm):
                    spec = models.RunSpec("E2_TEST", algorithm=algorithm,
                        training_strategy="global", include_station_id=True,
                        parameter_profile="e2", prefer_gpu=False)
                    result = models.run_holdout(spec, frame, [1, 2])
                    self.assertEqual(len(result["model_records"]), 7)
                    self.assertEqual({r["group"] for r in result["model_records"]}, {"global"})
                    self.assertIn("station_id_code", result["feature_list"])
                    self.assertEqual(set(result["predictions"].station_id), {1, 2})


if __name__ == "__main__":
    unittest.main()
