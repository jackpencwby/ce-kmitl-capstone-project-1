from __future__ import annotations

import unittest
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Tawan_run_stage2 as stage2


class HierarchicalSearchTests(unittest.TestCase):
    def test_horizon_target_order(self):
        self.assertEqual(stage2.TARGET_LABELS, ["t+1", "t+2", "t+3", "t+4", "t+5", "t+6", "t+7"])
        self.assertEqual(stage2.TARGET_COLS, [f"target_t{h}" for h in range(1, 8)])

    def test_candidate_counts_and_order_are_deterministic(self):
        phase1_a = stage2.phase1_candidates()
        phase1_b = stage2.phase1_candidates()
        self.assertEqual(len(phase1_a), 108)
        self.assertEqual(phase1_a, phase1_b)
        self.assertEqual(len(stage2.phase2_candidates(phase1_a[0])), 180)

        anchor = {
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.7,
            "colsample_bytree": 0.6,
            "gamma": 0.1,
            "reg_alpha": 0.01,
            "reg_lambda": 1,
        }
        phase3_a = stage2.phase3_candidates(anchor)
        phase3_b = stage2.phase3_candidates(anchor)
        self.assertEqual(len(phase3_a), 243)
        self.assertEqual(phase3_a, phase3_b)
        for candidate in phase3_a:
            for key in stage2.PHASE1_GRID:
                self.assertIn(candidate[key], stage2.ORIGINAL_DOMAIN[key])

    def test_stage2_architecture_mapping(self):
        expected = {
            1: ("global", "direct", 7),
            2: ("global", "multi", 1),
            7: ("regional", "direct", 28),
            8: ("regional", "multi", 4),
        }
        for experiment, (scope, forecast, fit_count) in expected.items():
            spec = stage2.make_spec(experiment, 42)
            self.assertEqual(spec.training_strategy, scope)
            self.assertEqual(spec.forecast_strategy, forecast)
            self.assertEqual(spec.algorithm, "xgboost")
            self.assertTrue(spec.prefer_gpu)
            self.assertTrue(spec.complete_target_evaluation)
            if forecast == "multi":
                model = stage2._xgb({}, 42, 10, 2, multi=True)
                self.assertEqual(model.get_params()["multi_strategy"], "multi_output_tree")
                self.assertEqual(model.get_params()["device"], "cuda:0")
            groups = ["North", "Central", "Northeast", "South"] if scope == "regional" else ["global"]
            self.assertEqual(stage2._device_count_for_context({"groups": groups, "spec": spec}), fit_count)

    def test_tuning_refuses_non_validation_context(self):
        args = stage2.parse_args(["--experiment", "1", "--development"])
        self.assertTrue(args.skip_test)
        with self.assertRaisesRegex(ValueError, "test context is locked"):
            stage2.hierarchical_search({"evaluation": "test"}, args, stage2.Path("."))

    def test_validation_tuning_context_excludes_locked_test_rows(self):
        spec = stage2.make_spec(1, 42)
        dates = pd.date_range("2025-11-01", "2026-04-25", freq="D")
        frame = pd.DataFrame({"station_id": 1, "date": dates})
        for target in stage2.TARGET_COLS:
            frame[target] = 10.0
        context = stage2.make_context(frame, [1], spec, "validation")
        self.assertTrue((context["frame"]["date"] < pd.Timestamp(stage2.config.TEST_START)).all())
        self.assertTrue((context["validation_rows"]["date"] < pd.Timestamp(stage2.config.TEST_START)).all())
        self.assertEqual(context["evaluation"], "validation")

    def test_cuda_native_multi_output_and_early_stopping_smoke(self):
        info = stage2.cuda_preflight(require_multi=True)
        self.assertEqual(info["cuda_training"], "PASS")
        self.assertEqual(info["multi_output_tree"], "PASS")
        self.assertEqual(info["multi_prediction_shape"], [8, 7])
        self.assertIn("early_stopping", info)


if __name__ == "__main__":
    unittest.main()
