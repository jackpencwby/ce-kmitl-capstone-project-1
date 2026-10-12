"""Contracts for the jack_ Stage 2 engine: single-model multi-head and hierarchical grid search."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jack_run_stage2 as stage2

FEATURES = ["a", "b"]
TARGETS = [f"target_t{h}" for h in stage2.config.FORECAST_HORIZONS]


def toy_frame(n=400):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    for h in stage2.config.FORECAST_HORIZONS:
        frame[f"target_t{h}"] = frame["a"] * h  # horizon-dependent, exactly learnable
    return frame


class MultiHeadTests(unittest.TestCase):
    def test_one_estimator_returns_seven_aligned_outputs(self):
        frame = toy_frame()
        estimator, rounds = stage2._fit_lgbm(
            {"learning_rate": 0.3, "num_leaves": 31, "min_child_samples": 5},
            stage2._stack_horizons(frame, FEATURES), stage2._stack_targets(frame, TARGETS),
            None, None, 300, None)
        self.assertEqual(rounds, 300)
        flat = estimator.predict(stage2._stack_horizons(frame, FEATURES))
        matrix = flat.reshape(len(stage2.config.FORECAST_HORIZONS), len(frame)).T
        self.assertEqual(matrix.shape, (len(frame), 7))
        # Column j must be the t+(j+1) forecast, i.e. a * (j + 1).
        for j in range(7):
            self.assertLess(np.abs(matrix[:, j] - frame["a"] * (j + 1)).mean(), 0.5, f"column {j}")

    def test_stacking_is_horizon_major_with_horizon_feature(self):
        frame = toy_frame(5)
        stacked = stage2._stack_horizons(frame, FEATURES)
        self.assertEqual(list(stacked.columns), FEATURES + [stage2.HORIZON_FEATURE])
        self.assertEqual(stacked[stage2.HORIZON_FEATURE].tolist(), [h for h in range(1, 8) for _ in range(5)])
        self.assertEqual(len(stage2._stack_targets(frame, TARGETS)), 35)

    def test_early_stopping_reports_tree_count_below_budget(self):
        frame = toy_frame()
        X, y = stage2._stack_horizons(frame, FEATURES), stage2._stack_targets(frame, TARGETS)
        noisy = y + np.random.default_rng(1).normal(scale=3.0, size=len(y))  # held-out noise stops improvement
        _, rounds = stage2._fit_lgbm({"learning_rate": 0.3, "min_child_samples": 5}, X, y, X, noisy, 2000, 10)
        self.assertLess(rounds, 2000)

    def test_probe_passes(self):
        stage2.probe_multi_head()


class GridTests(unittest.TestCase):
    def test_phase_grids_stay_inside_supplied_domain(self):
        for grid in (stage2.PHASE1_GRID, stage2.PHASE2_GRID):
            for key, values in grid.items():
                self.assertTrue(set(values) <= set(stage2.HYPERPARAMETER_GRID[key]), key)
        for budgets in stage2.PHASE_BUDGETS.values():
            self.assertTrue(set(budgets) <= set(stage2.HYPERPARAMETER_GRID["n_estimators"]))

    def test_candidate_counts_and_determinism(self):
        p1 = stage2.phase1_candidates()
        self.assertEqual(len(p1), 144)
        self.assertEqual(p1, stage2.phase1_candidates())
        best = p1[17]
        p2 = stage2.phase2_candidates(best)
        self.assertEqual(len(p2), 5 * 6 * 6)
        self.assertEqual(p2, stage2.phase2_candidates(dict(best)))
        self.assertTrue(all(c[k] == best[k] for c in p2 for k in stage2.STRUCTURAL_KEYS))
        p3 = stage2.phase3_candidates(p2[0])
        self.assertEqual(len(p3), 3 ** len(stage2.STRUCTURAL_KEYS))
        self.assertIn({**p2[0]}, p3)  # the incumbent is part of its own neighbourhood

    def test_neighbor_values_window_shifts_at_edges(self):
        domain = [1, 2, 3, 4, 5]
        self.assertEqual(stage2._neighbor_values(domain, 1), [1, 2, 3])
        self.assertEqual(stage2._neighbor_values(domain, 3), [2, 3, 4])
        self.assertEqual(stage2._neighbor_values(domain, 5), [3, 4, 5])

    def test_no_random_search_or_wrapper_left(self):
        source = Path(stage2.__file__).read_text(encoding="utf-8")
        self.assertNotIn("ParameterSampler", source)
        self.assertNotIn("import MultiOutputRegressor", source)


if __name__ == "__main__":
    unittest.main()
