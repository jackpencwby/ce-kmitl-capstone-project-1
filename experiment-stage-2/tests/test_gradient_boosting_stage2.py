from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.joint_gradient_boosting import JointMultiOutputGradientBoostingRegressor
from config.gradient_boosting import phase1_candidates, phase2_candidates, phase3_candidates
import run_stage2_gradient_boosting as stage2


class JointGBRTests(unittest.TestCase):
    def test_true_joint_outputs_and_staged_shapes(self):
        rng = np.random.RandomState(5)
        X = rng.normal(size=(80, 4))
        Y = np.column_stack([X[:, 0] * (h + 1) + X[:, 1] for h in range(7)])
        model = JointMultiOutputGradientBoostingRegressor(n_estimators=5, max_depth=2, random_state=42)
        model.fit(X, Y)
        pred = model.predict(X)
        self.assertEqual(pred.shape, (80, 7))
        self.assertEqual(model.n_outputs_, 7)
        self.assertEqual(len(model.estimators_), model.n_estimators_)
        self.assertTrue(all(isinstance(tree, DecisionTreeRegressor) and tree.n_outputs_ == 7
                            for tree in model.estimators_))
        self.assertEqual(sum(1 for _ in model.staged_predict(X)), 5)

    def test_serialization_round_trip(self):
        X = np.arange(80, dtype=float).reshape(20, 4)
        Y = np.column_stack([X[:, 0] + h * X[:, 1] for h in range(7)])
        model = JointMultiOutputGradientBoostingRegressor(n_estimators=3, random_state=7).fit(X, Y)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "joint.joblib"
            model.save_model(path)
            restored = JointMultiOutputGradientBoostingRegressor.load_model(path)
        np.testing.assert_allclose(model.predict(X), restored.predict(X), rtol=0, atol=1e-12)

    def test_scalar_architecture_is_seven_models(self):
        rng = np.random.RandomState(6)
        X = rng.normal(size=(50, 3))
        Y = np.column_stack([X[:, 0] + h * X[:, 1] for h in range(7)])
        models = [stage2._model({"max_depth": 2}, False, 42 + h, 3).fit(X, Y[:, h])
                  for h in range(7)]
        self.assertEqual(len(configured_horizons()), 7)
        self.assertEqual(len({id(model) for model in models}), 7)
        self.assertTrue(all(type(model) is GradientBoostingRegressor for model in models))
        self.assertEqual([stage2.make_spec(exp, 42).forecast_strategy for exp in (5, 6, 11, 12, 17, 18)],
                         ["direct", "multi", "direct", "multi", "direct", "multi"])


def configured_horizons():
    return [f"t+{h}" for h in range(1, 8)]


class HierarchicalGridTests(unittest.TestCase):
    def test_grids_and_local_generation_are_deterministic(self):
        first, second = phase1_candidates(), phase1_candidates()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 72)
        anchor = {**first[0], "min_samples_split": 10, "min_impurity_decrease": 1e-5, "ccp_alpha": 1e-4}
        self.assertEqual(phase2_candidates(anchor), phase2_candidates(anchor))
        self.assertEqual(phase3_candidates(anchor), phase3_candidates(anchor))


if __name__ == "__main__":
    unittest.main()
