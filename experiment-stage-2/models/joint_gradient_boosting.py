"""Joint multi-output squared-error gradient boosting for Stage 2.

Each boosting stage fits exactly one vector-valued DecisionTreeRegressor to
the residual matrix. This is not a wrapper around independent estimators.
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted


class JointMultiOutputGradientBoostingRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0,
                 max_features=None, min_impurity_decrease=0.0, ccp_alpha=0.0,
                 random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in (
            "n_estimators", "learning_rate", "max_depth", "min_samples_split",
            "min_samples_leaf", "subsample", "max_features",
            "min_impurity_decrease", "ccp_alpha", "random_state")}

    def set_params(self, **params):
        unknown = set(params) - set(self.get_params())
        if unknown:
            raise ValueError(f"Unknown parameter(s): {sorted(unknown)}")
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, Y, eval_set=None):
        X = check_array(X, dtype=np.float32, ensure_2d=True, ensure_all_finite=True)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim != 2 or Y.shape[0] != X.shape[0] or Y.shape[1] < 2:
            raise ValueError("Y must be a finite (n_samples, n_outputs>=2) matrix")
        if not np.isfinite(Y).all():
            raise ValueError("Y contains non-finite values")
        if int(self.n_estimators) < 1 or not 0 < float(self.subsample) <= 1:
            raise ValueError("n_estimators must be positive and subsample in (0, 1]")
        if float(self.learning_rate) <= 0:
            raise ValueError("learning_rate must be positive")
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = Y.shape[1]
        self.n_estimators_ = int(self.n_estimators)
        self.init_prediction_ = Y.mean(axis=0)
        self.estimators_ = []
        rng = check_random_state(self.random_state)
        pred = np.broadcast_to(self.init_prediction_, Y.shape).copy()
        n = len(X)
        for _ in range(self.n_estimators_):
            residual = Y - pred
            if self.subsample < 1:
                count = max(1, int(np.ceil(n * float(self.subsample))))
                rows = rng.choice(n, size=count, replace=False)
            else:
                rows = np.arange(n)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha,
                random_state=int(rng.randint(np.iinfo(np.int32).max)),
            )
            tree.fit(X[rows], residual[rows])
            self.estimators_.append(tree)
            pred += float(self.learning_rate) * tree.predict(X)
        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.estimators_], axis=0)
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        return self

    def staged_predict(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=np.float32, ensure_2d=True, ensure_all_finite=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        pred = np.broadcast_to(self.init_prediction_, (len(X), self.n_outputs_)).copy()
        for tree in self.estimators_:
            pred += float(self.learning_rate) * tree.predict(X)
            yield pred.copy()

    def predict(self, X):
        last = None
        for last in self.staged_predict(X):
            pass
        return last

    def save_model(self, path):
        joblib.dump(self, path)

    @classmethod
    def load_model(cls, path):
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, found {type(model).__name__}")
        return model
