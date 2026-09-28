"""Independent fixed screening parameters for Global E2 (all horizons).

These settings do not modify E1 recipes. n_estimators is currently a fixed
round count; temporal early stopping requires a separate training-policy change.
"""

_PARAMS = {
    "xgboost": dict(
        objective="reg:squarederror", n_estimators=1000, learning_rate=0.05,
        max_depth=5, min_child_weight=20, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=1.0, gamma=0.0,
    ),
    "lightgbm": dict(
        objective="regression", n_estimators=1000, learning_rate=0.05,
        max_depth=5, num_leaves=31, min_child_samples=20, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, reg_alpha=0.0,
        reg_lambda=1.0, min_split_gain=0.0,
    ),
    "gbr": dict(
        loss="squared_error", n_estimators=1000, learning_rate=0.05,
        max_depth=5, min_samples_leaf=20, min_samples_split=40,
        subsample=0.8, max_features=0.8,
    ),
}


def params_for(algorithm: str) -> dict:
    """Return a fresh copy so estimator changes cannot mutate the recipe."""
    return dict(_PARAMS[algorithm])
