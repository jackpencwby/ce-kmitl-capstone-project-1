"""Research domain and deterministic coarse-to-fine grids for GBR."""
from __future__ import annotations
import itertools

GBR_PARAM_DOMAIN = {
    "n_estimators": [300, 500, 800, 1200, 2000],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4, 5, 6],
    "min_samples_split": [2, 5, 10, 20, 50],
    "min_samples_leaf": [1, 2, 5, 10, 20, 50],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "max_features": [0.5, 0.7, 0.9, 1.0, "sqrt"],
    "min_impurity_decrease": [0.0, 1e-5, 1e-4, 1e-3],
    "ccp_alpha": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
}
PHASE1_GRID = {
    "learning_rate": [0.03, 0.1], "max_depth": [2, 3, 5],
    "min_samples_leaf": [1, 10, 50], "subsample": [0.7, 1.0],
    "max_features": [0.7, 1.0],
}
PHASE2_GRID = {"min_samples_split": [2, 10, 50],
               "min_impurity_decrease": [0.0, 1e-5, 1e-4, 1e-3],
               "ccp_alpha": [0.0, 1e-5, 1e-4, 1e-3]}
FIXED = {"min_samples_split": 2, "min_impurity_decrease": 0.0, "ccp_alpha": 0.0}

def _grid(values):
    keys = list(values)
    return [dict(zip(keys, items)) for items in itertools.product(*(values[k] for k in keys))]

def phase1_candidates():
    return _grid(PHASE1_GRID)

def phase2_candidates(anchor):
    structural = {k: anchor[k] for k in PHASE1_GRID}
    return [{**structural, **x} for x in _grid(PHASE2_GRID)]

def phase3_candidates(anchor):
    """Neighbor values around anchor, with deterministic, bounded local grid."""
    def neighbors(key):
        values = GBR_PARAM_DOMAIN[key]
        idx = values.index(anchor[key])
        return values[max(0, idx-1):min(len(values), idx+2)]
    grid = {k: neighbors(k) for k in PHASE1_GRID}
    # Keep split/pruning anchored during the main local structural refinement;
    # their one-step neighbors are represented in Phase 2.
    return [{**anchor, **x} for x in _grid(grid)]
