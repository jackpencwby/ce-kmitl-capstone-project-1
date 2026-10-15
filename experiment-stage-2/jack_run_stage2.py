#!/usr/bin/env python3
#!/usr/bin/env python3
"""Stage 2 LightGBM tuning on CPU (coarse-to-fine hierarchical grid search).

The supported Stage 2 experiments are:

1. Global single-output forecasting (one target/horizon per estimator)
2. Global multi-output forecasting (ONE LightGBM model returns t+1...t+7)
3. Regional single-output forecasting
4. Regional multi-output forecasting
15. Global + per-station tree residual correction, single-output
16. Global + per-station tree residual correction, multi-output

Hyperparameters are searched with a deterministic grid search over the supplied
finite domain.  The full Cartesian product (16.2M combinations) is infeasible,
so the grid is walked in three phases (structure -> regularization -> local
refinement) with successive-halving style pruning.  Trials use the validation
holdout only.  The selected configuration is then refit on train+validation and
evaluated exactly once on the locked test split. Use ``--device gpu`` for the
OpenCL backend on Windows or ``--device cpu`` for CPU training. The selected
backend must pass preflight; failures do not silently switch devices.

Multi-output ("multiple prediction head") means ONE LightGBM model.  LightGBM
has no vector-leaf trees, so the seven targets are stacked into a long table
with a ``horizon`` feature (see ``_multi_output_partition``).
``MultiOutputRegressor`` and seven hidden per-horizon models are NOT used.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import random
import signal
import sys
import warnings
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


# Stage 2 intentionally reuses Stage 1's frozen data contract, causal feature
# construction, splits, metrics, and artifact schema.  The directory name has
# a hyphen, so it must be placed on sys.path rather than imported as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_DIR = REPO_ROOT / "experiment-stage-1"
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

from common import artifacts, config, data, metrics, models, runner, splits  # noqa: E402
from lgbm_checkpoint import LGBMCheckpoint, lightgbm_identity, complete_run, completed_run
from xgb_checkpoint import CheckpointError, atomic_json


LOGGER = logging.getLogger(__name__)


@dataclass
class Stage2RunSpec(models.RunSpec):
    gpu_platform_id: int = -1
    gpu_device_id: int = -1


def _device_params(device: str, platform_id: int = -1, device_id: int = -1) -> dict[str, Any]:
    params = {"device_type": device}
    if device == "gpu":
        params.update(gpu_platform_id=platform_id, gpu_device_id=device_id)
    return params

# Allowed parameter domain (the supplied LightGBM hyperparameters).
HYPERPARAMETER_GRID: dict[str, list[Any]] = {
    "n_estimators": [300, 500, 800, 1200, 2000],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "num_leaves": [15, 31, 63, 127, 255],
    "max_depth": [-1, 5, 7, 10, 12],
    "min_child_samples": [5, 10, 20, 30, 50, 100],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 0.001, 0.01, 0.1, 1, 5],
    "reg_lambda": [0, 0.01, 0.1, 1, 5, 10],
    "min_split_gain": [0, 0.01, 0.1, 0.5, 1],
}
GRID_SIZE = math.prod(len(values) for values in HYPERPARAMETER_GRID.values())

# Phase 1: coarse structural grid (regularization fixed at FIXED_REGULARIZATION).
PHASE1_GRID: dict[str, list[Any]] = {
    "learning_rate": [0.03, 0.1],
    "num_leaves": [15, 63, 255],
    "max_depth": [-1, 7],
    "min_child_samples": [10, 50],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}
# Phase 2: the full regularization domain, structure frozen at the Phase 1 winner.
PHASE2_GRID: dict[str, list[Any]] = {
    key: HYPERPARAMETER_GRID[key] for key in ("min_split_gain", "reg_alpha", "reg_lambda")
}
FIXED_REGULARIZATION = {"min_split_gain": 0, "reg_alpha": 0, "reg_lambda": 1}
STRUCTURAL_KEYS = tuple(PHASE1_GRID)

# n_estimators is not a free grid axis: every trial trains up to a budget taken
# from the n_estimators domain and early stopping picks the number of trees.
MAX_BOOSTING_ROUNDS = max(HYPERPARAMETER_GRID["n_estimators"])
PHASE_BUDGETS = {1: (300, 800, MAX_BOOSTING_ROUNDS),
                 2: (800, MAX_BOOSTING_ROUNDS),
                 3: (300, 800, MAX_BOOSTING_ROUNDS)}
PHASE1_KEEP_FRACTION = 3  # keep ceil(n / 3) after each Phase 1 round
PHASE_KEEP = {2: (36,), 3: (72, 12)}  # survivors after each non-final round
DEVELOPMENT_BUDGETS = {1: (20, 40, 60), 2: (40, 60), 3: (20, 40, 60)}

EXPERIMENTS = {
    1: dict(scope="global", forecast="direct", label="GLOBAL__SINGLE_OUTPUT"),
    2: dict(scope="global", forecast="multi", label="GLOBAL__MULTI_OUTPUT"),
    3: dict(scope="regional", forecast="direct", label="REGIONAL__SINGLE_OUTPUT"),
    4: dict(scope="regional", forecast="multi", label="REGIONAL__MULTI_OUTPUT"),
    15: dict(scope="global_local_tree", forecast="direct", label="GLOBAL_LOCAL_TREE__SINGLE_OUTPUT"),
    16: dict(scope="global_local_tree", forecast="multi", label="GLOBAL_LOCAL_TREE__MULTI_OUTPUT"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), type=int, required=True)
    parser.add_argument("--train", action="store_true", help="Run tuning and write artifacts.")
    parser.add_argument("--source", choices=["auto", "local", "gcs"], default="auto")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                        help="LightGBM backend; gpu uses OpenCL on Windows and requires a GPU-enabled build.")
    parser.add_argument("--gpu-platform-id", type=int, default=-1, help="OpenCL platform index (-1: automatic).")
    parser.add_argument("--gpu-device-id", type=int, default=-1, help="OpenCL device index (-1: automatic).")
    parser.add_argument("--stations", nargs="+", default=["all"])
    parser.add_argument("--max-stations", type=int, default=None)
    parser.add_argument("--seed", "--random-seed", dest="seed", type=int, default=config.SEED,
                        help="Seed for LightGBM sampling and candidate tie-breaking.")
    parser.add_argument("--early-stopping-rounds", type=int, default=100,
                        help="Early stopping patience; E2_15/16 use a purged internal training tail.")
    parser.add_argument("--phase1-pruning", action=argparse.BooleanOptionalAction, default=True,
                        help="Successive halving in Phase 1 (default on).")
    parser.add_argument("--max-phase1-candidates", type=int, default=None)
    parser.add_argument("--max-phase2-candidates", type=int, default=None)
    parser.add_argument("--max-phase3-candidates", type=int, default=None)
    parser.add_argument("--development", action="store_true",
                        help="Smoke-test mode: tiny candidate subsets and boosting budgets, "
                             "skips the test split, artifacts are prefixed DEV_.")
    parser.add_argument("--skip-test", action="store_true",
                        help="Tune on validation and save the winner without touching the test split.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--checkpoint", action="store_true", help="E15/16: save completed fits for resume.")
    parser.add_argument("--resume", type=Path, help="E15/16: resume an existing artifact directory.")
    parser.add_argument("--run-dir", type=Path, help="E15/16 checkpoint output directory (for Slurm continuation).")
    args = parser.parse_args(argv)
    if args.checkpoint or args.resume or args.run_dir:
        if args.experiment not in (15, 16) or not args.train:
            parser.error("Checkpoint options require --experiment 15/16 and --train")
        if args.run_dir and not (args.checkpoint or args.resume):
            parser.error("--run-dir requires --checkpoint or --resume")
        if args.run_dir and args.resume and args.run_dir.resolve() != args.resume.resolve():
            parser.error("--run-dir and --resume must refer to the same directory")
    if args.gpu_platform_id < -1 or args.gpu_device_id < -1:
        parser.error("GPU platform/device IDs must be -1 or non-negative")
    if args.early_stopping_rounds < 1:
        parser.error("--early-stopping-rounds must be positive")
    for name in ("max_stations", "max_phase1_candidates", "max_phase2_candidates",
                 "max_phase3_candidates"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    args.budgets = PHASE_BUDGETS
    if args.development:
        for name in ("max_phase1_candidates", "max_phase2_candidates", "max_phase3_candidates"):
            setattr(args, name, min(getattr(args, name) or 3, 3))
        args.budgets = DEVELOPMENT_BUDGETS
        args.early_stopping_rounds = min(args.early_stopping_rounds, 10)
        args.skip_test = True
    return args


def require_lightgbm(device: str = "cpu", platform_id: int = -1, device_id: int = -1) -> None:
    """Verify the requested backend before loading data; never silently fall back."""
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LightGBM is missing; install experiment-stage-2/requirements.txt") from error

    try:
        probe = lgb.LGBMRegressor(
            objective="regression",
            **_device_params(device, platform_id, device_id),
            n_estimators=1,
            min_child_samples=1,
            random_state=config.SEED,
            n_jobs=1,
            verbosity=1 if device == "gpu" else -1,
        )
        X = np.random.default_rng(config.SEED).normal(size=(200, 4))
        probe.fit(X, X[:, 0])
    except Exception as error:  # noqa: BLE001 - preserve the underlying error
        raise RuntimeError(
            f"LightGBM device={device} preflight failed: {error}. "
            "For gpu, install a GPU-enabled OpenCL build; use --device cpu explicitly to run on CPU."
        ) from error


def probe_multi_head(device: str = "cpu", platform_id: int = -1, device_id: int = -1) -> None:
    """Check that ONE LightGBM model fits t+1...t+7 jointly and predicts (n, 7)."""
    rng = np.random.default_rng(config.SEED)
    n = 60
    frame = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    for h in config.FORECAST_HORIZONS:
        frame[f"target_t{h}"] = frame["a"] * h + rng.normal(scale=0.1, size=n)
    targets = [f"target_t{h}" for h in config.FORECAST_HORIZONS]
    estimator, _ = _fit_lgbm({"min_child_samples": 1, **_device_params(device, platform_id, device_id)}, _stack_horizons(frame, ["a", "b", "c"]),
                             _stack_targets(frame, targets), None, None, 5, None)
    shape = np.asarray(estimator.predict(_stack_horizons(frame, ["a", "b", "c"]))).reshape(
        len(config.FORECAST_HORIZONS), n).T.shape
    if shape != (n, len(targets)):
        raise RuntimeError(f"Single-model multi-head probe returned {shape}; expected {(n, len(targets))}")


def make_cpu_lgbm(params: dict[str, Any], prefer_gpu: bool = True):
    """Build the explicitly selected backend (CPU by default).

    ``prefer_gpu`` is accepted because Stage 1's shared fitting helper passes
    it to factories. Device is carried in params so every nested fit uses it.
    The historical function name is retained for callers and test factories.
    """
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        objective="regression",
        random_state=SEED,
        n_jobs=-1,
        subsample_freq=1,  # enables row sampling when subsample < 1
        verbosity=-1,
        **{"device_type": "cpu", **params},
    )


def make_spec(experiment: int, seed: int = config.SEED, device: str = "cpu",
              gpu_platform_id: int = -1, gpu_device_id: int = -1) -> Stage2RunSpec:
    setting = EXPERIMENTS[experiment]
    scope = setting["scope"]
    return Stage2RunSpec(
        run_id=f"S2_{experiment}__{setting['label']}__LGBM_{device.upper()}__SEED{seed}",
        algorithm="lightgbm",
        training_strategy=scope,
        forecast_strategy=setting["forecast"],
        spatial_mode="none",
        prefer_gpu=(device == "gpu"),
        gpu_platform_id=gpu_platform_id,
        gpu_device_id=gpu_device_id,
        include_station_id=True,
        include_region_id=(scope == "regional"),
        # Multi-output fitting needs one complete seven-target row per sample.
        complete_target_evaluation=(setting["forecast"] == "multi"),
    )


def _evaluation_masks(df: pd.DataFrame, evaluation: str) -> dict[str, pd.Series]:
    masks = splits.date_masks(df)
    if evaluation == "test":
        masks["train"] = masks["train"] | masks["validation"]
        masks["validation"] = masks["test"]
    elif evaluation != "validation":
        raise ValueError(f"Unknown evaluation split: {evaluation}")
    return masks


HORIZON_FEATURE = "forecast_horizon"
SEED = config.SEED  # overwritten by --seed in main()


def _round_key(group: Any, horizon: int | None) -> str:
    """Identifier of one fitted model: ``<group>:t+h`` or ``<group>:multi_output``."""
    return f"{group}:multi_output" if horizon is None else f"{group}:t+{horizon}"


def _fit_lgbm(params: dict[str, Any], X: pd.DataFrame, y: Any, X_eval: pd.DataFrame | None,
              y_eval: Any, n_estimators: int, early_stopping_rounds: int | None):
    """Fit one LightGBM regressor; returns (estimator, number_of_trees_used)."""
    estimator = make_cpu_lgbm({**params, "n_estimators": int(n_estimators)})
    if early_stopping_rounds and X_eval is not None and len(X_eval):
        import lightgbm as lgb

        with warnings.catch_warnings():
            # LightGBM >= 4.7 deprecates eval_set; it is still the only spelling older 4.x accepts.
            warnings.filterwarnings("ignore", message=".*eval_set.*")
            estimator.fit(
                X, y, eval_set=[(X_eval, y_eval)], eval_metric="rmse",
                callbacks=[lgb.early_stopping(int(early_stopping_rounds), verbose=False)],
            )
        return estimator, int(getattr(estimator, "best_iteration_", None) or n_estimators)
    estimator.fit(X, y)
    return estimator, int(n_estimators)


def _rounds_for(fit_rounds: dict[str, int] | None, key: str, budget: int) -> int:
    """Tree count for a final refit: the validation-selected one when known."""
    if fit_rounds is None:
        return int(budget)
    if key not in fit_rounds:
        LOGGER.warning("No validation-selected tree count for %s; using budget %d", key, budget)
        return int(budget)
    return int(fit_rounds[key])


def _fit_residual_lgbm(params, X, y, dates, key, patience, selected, used_rounds):
    """Select on a purged training tail, then refit all training rows.

    Never consume outer scoring labels, including OOF scoring labels. At test
    time reuse validation-selected counts, including counts for each OOF fold.
    """
    budget = _rounds_for(selected, key, params["n_estimators"])
    used = budget
    if selected is None and patience:
        dates = pd.Series(np.asarray(dates))
        cutoff = dates.quantile(.85)
        train = (dates + pd.Timedelta(days=max(config.FORECAST_HORIZONS)) < cutoff).to_numpy()
        stop = (dates >= cutoff).to_numpy()
        if train.sum() >= 50 and stop.sum() >= 10:
            _, used = _fit_lgbm(params, X.iloc[train], np.asarray(y)[train],
                                X.iloc[stop], np.asarray(y)[stop], budget, patience)
        else:
            LOGGER.debug("%s: insufficient training-tail rows; using budget %d", key, budget)
    estimator, _ = _fit_lgbm(params, X, y, None, None, used, None)
    used_rounds[key] = int(used)
    return estimator, int(used)


def _stack_horizons(rows: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Repeat ``rows`` once per horizon, adding the ``forecast_horizon`` feature.

    The output is horizon-major: all rows for t+1, then all rows for t+2, ...
    """
    blocks = []
    for horizon in config.FORECAST_HORIZONS:
        block = rows[feature_columns].copy()
        block[HORIZON_FEATURE] = horizon
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def _stack_targets(rows: pd.DataFrame, target_columns: list[str]) -> np.ndarray:
    return np.concatenate([rows[column].to_numpy() for column in target_columns])


def _boundaries(spec: models.RunSpec, validation_frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(label boundary for evaluation rows, label boundary for training rows)."""
    boundary = (
        pd.Timestamp(spec.evaluation_end)
        if spec.evaluation_end
        else pd.Timestamp(config.TEST_START)
        if validation_frame[config.DATE_COL].min() < pd.Timestamp(config.TEST_START)
        else pd.Timestamp.max
    )
    train_label_end = (
        pd.Timestamp(spec.evaluation_start)
        if spec.evaluation_start
        else validation_frame[config.DATE_COL].min()
    )
    return boundary, train_label_end


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=[config.STATION_ID_COL, config.DATE_COL, "horizon", "y_true", "y_pred"])


def _direct_partition(
    spec: models.RunSpec,
    params: dict[str, Any],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    group_name: str,
    early_stopping_rounds: int | None = None,
    fit_rounds: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[dict[str, Any]]]:
    """Single-output forecasting: seven independent LightGBM models, one per horizon.

    Row selection mirrors Stage 1's ``_fit_partition`` direct branch; the only
    addition is optional validation early stopping.
    """
    id_columns = [config.STATION_ID_COL, config.DATE_COL]
    boundary, train_label_end = _boundaries(spec, validation_frame)
    predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []

    for horizon in config.FORECAST_HORIZONS:
        target = f"target_t{horizon}"
        delta = pd.Timedelta(days=horizon)
        train_rows = train_frame.dropna(subset=[target])
        evaluation_rows = validation_frame[validation_frame[config.DATE_COL] + delta < boundary]
        if len(evaluation_rows):
            train_rows = train_rows[train_rows[config.DATE_COL] + delta < train_label_end]
        if train_rows.empty or evaluation_rows.empty:
            continue
        scored = evaluation_rows.dropna(subset=[target])
        key = _round_key(group_name, horizon)
        estimator, used = _fit_lgbm(
            params, train_rows[feature_columns], train_rows[target],
            scored[feature_columns], scored[target],
            _rounds_for(fit_rounds, key, params["n_estimators"]),
            None if fit_rounds is not None else early_stopping_rounds,
        )
        block = evaluation_rows[id_columns].copy()
        block["horizon"] = horizon
        block["y_true"] = evaluation_rows[target].to_numpy()
        block["y_pred"] = estimator.predict(evaluation_rows[feature_columns])
        predictions.append(block)
        importance = models._importance_frame(estimator, feature_columns, f"h{horizon}")
        if not importance.empty:
            importance["group"] = str(group_name)
            importances.append(importance)
        records.append({"estimator": estimator, "group": str(group_name), "horizon": horizon,
                        "role": "main", "feature_columns": feature_columns,
                        "best_rounds": used})
    if not predictions:
        return _empty_predictions(), importances, records
    return pd.concat(predictions, ignore_index=True), importances, records


def _multi_output_partition(
    spec: models.RunSpec,
    params: dict[str, Any],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    group_name: str,
    early_stopping_rounds: int | None = None,
    fit_rounds: dict[str, int] | None = None,
    residual_fit=None,
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[dict[str, Any]]]:
    """Multi-output forecasting with ONE LightGBM model for t+1...t+7.

    LightGBM cannot learn a vector-valued target natively, and
    ``MultiOutputRegressor`` would only hide seven independent models.  Instead
    each complete 7-target row is stacked into seven long-format rows that carry
    a ``forecast_horizon`` feature, and a single ``LGBMRegressor`` is fitted once
    on the stacked table.  Every tree is shared by all horizons; the horizon
    feature lets the trees specialize per horizon where that helps.  Prediction
    stacks the evaluation rows the same way and reshapes back to (n_rows, 7).
    """
    target_columns = [f"target_t{h}" for h in config.FORECAST_HORIZONS]
    id_columns = [config.STATION_ID_COL, config.DATE_COL]
    max_horizon = max(config.FORECAST_HORIZONS)
    boundary, train_label_end = _boundaries(spec, validation_frame)

    train_rows = train_frame.dropna(subset=target_columns)
    train_rows = train_rows[
        train_rows[config.DATE_COL] + pd.Timedelta(days=max_horizon) < train_label_end
    ]
    validation_rows = validation_frame.dropna(subset=target_columns)
    validation_rows = validation_rows[
        validation_rows[config.DATE_COL] + pd.Timedelta(days=max_horizon) < boundary
    ]
    if train_rows.empty or validation_rows.empty:
        return _empty_predictions(), [], []

    stacked_columns = feature_columns + [HORIZON_FEATURE]
    key = _round_key(group_name, None)
    validation_stack = _stack_horizons(validation_rows, feature_columns)
    if residual_fit is not None:
        estimator, used = residual_fit(
            _stack_horizons(train_rows, feature_columns), _stack_targets(train_rows, target_columns),
            np.tile(train_rows[config.DATE_COL].to_numpy(), len(config.FORECAST_HORIZONS)), key,
        )
    else:
        estimator, used = _fit_lgbm(
            params, _stack_horizons(train_rows, feature_columns), _stack_targets(train_rows, target_columns),
            validation_stack, _stack_targets(validation_rows, target_columns),
            _rounds_for(fit_rounds, key, params["n_estimators"]),
            None if fit_rounds is not None else early_stopping_rounds,
        )
    flat = np.asarray(estimator.predict(validation_stack))
    n_horizons = len(config.FORECAST_HORIZONS)
    if flat.shape != (len(validation_rows) * n_horizons,):
        raise RuntimeError(
            f"Multi-output model returned an unexpected prediction shape: {flat.shape}; "
            f"expected ({len(validation_rows) * n_horizons},)"
        )
    prediction_matrix = flat.reshape(n_horizons, len(validation_rows)).T  # (n_rows, 7)

    prediction_blocks: list[pd.DataFrame] = []
    for output_index, horizon in enumerate(config.FORECAST_HORIZONS):
        block = validation_rows[id_columns].copy()
        block["horizon"] = horizon
        block["y_true"] = validation_rows[target_columns[output_index]].to_numpy()
        block["y_pred"] = prediction_matrix[:, output_index]
        prediction_blocks.append(block)

    importances: list[pd.DataFrame] = []
    importance = models._importance_frame(estimator, stacked_columns, "multi_output")
    if not importance.empty:
        importance["group"] = str(group_name)
        importances.append(importance)

    record = {
        "estimator": estimator,
        "group": str(group_name),
        "horizon": 0,
        "role": "multi_output",
        "feature_columns": stacked_columns,
        "target_columns": target_columns,
        "best_rounds": used,
    }
    return pd.concat(prediction_blocks, ignore_index=True), importances, [record]


def _run_multi_residual(spec: models.RunSpec, frame: pd.DataFrame,
                        masks: dict[str, pd.Series], feature_columns: list[str],
                        params: dict[str, Any], residual_fit=None) -> dict[str, Any]:
    """E1.3's OOF residual method with one horizon-stacked model at each level."""
    frame, feature_columns = models._add_categorical_codes(
        frame, spec, feature_columns, masks["train"]
    )
    train = frame.loc[masks["train"]]
    evaluation = frame.loc[masks["validation"]]
    predictions, importances, records = _multi_output_partition(
        spec, params, train, evaluation, feature_columns, "global", residual_fit=residual_fit
    )
    if predictions.empty:
        return models._finalize(spec, predictions, importances, feature_columns, records)

    cutoff = pd.Timestamp(spec.evaluation_start)
    folds = splits.residual_oof_folds(train, cutoff)
    oof_parts = []
    for fold in folds:
        train_mask, valid_mask = splits.fold_masks(train, fold)
        # Fold targets may extend beyond the fold's last origin, but never
        # into the held-out evaluation period. Training labels must precede
        # the first fold prediction origin.
        fold_spec = replace(spec, evaluation_start=fold.valid_start.isoformat(),
                            evaluation_end=cutoff.isoformat())
        oof, _, _ = _multi_output_partition(
            fold_spec, params, train.loc[train_mask], train.loc[valid_mask],
            feature_columns, f"global_oof_{fold.index}", residual_fit=residual_fit
        )
        if not oof.empty:
            oof_parts.append(oof)

    if oof_parts:
        ids = [config.STATION_ID_COL, config.DATE_COL]
        targets = [f"target_t{h}" for h in config.FORECAST_HORIZONS]
        oof = pd.concat(oof_parts, ignore_index=True)
        oof["residual"] = oof.y_true - oof.y_pred
        residuals = oof.pivot(index=ids, columns="horizon", values="residual")
        residuals = residuals.rename(columns={h: f"target_t{h}" for h in config.FORECAST_HORIZONS})
        local_train = train.drop(columns=targets).merge(
            residuals.reset_index(), on=ids, how="inner", validate="one_to_one"
        ).dropna(subset=targets)
        corrections = []
        for station, station_train in local_train.groupby(config.STATION_ID_COL):
            if len(station_train) < config.MLP_PARAMS.min_station_rows:
                continue
            station_eval = evaluation[evaluation[config.STATION_ID_COL] == station]
            correction, local_importances, local_records = _multi_output_partition(
                spec, params, station_train, station_eval, feature_columns, str(station),
                residual_fit=residual_fit
            )
            for record in local_records:
                record["role"] = "local_residual_multi_output"
            records.extend(local_records)
            importances.extend(local_importances)
            if not correction.empty:
                corrections.append(correction[ids + ["horizon", "y_pred"]])
        if corrections:
            correction_frame = pd.concat(corrections, ignore_index=True).rename(
                columns={"y_pred": "residual_pred"}
            )
            predictions = predictions.merge(correction_frame, on=ids + ["horizon"],
                                            how="left", validate="one_to_one")
            predictions["y_pred"] += predictions.pop("residual_pred").fillna(0.)
    return models._finalize(spec, predictions, importances, feature_columns, records)


def run_pooled(spec: models.RunSpec, df: pd.DataFrame, stations: list,
               params: dict[str, Any], evaluation: str,
               budget: int = MAX_BOOSTING_ROUNDS,
               early_stopping_rounds: int | None = None,
               fit_rounds: dict[str, int] | None = None,
               checkpoint=None, cache_summary=False) -> dict[str, Any]:
    """Train pooled LightGBM models for one split using a supplied trial config.

    ``budget`` is the maximum number of trees.  With ``early_stopping_rounds``
    the validation split picks the tree count (returned as ``fit_rounds``);
    ``fit_rounds`` from such a run fixes the tree counts for the test refit.
    Residual experiments (15/16) select rounds on purged internal training
    tails, refit all training rows, and reuse selected counts at test time.
    """
    frame = df[df[config.STATION_ID_COL].isin(stations)].copy()
    if evaluation == "validation":
        # Keep the locked test rows out of every tuning run.
        frame = frame[frame[config.DATE_COL] < pd.Timestamp(config.TEST_START)].copy()
    params = {**params, "n_estimators": int(budget),
              **_device_params("gpu" if spec.prefer_gpu else "cpu",
                               getattr(spec, "gpu_platform_id", -1), getattr(spec, "gpu_device_id", -1))}
    scope = checkpoint.scope({'evaluation': evaluation, 'params': params,
                              'patience': early_stopping_rounds, 'fit_rounds': fit_rounds,
                              'cache_summary': cache_summary}) if checkpoint else None
    if scope and cache_summary:
        cached = scope.load_summary()
        if cached is not None:
            return {'reports': {'overall': {'macro': {'primary_macro_rmse': cached['score']}},
                                'by_horizon': pd.DataFrame([
                                    {'horizon': int(h), 'rmse': v} for h, v in cached['rmse_by_horizon'].items()])},
                    'fit_rounds': cached['selected_rounds'], 'model_records': []}
    masks = _evaluation_masks(frame, evaluation)
    eval_spec = replace(
        spec,
        evaluation_start=(config.TEST_START if evaluation == "test" else config.VALIDATION_START),
        evaluation_end=(None if evaluation == "test" else config.TEST_START),
        selection_patience=None,
    )
    base_columns = models.assemble_feature_columns(eval_spec)
    factory = partial(make_cpu_lgbm, dict(params))
    if eval_spec.training_strategy == "global_local_tree":
        used_rounds = {}

        def residual_fit(X, y, dates, key):
            _check_stop()
            train = lambda: _fit_residual_lgbm(params, X, y, dates, key, early_stopping_rounds,
                                               fit_rounds, used_rounds)
            estimator, rounds = scope.fit(key, train, params['device_type']) if scope else train()
            used_rounds[key] = rounds
            _check_stop()
            return estimator, rounds

        def direct_fit(factory, prefer_gpu, X_tr, y_tr, X_val, y_val=None,
                       train_dates=None, horizon=0, group="global"):
            estimator, _ = residual_fit(X_tr, y_tr, train_dates, _round_key(group, horizon))
            return estimator.predict(X_val), estimator

        if eval_spec.forecast_strategy == "multi":
            result = _run_multi_residual(eval_spec, frame, masks, base_columns, params, residual_fit)
        else:
            result = models._run_residual(eval_spec, frame, factory, base_columns, masks,
                                          fit_predict=direct_fit)
        result["fit_rounds"] = used_rounds
        if scope and cache_summary:
            per_horizon = result['reports']['by_horizon'].set_index('horizon')['rmse'].to_dict()
            scope.save_summary({'score': primary_score(result), 'selected_rounds': used_rounds,
                                'best_iterations': used_rounds, 'best_scores': {},
                                'rmse_by_horizon': {str(h): float(v) for h, v in per_horizon.items()}})
        return result

    frame, feature_columns = models._add_categorical_codes(frame, eval_spec, base_columns, masks["train"])
    groups = [("global", frame)] if eval_spec.training_strategy == "global" else list(frame.groupby("region_id"))
    partition = _multi_output_partition if eval_spec.forecast_strategy == "multi" else _direct_partition
    predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []

    for group_name, group_frame in groups:
        train_frame = group_frame.loc[masks["train"].reindex(group_frame.index)]
        validation_frame = group_frame.loc[masks["validation"].reindex(group_frame.index)]
        group_predictions, group_importances, group_records = partition(
            eval_spec, params, train_frame, validation_frame, feature_columns, str(group_name),
            early_stopping_rounds=early_stopping_rounds, fit_rounds=fit_rounds,
        )
        predictions.append(group_predictions)
        importances.extend(group_importances)
        records.extend(group_records)

    combined = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    result = models._finalize(eval_spec, combined, importances, feature_columns, records)
    result["fit_rounds"] = {
        _round_key(record["group"], None if record["role"] == "multi_output" else record["horizon"]):
            record["best_rounds"]
        for record in records
    }
    return result


def primary_score(result: dict[str, Any]) -> float:
    value = result["reports"]["overall"]["macro"].get("primary_macro_rmse")
    if value is None or not np.isfinite(value):
        raise ValueError("Trial produced no finite primary macro RMSE")
    return float(value)


def artifact_directory(run_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(__file__).resolve().parent / "artifacts" / f"{run_id}__{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_persistence_comparison(run_dir: Path, split: str, prediction: pd.DataFrame,
                                 source_rows: pd.DataFrame) -> dict[str, Any]:
    comparison = metrics.paired_persistence_reports(prediction, source_rows)
    artifacts.write_json(run_dir / f"persistence_comparison_{split}.json", {
        "n": comparison["n"],
        "model": comparison["model"]["overall"],
        "persistence": comparison["persistence"]["overall"],
        "cohort": "paired finite station/date/horizon support",
    })
    return comparison


def save_split(run_dir: Path, spec: models.RunSpec, df: pd.DataFrame, stations: list,
               result: dict[str, Any], run_config: dict[str, Any], split: str) -> None:
    artifacts.save_run(
        run_dir=run_dir,
        run_id=spec.run_id,
        run_config=run_config,
        df=df,
        stations=stations,
        folds_manifest=(
            splits.fold_manifest(splits.residual_oof_folds(
                df[df[config.STATION_ID_COL].isin(stations)],
                pd.Timestamp(config.TEST_START if split == "test" else config.VALIDATION_START),
            )) if spec.training_strategy == "global_local_tree" else pd.DataFrame()
        ),
        reports=result["reports"],
        predictions=result["predictions"],
        feature_list=result["feature_list"],
        feature_importance=result["importance"],
        training_log=run_config["description"],
        split=split,
        model_records=result["model_records"],
        prefer_gpu=spec.prefer_gpu,
    )


# ---------------------------------------------------------------------------
# Hierarchical grid search (validation only)
# ---------------------------------------------------------------------------
def _grid_candidates(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of ``grid`` in a fixed, deterministic order."""
    names = list(grid)
    return [dict(zip(names, values)) for values in itertools.product(*(grid[name] for name in names))]


def phase1_candidates() -> list[dict[str, Any]]:
    """Coarse structural grid; regularization is held at its baseline."""
    return [{**candidate, **FIXED_REGULARIZATION} for candidate in _grid_candidates(PHASE1_GRID)]


def phase2_candidates(best_structure: dict[str, Any]) -> list[dict[str, Any]]:
    """Full regularization grid around the frozen Phase 1 structure."""
    anchor = {key: best_structure[key] for key in STRUCTURAL_KEYS}
    return [{**anchor, **candidate} for candidate in _grid_candidates(PHASE2_GRID)]


def _neighbor_values(domain: list[Any], center: Any, count: int = 3) -> list[Any]:
    """``count`` adjacent domain values containing ``center`` (window shifts at the edges)."""
    index = domain.index(center)
    start = max(0, min(index - count // 2, len(domain) - count))
    return domain[start:start + count]


def phase3_candidates(best_params: dict[str, Any]) -> list[dict[str, Any]]:
    """Local grid of neighbouring domain values around the Phase 2 winner."""
    local = {key: _neighbor_values(HYPERPARAMETER_GRID[key], best_params[key]) for key in STRUCTURAL_KEYS}
    return [{**best_params, **candidate} for candidate in _grid_candidates(local)]


def _candidate_key(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _cap(candidates: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    limited = candidates if limit is None else candidates[:limit]
    if not limited:
        raise ValueError("Candidate grid is empty")
    return limited


def _run_round(phase: int, round_id: int, candidates: list[dict[str, Any]], budget: int,
               spec: models.RunSpec, df: pd.DataFrame, stations: list, args: argparse.Namespace,
               run_dir: Path, trial_rows: list[dict[str, Any]], counters: dict[str, float]
               ) -> list[tuple[float, dict[str, Any]]]:
    """Evaluate ``candidates`` on validation; returns (score, params) sorted best-first."""
    scored: list[tuple[float, str, dict[str, Any]]] = []
    started_phase = perf_counter()
    for index, params in enumerate(candidates, start=1):
        _check_stop()
        started = perf_counter()
        trial_id = f"P{phase}R{round_id}T{index:04d}"
        row: dict[str, Any] = {
            "phase": phase, "round": round_id, "trial_id": trial_id,
            "experiment_id": args.experiment, "parameters": _candidate_key(params),
            "boosting_budget": budget, "best_iteration": None,
            "validation_primary_macro_rmse": None,
            **{f"rmse_t{h}": None for h in config.FORECAST_HORIZONS},
            "elapsed_seconds": None, "status": "running", "failure_reason": "",
        }
        try:
            result = run_pooled(spec, df, stations, params, "validation", budget=budget,
                                early_stopping_rounds=args.early_stopping_rounds,
                                checkpoint=getattr(args, '_checkpoint', None), cache_summary=True)
            score = primary_score(result)
            by_horizon = result["reports"]["by_horizon"]
            per_horizon = by_horizon.set_index("horizon")["rmse"].to_dict() if len(by_horizon) else {}
            row.update(
                best_iteration=json.dumps(result["fit_rounds"], sort_keys=True),
                validation_primary_macro_rmse=score,
                **{f"rmse_t{h}": per_horizon.get(h) for h in config.FORECAST_HORIZONS},
                status="ok",
            )
            counters["models"] += len(result["model_records"])
            scored.append((score, row["parameters"], params))
            LOGGER.info("%s RMSE %.6f (budget %d)", trial_id, score, budget)
        except CheckpointError:
            raise
        except Exception as error:  # noqa: BLE001 - keep candidate-level failures auditable
            row.update(status="failed", failure_reason=f"{type(error).__name__}: {error}")
            LOGGER.exception("%s candidate %s failed", trial_id, row["parameters"])
        finally:
            row["elapsed_seconds"] = perf_counter() - started
            trial_rows.append(row)
            pd.DataFrame(trial_rows).to_csv(run_dir / "tuning_trials.csv", index=False)
    counters[f"phase{phase}_seconds"] = counters.get(f"phase{phase}_seconds", 0.0) + perf_counter() - started_phase
    # Ties are broken by the canonical parameter string, so ranking is deterministic.
    scored.sort(key=lambda item: (item[0], item[1]))
    return [(score, params) for score, _, params in scored]


def _run_phase(phase: int, candidates: list[dict[str, Any]], keep, spec: models.RunSpec,
               df: pd.DataFrame, stations: list, args: argparse.Namespace, run_dir: Path,
               trial_rows: list[dict[str, Any]], counters: dict[str, float],
               budgets: tuple[int, ...]) -> tuple[dict[str, Any], int]:
    """Successive halving over ``budgets``; returns (winner, survivors after first round)."""
    survivors = candidates
    first_round_survivors = len(candidates)
    for round_id, budget in enumerate(budgets, start=1):
        ranked = _run_round(phase, round_id, survivors, budget, spec, df, stations, args,
                            run_dir, trial_rows, counters)
        if not ranked:
            raise RuntimeError(f"All Phase {phase} candidates failed in round {round_id}")
        if round_id < len(budgets):
            survivors = [params for _, params in ranked[:max(1, min(keep(round_id, len(ranked)), len(ranked)))]]
            if round_id == 1:
                first_round_survivors = len(survivors)
        else:
            survivors = [params for _, params in ranked]
    return survivors[0], first_round_survivors


def hierarchical_search(spec: models.RunSpec, df: pd.DataFrame, stations: list,
                        args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    """Coarse-to-fine grid search; every score comes from the validation split only."""
    started = perf_counter()
    trial_rows: list[dict[str, Any]] = []
    counters: dict[str, float] = {"models": 0}
    budgets = args.budgets

    if args.phase1_pruning:
        phase1_budgets = budgets[1]
        phase1_keep = lambda _round, n: math.ceil(n / PHASE1_KEEP_FRACTION)  # noqa: E731
    else:
        phase1_budgets = budgets[1][-1:]
        phase1_keep = lambda _round, n: n  # noqa: E731
    all_phase1 = phase1_candidates()
    p1 = _cap(all_phase1, args.max_phase1_candidates)
    phase1_winner, p1_survivors = _run_phase(1, p1, phase1_keep, spec, df, stations, args, run_dir,
                                             trial_rows, counters, phase1_budgets)
    _write_phase_csv(run_dir, trial_rows, 1)

    all_phase2 = phase2_candidates(phase1_winner)
    p2 = _cap(all_phase2, args.max_phase2_candidates)
    phase2_winner, p2_survivors = _run_phase(
        2, p2, lambda r, n: PHASE_KEEP[2][r - 1], spec, df, stations, args, run_dir,
        trial_rows, counters, budgets[2])
    _write_phase_csv(run_dir, trial_rows, 2)

    all_phase3 = phase3_candidates(phase2_winner)
    p3 = _cap(all_phase3, args.max_phase3_candidates)
    phase3_winner, p3_survivors = _run_phase(
        3, p3, lambda r, n: PHASE_KEEP[3][r - 1], spec, df, stations, args, run_dir,
        trial_rows, counters, budgets[3])
    _write_phase_csv(run_dir, trial_rows, 3)

    # The winner is the best configuration scored at the full boosting budget
    # (the last round of each phase), so scores are comparable across phases.
    final_budget = budgets[3][-1]
    finalists = [r for r in trial_rows if r["status"] == "ok" and r["boosting_budget"] == final_budget]
    if not finalists:
        raise RuntimeError("No hyperparameter trial completed at the full budget; see tuning_trials.csv")
    best_row = min(finalists, key=lambda r: (r["validation_primary_macro_rmse"], r["parameters"]))
    best_params = json.loads(best_row["parameters"])

    summary = {
        "full_grid_size": GRID_SIZE,
        "phase1_candidate_count": len(all_phase1), "phase1_candidates_used": len(p1),
        "phase1_survivors": p1_survivors,
        "phase2_candidate_count": len(all_phase2), "phase2_candidates_used": len(p2),
        "phase2_survivors": p2_survivors,
        "phase3_candidate_count": len(all_phase3), "phase3_candidates_used": len(p3),
        "phase3_survivors": p3_survivors,
        "total_candidate_evaluations": len(trial_rows),
        "total_models_trained": int(counters["models"]),
        "selected_phase": int(best_row["phase"]),
        "best_validation_score": float(best_row["validation_primary_macro_rmse"]),
        "best_parameters": best_params,
        "best_iteration": json.loads(best_row["best_iteration"]),
        "boosting_budgets": {f"phase{p}": list(b) for p, b in budgets.items()},
        "early_stopping_rounds": args.early_stopping_rounds,
        "phase_seconds": {k: v for k, v in counters.items() if k.endswith("_seconds")},
        "total_tuning_seconds": perf_counter() - started,
    }
    artifacts.write_json(run_dir / "tuning_summary.json", summary)
    return {"best_params": best_params, "best_row": best_row, "summary": summary,
            "trials": trial_rows, "final_budget": final_budget}


def _write_phase_csv(run_dir: Path, trial_rows: list[dict[str, Any]], phase: int) -> None:
    pd.DataFrame([r for r in trial_rows if r["phase"] == phase]).to_csv(
        run_dir / f"phase{phase}_results.csv", index=False)


_STOP_REQUESTED = False


def _stop_for_resume(signum, frame):
    # Never unwind through LightGBM's native/OpenCL training call.
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    LOGGER.warning('Received signal %s; stopping after the current fit is committed', signum)


def _check_stop():
    if _STOP_REQUESTED:
        raise SystemExit(75)


def main(argv: list[str] | None = None) -> int:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    try:
        with ExitStack() as stack:
            return _main(argv, stack)
    finally:
        _STOP_REQUESTED = False


def _main(argv, stack) -> int:
    run_started = perf_counter()
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    global SEED
    SEED = args.seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    require_lightgbm(args.device, args.gpu_platform_id, args.gpu_device_id)
    probe_multi_head(args.device, args.gpu_platform_id, args.gpu_device_id)

    spec = make_spec(args.experiment, args.seed, args.device, args.gpu_platform_id, args.gpu_device_id)
    if not args.train:
        print(f"[dry run] {args.device.upper()} LightGBM and single-model multi-head probes passed. "
              "Re-run with --train to load data and tune.")
        return 0

    df = runner.prepare_dataframe(spec, source=args.source)
    stations = data.resolve_stations(df, args.stations)
    if args.max_stations is not None:
        stations = stations[:args.max_stations]
    if not stations:
        raise ValueError("No eligible stations remain after selection")

    run_id = f"DEV_{spec.run_id}" if args.development else spec.run_id
    run_dir = (args.resume or args.run_dir).resolve() if (args.resume or args.run_dir) else artifact_directory(run_id)
    checkpoint = None
    if args.checkpoint or args.resume:
        checkpoint = stack.enter_context(LGBMCheckpoint(run_dir, lightgbm_identity(df, stations, args),
                                                       resume=bool(args.resume)))
        args._checkpoint = checkpoint
        for sig in (signal.SIGTERM, getattr(signal, 'SIGUSR1', signal.SIGTERM)):
            previous = signal.signal(sig, _stop_for_resume)
            stack.callback(signal.signal, sig, previous)
        if completed_run(checkpoint, args.experiment, args.skip_test):
            LOGGER.info('Run already completed: %s', run_dir)
            return 0
    LOGGER.info("Stage 2 experiment %d: %s; stations=%d; output=%s%s",
                args.experiment, EXPERIMENTS[args.experiment]["label"], len(stations), run_dir,
                " [development run]" if args.development else "")

    search = hierarchical_search(spec, df, stations, args, run_dir)
    best_params, best_score = search["best_params"], search["summary"]["best_validation_score"]
    final_budget = search["final_budget"]
    LOGGER.info("Selected parameters (phase %d): %s; validation RMSE %.6f",
                search["summary"]["selected_phase"], best_params, best_score)

    # Re-run the winner on validation to write a complete validation artifact and
    # to recover the early-stopped tree counts used by the train+validation refit.
    # Test is not involved in tuning.
    validation_result = run_pooled(spec, df, stations, best_params, "validation", budget=final_budget,
                                   early_stopping_rounds=args.early_stopping_rounds, checkpoint=checkpoint)
    selected_rounds = validation_result["fit_rounds"]
    validation_rows = df[df[config.STATION_ID_COL].isin(stations) & splits.date_masks(df)["validation"]]
    validation_pair = write_persistence_comparison(
        run_dir, "validation", validation_result["predictions"], validation_rows
    )
    output_mode = "multi-output (one model)" if spec.forecast_strategy == "multi" else "single-output/direct"
    description = (
        f"Stage 2 experiment {args.experiment}: {EXPERIMENTS[args.experiment]['scope']} LightGBM "
        f"with {output_mode} forecasting; {args.device.upper()} training. Parameters were selected by validation "
        "primary macro RMSE using a coarse-to-fine hierarchical grid search."
    )
    run_config = {
        "description": description,
        "experiment_id": args.experiment,
        "scope": EXPERIMENTS[args.experiment]["scope"],
        "algorithm": "lightgbm",
        "training_strategy": spec.training_strategy,
        "local_correction": (
            {"method": "out_of_fold_tree_residual", "group_by": config.STATION_ID_COL,
             "min_station_rows": config.MLP_PARAMS.min_station_rows,
             "fallback": "global_prediction", "parameter_policy": "same_trial_params_as_global"}
            if spec.training_strategy == "global_local_tree" else None
        ),
        "forecast_strategy": "multi_output" if spec.forecast_strategy == "multi" else "direct",
        "prediction_output": "multi_output" if spec.forecast_strategy == "multi" else "single_output",
        "multi_output_implementation": (
            "One LGBMRegressor per scope fitted once on horizon-stacked rows "
            f"(each sample repeated for t+1..t+7 with a '{HORIZON_FEATURE}' feature); "
            "MultiOutputRegressor and per-horizon models are not used."
            if spec.forecast_strategy == "multi" else None
        ),
        "spatial_mode": "none",
        "include_station_id": True,
        "include_region_id": spec.include_region_id,
        "requested_device": args.device,
        "actual_device_requirement": args.device,
        "gpu_platform_id": args.gpu_platform_id,
        "gpu_device_id": args.gpu_device_id,
        "selection_split": "validation",
        "test_locked_during_tuning": True,
        "tuning_mode": "hierarchical-grid",
        "hyperparameter_grid": HYPERPARAMETER_GRID,
        "phase_grids": {"phase1": PHASE1_GRID, "phase2": PHASE2_GRID,
                        "fixed_regularization_phase1": FIXED_REGULARIZATION},
        "boosting_budgets": {f"phase{p}": list(b) for p, b in args.budgets.items()},
        "early_stopping_rounds": args.early_stopping_rounds,
        "early_stopping_source": ("purged_training_tail_15_percent"
                                  if spec.training_strategy == "global_local_tree" else "validation"),
        "pruning_enabled": args.phase1_pruning,
        "development_run": args.development,
        "seed": args.seed,
        "selected_phase": search["summary"]["selected_phase"],
        "n_trials_completed": sum(t["status"] == "ok" for t in search["trials"]),
        "best_params": best_params,
        "selected_boosting_rounds": selected_rounds or {"all_models": final_budget},
        "best_validation_primary_macro_rmse": best_score,
        "paired_validation_n": validation_pair["n"],
        "paired_validation_model_primary_macro_rmse": validation_pair["model"]["overall"]["macro"].get("primary_macro_rmse"),
        "naive_validation_primary_macro_rmse": validation_pair["persistence"]["overall"]["macro"].get("primary_macro_rmse"),
        "evaluation_splits": ["validation"] if args.skip_test else ["validation", "test"],
        "feature_count": len(validation_result["feature_list"]),
        "spatial_config": None,
    }
    save_split(run_dir, spec, df, stations, validation_result, run_config, "validation")
    artifacts.write_json(run_dir / "best_params.json", {
        "selected_phase": search["summary"]["selected_phase"],
        "primary_macro_rmse": best_score,
        "selected_boosting_rounds": selected_rounds or {"all_models": final_budget},
        "params": best_params,
    })

    if args.skip_test:
        LOGGER.info("Validation winner saved. Test was skipped by request.")
        if checkpoint:
            complete_run(checkpoint, args.experiment, True)
        return 0

    test_started = perf_counter()
    test_result = run_pooled(spec, df, stations, best_params, "test", budget=final_budget,
                             fit_rounds=selected_rounds or None, checkpoint=checkpoint)
    test_rows = df[df[config.STATION_ID_COL].isin(stations) & splits.date_masks(df)["test"]]
    test_pair = write_persistence_comparison(run_dir, "test", test_result["predictions"], test_rows)
    run_config.update({
        "test_seconds": perf_counter() - test_started,
        "paired_test_n": test_pair["n"],
        "paired_test_model_primary_macro_rmse": test_pair["model"]["overall"]["macro"].get("primary_macro_rmse"),
        "naive_test_primary_macro_rmse": test_pair["persistence"]["overall"]["macro"].get("primary_macro_rmse"),
        "total_run_seconds": perf_counter() - run_started,
    })
    save_split(run_dir, spec, df, stations, test_result, run_config, "test")
    LOGGER.info("Complete. Validation RMSE=%.6f; test RMSE=%s",
                best_score, test_pair["model"]["overall"]["macro"].get("primary_macro_rmse"))
    if checkpoint:
        complete_run(checkpoint, args.experiment, False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
