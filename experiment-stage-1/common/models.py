"""Model construction and the reusable training/evaluation runner.

This module encapsulates every axis Stage 1 varies so the individual
E*_*.py scripts stay thin:

  * algorithm       : "xgboost" | "lightgbm" | "gbr"
  * training_strategy: "local" | "global" | "regional"
                       | "global_local_tree" | "global_local_mlp"
  * forecast_strategy: "direct" (one model per horizon)
                       | "multi"  (one model, 7 outputs)
  * spatial mode    : handled upstream in features.add_neighbor_features

The runner fits on the main training window, predicts the validation
holdout, and returns a long prediction frame + feature importances +
per-fold walk-forward metrics. Preprocessing (scaler/encoder) that must be
fit is fit on training rows only (plan section 15).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import config, features, metrics, splits

LOGGER = logging.getLogger(__name__)


def _record(records, estimator, group, horizon, role="main"):
    records.append({"estimator": estimator, "group": str(group),
                    "horizon": int(horizon), "role": role})


# ---------------------------------------------------------------------------
# Estimator factories (fixed reasonable defaults, plan section 5 & 11)
# ---------------------------------------------------------------------------
def make_xgb(prefer_gpu: bool = True, horizon: int = 1):
    import xgboost as xgb
    params = config.xgb_params_for_horizon(horizon)
    device = "cpu"
    tree_method = "hist"
    if prefer_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
        except Exception:  # noqa: BLE001
            device = "cpu"
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        tree_method=tree_method,
        device=device,
        random_state=config.SEED,
        n_jobs=0,
        **params,
    )


def make_baseline_xgb(prefer_gpu: bool = True, horizon: int = 1):
    """XGBoost recipe selected for the given horizon, with early stopping."""
    import xgboost as xgb
    device = make_xgb(prefer_gpu).get_params()["device"]
    return xgb.XGBRegressor(
        objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
        device=device, random_state=config.SEED, n_jobs=4,
        early_stopping_rounds=config.BASELINE_EARLY_STOPPING_ROUNDS,
        **config.xgb_params_for_horizon(horizon),
    )


def _horizon_factory(factory: Callable, horizon: int) -> Callable:
    if factory in (make_xgb, make_baseline_xgb):
        return partial(factory, horizon=horizon)
    return factory


_LGBM_GPU_SUPPORTED: Optional[bool] = None


def _lgbm_gpu_supported() -> bool:
    """Probe once whether this LightGBM build actually supports GPU.

    Many pip wheels are CPU-only; the plan requires a clean CPU fallback in
    that case (section 3.3 / 11.2). We fit a 1-row model on the GPU device
    and cache whether it raised the "GPU Tree Learner was not enabled" error.
    """
    global _LGBM_GPU_SUPPORTED
    if _LGBM_GPU_SUPPORTED is not None:
        return _LGBM_GPU_SUPPORTED
    try:
        import torch
        if not torch.cuda.is_available():
            _LGBM_GPU_SUPPORTED = False
            return False
    except Exception:  # noqa: BLE001
        _LGBM_GPU_SUPPORTED = False
        return False
    try:
        import lightgbm as lgb
        probe = lgb.LGBMRegressor(device_type="gpu", n_estimators=1, min_child_samples=1, verbose=-1)
        probe.fit(np.zeros((4, 2)), np.array([0.0, 1.0, 0.0, 1.0]))
        _LGBM_GPU_SUPPORTED = True
    except Exception as error:  # noqa: BLE001
        LOGGER.warning("LightGBM GPU not available (%s); using CPU.", error)
        _LGBM_GPU_SUPPORTED = False
    return _LGBM_GPU_SUPPORTED


def make_lgbm(prefer_gpu: bool = True):
    import lightgbm as lgb
    params = config.LGBM_PARAMS.as_dict()
    device_type = "gpu" if (prefer_gpu and _lgbm_gpu_supported()) else "cpu"
    return lgb.LGBMRegressor(
        objective="regression",
        random_state=config.SEED,
        n_jobs=-1,
        device_type=device_type,
        verbose=-1,
        **params,
    )


def make_gbr(prefer_gpu: bool = True):  # GBR is CPU-only regardless.
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    params = config.GBR_PARAMS.as_dict()
    return make_pipeline(
        SimpleImputer(strategy="constant", fill_value=-999.0,
                      keep_empty_features=True),
        GradientBoostingRegressor(random_state=config.SEED, **params),
    )


ESTIMATOR_FACTORIES: dict[str, Callable] = {
    "xgboost": make_xgb,
    "lightgbm": make_lgbm,
    "gbr": make_gbr,
}


# ---------------------------------------------------------------------------
# Run specification
# ---------------------------------------------------------------------------
@dataclass
class RunSpec:
    run_id: str
    algorithm: str = "xgboost"
    training_strategy: str = "local"
    forecast_strategy: str = "direct"
    spatial_mode: str = "none"          # none|unweighted|distance|wind
    prefer_gpu: bool = True
    include_station_id: bool = False    # global/regional add station_id feature
    include_region_id: bool = False     # regional adds region_id feature
    baseline_xgb: bool = False          # E1 shared XGBoost recipe + early stopping


# ---------------------------------------------------------------------------
# Feature matrix assembly
# ---------------------------------------------------------------------------
def assemble_feature_columns(spec: RunSpec) -> list[str]:
    cols = config.baseline_feature_list()
    if spec.spatial_mode != "none":
        cols = cols + features.neighbor_feature_cols()
    return cols


def _fit_predict_estimator(
    factory: Callable, prefer_gpu: bool,
    X_tr: pd.DataFrame, y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series | None = None,
    train_dates: pd.Series | None = None,
    horizon: int = 0,
) -> tuple[np.ndarray, object]:
    """Fit a fresh estimator on training rows, predict validation rows."""
    est = factory(prefer_gpu=prefer_gpu)
    if y_val is None:
        if getattr(est, "early_stopping_rounds", None):
            used_internal_eval = False
            if train_dates is not None:
                dates = pd.to_datetime(train_dates)
                cutoff = dates.quantile(0.85)
                fit_mask = dates + pd.Timedelta(days=horizon) < cutoff
                eval_mask = dates >= cutoff
                if fit_mask.sum() >= 50 and eval_mask.sum() >= 10:
                    est.fit(X_tr.loc[fit_mask], y_tr.loc[fit_mask],
                            eval_set=[(X_tr.loc[fit_mask], y_tr.loc[fit_mask]),
                                      (X_tr.loc[eval_mask], y_tr.loc[eval_mask])],
                            verbose=False)
                    used_internal_eval = True
            if not used_internal_eval:
                est.set_params(early_stopping_rounds=None)
                est.fit(X_tr, y_tr)
        else:
            est.fit(X_tr, y_tr)
    else:
        valid = y_val.notna()
        if not valid.any():
            raise ValueError("Early stopping requires validation targets")
        est.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr),
                                     (X_val.loc[valid], y_val.loc[valid])],
                verbose=False)
    return est.predict(X_val), est


def _importance_frame(est, feature_cols: list[str], tag: str) -> pd.DataFrame:
    fitted = est.steps[-1][1] if hasattr(est, "steps") else est
    imp = getattr(fitted, "feature_importances_", None)
    if imp is None or len(imp) != len(feature_cols):
        return pd.DataFrame()
    return pd.DataFrame({"feature": feature_cols, "importance": imp, "model": tag})


# ---------------------------------------------------------------------------
# Direct vs multi-output fitting on a given train/val partition
# ---------------------------------------------------------------------------
def _fit_partition(
    spec: RunSpec, factory: Callable,
    train_df: pd.DataFrame, val_df: pd.DataFrame,
    feature_cols: list[str],
    model_records: list | None = None,
    group: str = "global",
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Fit and predict for one train/val partition.

    Returns (long prediction frame, list of importance frames). Handles both
    direct (per-horizon) and multi-output forecasting.
    """
    id_cols = [config.STATION_ID_COL, config.DATE_COL]
    tcols = features.target_cols()
    preds: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []

    if spec.forecast_strategy == "multi":
        # Rows must have all 7 targets present for training (plan 2.1, 8).
        tr = train_df.dropna(subset=tcols)
        va = val_df[
            val_df[config.DATE_COL] + pd.Timedelta(days=max(config.FORECAST_HORIZONS))
            < (pd.Timestamp(config.TEST_START) if val_df[config.DATE_COL].min()
               < pd.Timestamp(config.TEST_START) else pd.Timestamp.max)
        ]
        if len(va):
            tr = tr[tr[config.DATE_COL] + pd.Timedelta(days=max(config.FORECAST_HORIZONS))
                    < va[config.DATE_COL].min()]
        if len(tr) == 0 or len(va) == 0:
            return pd.DataFrame(columns=id_cols + ["horizon", "y_true", "y_pred"]), importances
        # MultiOutputRegressor fits independent target estimators. Fit them
        # explicitly so each horizon receives its own selected parameters.
        for h in config.FORECAST_HORIZONS:
            hfactory = _horizon_factory(factory, h)
            yhat, estimator = _fit_predict_estimator(
                hfactory, spec.prefer_gpu, tr[feature_cols], tr[f"target_t{h}"],
                va[feature_cols],
                va[f"target_t{h}"] if spec.baseline_xgb else None,
            )
            if model_records is not None:
                _record(model_records, estimator, group, h)
            block = va[id_cols].copy()
            block["horizon"] = h
            block["y_true"] = va[f"target_t{h}"].to_numpy()
            block["y_pred"] = yhat
            preds.append(block)
            importances.append(_importance_frame(estimator, feature_cols, f"multi_h{h}"))
        return pd.concat(preds, ignore_index=True), importances

    # Direct: one estimator per horizon.
    for h in config.FORECAST_HORIZONS:
        tcol = f"target_t{h}"
        tr = train_df.dropna(subset=[tcol])
        boundary = (pd.Timestamp(config.TEST_START) if val_df[config.DATE_COL].min()
                    < pd.Timestamp(config.TEST_START) else pd.Timestamp.max)
        va = val_df[val_df[config.DATE_COL] + pd.Timedelta(days=h) < boundary]
        if len(va):
            tr = tr[tr[config.DATE_COL] + pd.Timedelta(days=h)
                    < va[config.DATE_COL].min()]
        if len(tr) == 0 or len(va) == 0 or (spec.baseline_xgb and not va[tcol].notna().any()):
            continue
        yhat, est = _fit_predict_estimator(
            _horizon_factory(factory, h), spec.prefer_gpu, tr[feature_cols], tr[tcol], va[feature_cols],
            va[tcol] if spec.baseline_xgb else None,
        )
        if model_records is not None:
            _record(model_records, est, group, h)
        block = va[id_cols].copy()
        block["horizon"] = h
        block["y_true"] = va[tcol].to_numpy()
        block["y_pred"] = yhat
        preds.append(block)
        importances.append(_importance_frame(est, feature_cols, f"h{h}"))
    if not preds:
        return pd.DataFrame(columns=id_cols + ["horizon", "y_true", "y_pred"]), importances
    return pd.concat(preds, ignore_index=True), importances


# ---------------------------------------------------------------------------
# Encoding station_id / region_id for global & regional strategies
# ---------------------------------------------------------------------------
def _add_categorical_codes(df: pd.DataFrame, spec: RunSpec, feature_cols: list[str],
                           train_mask: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    """Add integer-coded station_id / region_id fit from training rows only.

    Trees handle integer category codes fine; categories unseen in training
    map to -1 (unknown), satisfying plan section 3.2.
    """
    df = df.copy()
    cols = list(feature_cols)
    if spec.include_station_id:
        cats = pd.Index(sorted(df.loc[train_mask, config.STATION_ID_COL].unique()))
        mapping = {c: i for i, c in enumerate(cats)}
        df["station_id_code"] = df[config.STATION_ID_COL].map(mapping).fillna(-1).astype(int)
        cols.append("station_id_code")
    if spec.include_region_id and "region_id" in df.columns:
        cats = pd.Index(sorted(df.loc[train_mask, "region_id"].dropna().unique()))
        mapping = {c: i for i, c in enumerate(cats)}
        df["region_id_code"] = df["region_id"].map(mapping).fillna(-1).astype(int)
        cols.append("region_id_code")
    return df, cols


# ---------------------------------------------------------------------------
# Main holdout run (train window -> validation window)
# ---------------------------------------------------------------------------
def run_holdout(spec: RunSpec, df: pd.DataFrame, stations: list,
                evaluation: str = "validation") -> dict:
    """Fit on the training window and predict the validation holdout.

    Returns predictions, reports, feature importances, and the feature list.
    Dispatches on spec.training_strategy.
    """
    factory = make_baseline_xgb if spec.baseline_xgb else ESTIMATOR_FACTORIES[spec.algorithm]
    feature_cols = assemble_feature_columns(spec)
    masks = splits.date_masks(df)

    df = df[df[config.STATION_ID_COL].isin(stations)].copy()
    masks = splits.date_masks(df)
    if evaluation == "test":
        masks["train"] = masks["train"] | masks["validation"]
        masks["validation"] = masks["test"]
    elif evaluation != "validation":
        raise ValueError(f"Unknown evaluation split: {evaluation}")

    if spec.training_strategy == "local":
        return _run_local(spec, df, factory, feature_cols, masks)
    if spec.training_strategy in ("global", "regional"):
        return _run_pooled(spec, df, factory, feature_cols, masks)
    if spec.training_strategy in ("global_local_tree", "global_local_mlp"):
        return _run_residual(spec, df, factory, feature_cols, masks)
    raise ValueError(f"Unknown training strategy: {spec.training_strategy}")


def predict_local_test_from_validation_models(spec: RunSpec, df: pd.DataFrame,
                                              records: list) -> dict:
    """Backward-compatible E1.1 entry point for the shared E1 scorer."""
    stations = sorted(df[config.STATION_ID_COL].unique())
    return predict_e1_test_from_validation_models(spec, df, stations, records)


def predict_e1_test_from_validation_models(spec: RunSpec, df: pd.DataFrame,
                                           stations: list, records: list) -> dict:
    """Score E1 test without fitting or selecting anything from test targets."""
    if not spec.baseline_xgb:
        raise ValueError("Validation-model reuse requires the shared E1 XGBoost recipe")
    cols = assemble_feature_columns(spec)
    chosen = df[df[config.STATION_ID_COL].isin(stations)].copy()
    masks = splits.date_masks(chosen)
    if spec.include_station_id or spec.include_region_id:
        chosen, cols = _add_categorical_codes(chosen, spec, cols, masks["train"])
    test = chosen[masks["test"]]
    model_index = {(rec["role"], rec["group"], rec["horizon"]): rec
                   for rec in records}
    predictions = []
    for h in config.FORECAST_HORIZONS:
        if spec.training_strategy == "local":
            groups = ((str(sid), part) for sid, part in test.groupby(config.STATION_ID_COL))
        elif spec.training_strategy == "regional":
            groups = ((str(region), part) for region, part in test.groupby("region_id"))
        else:
            groups = (("global", test),)
        for group, part in groups:
            rec = model_index.get(("main", group, h))
            if rec is None:
                continue
            rows = part
            if rows.empty:
                continue
            global_prediction = np.asarray(rec["estimator"].predict(rows[cols]))
            prediction = global_prediction.copy()
            if spec.training_strategy in ("global_local_tree", "global_local_mlp"):
                for sid, idx in rows.groupby(config.STATION_ID_COL).groups.items():
                    local = model_index.get(("local_residual", str(sid), h))
                    if local is None:
                        local = model_index.get(("local_residual_mlp", str(sid), h))
                    if local is None:
                        continue
                    positions = rows.index.get_indexer(idx)
                    station_rows = rows.loc[idx]
                    if local["role"] == "local_residual_mlp":
                        import torch
                        local_features = station_rows[cols].copy()
                        local_features["global_pred"] = global_prediction[positions]
                        X = local_features[local["feature_columns"]].to_numpy(dtype=np.float32)
                        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                        X = (X - local["mean"]) / local["std"]
                        with torch.no_grad():
                            correction = local["estimator"](
                                torch.tensor(X, dtype=torch.float32)).numpy().reshape(-1)
                    else:
                        correction = local["estimator"].predict(station_rows[cols])
                    prediction[positions] += correction
            block = rows[[config.STATION_ID_COL, config.DATE_COL]].copy()
            block["horizon"] = h
            block["y_true"] = rows[f"target_t{h}"].to_numpy()
            block["y_pred"] = prediction
            predictions.append(block)
    prediction = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return _finalize(spec, prediction, [], cols, [])


def _run_local(spec, df, factory, feature_cols, masks) -> dict:
    """One independent model set per station (E1.1)."""
    all_preds, all_imp, records = [], [], []
    for sid, sub in df.groupby(config.STATION_ID_COL):
        # Preserve the caller's evaluation split, including train+validation
        # refits for test. Recomputing date_masks here resets it to validation.
        tr = sub.loc[masks["train"].reindex(sub.index)]
        va = sub.loc[masks["validation"].reindex(sub.index)]
        preds, imps = _fit_partition(spec, factory, tr, va, feature_cols,
                                     records, str(sid))
        all_preds.append(preds)
        for im in imps:
            if len(im):
                im = im.copy()
                im[config.STATION_ID_COL] = sid
                all_imp.append(im)
    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return _finalize(spec, predictions, all_imp, feature_cols, records)


def _run_pooled(spec, df, factory, feature_cols, masks) -> dict:
    """Global (all stations) or Regional (per region) pooled models."""
    df2, cols = _add_categorical_codes(df, spec, feature_cols, masks["train"])
    all_preds, all_imp, records = [], [], []

    if spec.training_strategy == "global":
        groups = [("global", df2)]
    else:  # regional: split by region_id, fallback handled by 'Central'
        groups = list(df2.groupby("region_id"))

    for gname, gdf in groups:
        tr = gdf.loc[masks["train"].reindex(gdf.index)]
        va = gdf.loc[masks["validation"].reindex(gdf.index)]
        preds, imps = _fit_partition(spec, factory, tr, va, cols,
                                     records, str(gname))
        all_preds.append(preds)
        for im in imps:
            if len(im):
                im = im.copy()
                im["group"] = gname
                all_imp.append(im)
    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return _finalize(spec, predictions, all_imp, cols, records)


def _run_residual(spec, df, factory, feature_cols, masks) -> dict:
    """Global model + per-station local residual corrector (E1.3 / E1.5).

    Steps (plan E1.3):
      1. Fit global model on training rows.
      2. Out-of-fold global predictions on training rows (walk-forward).
      3. residual = y - global_oof on training rows.
      4. Fit local corrector per station to predict the residual.
      5. validation prediction = global(val) + local_residual(val).
    """
    df2, gcols = _add_categorical_codes(df, spec, feature_cols, masks["train"])
    evaluation_start = df2.loc[masks["validation"], config.DATE_COL].min()
    folds = splits.residual_oof_folds(df2, evaluation_start)

    # --- Global model, direct per-horizon (residual is defined per horizon) ---
    global_val_preds: dict[int, pd.DataFrame] = {}
    global_oof: dict[int, pd.DataFrame] = {}
    all_imp, records = [], []

    for h in config.FORECAST_HORIZONS:
        tcol = f"target_t{h}"
        train_rows = df2[masks["train"]].dropna(subset=[tcol])
        val_rows = df2[masks["validation"]]
        if evaluation_start < pd.Timestamp(config.TEST_START):
            val_rows = val_rows[val_rows[config.DATE_COL] + pd.Timedelta(days=h)
                                < pd.Timestamp(config.TEST_START)]
        if len(val_rows):
            train_rows = train_rows[
                train_rows[config.DATE_COL] + pd.Timedelta(days=h)
                < val_rows[config.DATE_COL].min()]

        # Global fit on full training window -> validation predictions.
        if len(train_rows) and len(val_rows) and (
                not spec.baseline_xgb or val_rows[tcol].notna().any()):
            yhat, est = _fit_predict_estimator(
                _horizon_factory(factory, h), spec.prefer_gpu, train_rows[gcols], train_rows[tcol], val_rows[gcols],
                val_rows[tcol] if spec.baseline_xgb else None,
            )
            vp = val_rows[[config.STATION_ID_COL, config.DATE_COL]].copy()
            vp["global_pred"] = yhat
            vp["y_true"] = val_rows[tcol].to_numpy()
            vp[gcols] = val_rows[gcols].to_numpy()
            global_val_preds[h] = vp
            _record(records, est, "global", h)
            all_imp.append(_importance_frame(est, gcols, f"global_h{h}"))

        # Out-of-fold global predictions on training rows.
        oof_parts = []
        for fold in folds:
            tmask, vmask = splits.fold_masks(df2, fold)
            ftr = df2[tmask].dropna(subset=[tcol])
            fva = df2[vmask].dropna(subset=[tcol])
            fva = fva[fva[config.DATE_COL] + pd.Timedelta(days=h) < evaluation_start]
            ftr = ftr[ftr[config.DATE_COL] + pd.Timedelta(days=h)
                      < fold.valid_start]
            if len(ftr) == 0 or len(fva) == 0:
                continue
            yhat, _ = _fit_predict_estimator(
                _horizon_factory(factory, h), spec.prefer_gpu, ftr[gcols], ftr[tcol], fva[gcols],
                train_dates=ftr[config.DATE_COL], horizon=h,
            )
            part = fva[[config.STATION_ID_COL, config.DATE_COL]].copy()
            part["global_pred"] = yhat
            part["y_true"] = fva[tcol].to_numpy()
            part[[*gcols]] = fva[gcols].to_numpy()
            oof_parts.append(part)
        if oof_parts:
            global_oof[h] = pd.concat(oof_parts, ignore_index=True)

    # --- Local residual corrector per station and horizon ---
    corrector = _mlp_corrector if spec.training_strategy == "global_local_mlp" else _tree_corrector
    final_preds = []
    for h in config.FORECAST_HORIZONS:
        if h not in global_val_preds:
            continue
        vp = global_val_preds[h].copy()
        oof = global_oof.get(h)
        residual_pred = np.zeros(len(vp))
        if oof is not None and len(oof):
            residual_pred = corrector(spec, _horizon_factory(factory, h), oof, vp, gcols,
                                      records, h)
        vp["y_pred"] = vp["global_pred"].to_numpy() + residual_pred
        vp["horizon"] = h
        final_preds.append(vp[[config.STATION_ID_COL, config.DATE_COL, "horizon", "y_true", "y_pred"]])

    predictions = pd.concat(final_preds, ignore_index=True) if final_preds else pd.DataFrame()
    return _finalize(spec, predictions, all_imp, gcols, records)


def _tree_corrector(spec, factory, oof, val_pred, gcols,
                    records=None, horizon=0) -> np.ndarray:
    """Per-station tree residual model (E1.3). Falls back to 0 when sparse."""
    residual_pred = np.zeros(len(val_pred))
    oof = oof.copy()
    oof["residual"] = oof["y_true"] - oof["global_pred"]
    val_by_station = {sid: idx for sid, idx in val_pred.groupby(config.STATION_ID_COL).groups.items()}
    for sid, otr in oof.groupby(config.STATION_ID_COL):
        if sid not in val_by_station:
            continue
        vidx = val_by_station[sid]
        va = val_pred.loc[vidx]
        if len(otr) < config.MLP_PARAMS.min_station_rows:
            continue  # fallback residual = 0
        predicted, est = _fit_predict_estimator(
            factory, spec.prefer_gpu, otr[gcols], otr["residual"], va[gcols],
            train_dates=otr[config.DATE_COL], horizon=horizon)
        if records is not None:
            _record(records, est, sid, horizon, "local_residual")
        residual_pred[val_pred.index.get_indexer(vidx)] = predicted
    return residual_pred


def _mlp_corrector(spec, factory, oof, val_pred, gcols,
                   records=None, horizon=0) -> np.ndarray:
    """Per-station MLP residual model (E1.5) with train-only standardisation."""
    import torch
    from torch import nn

    device = "cuda" if (spec.prefer_gpu and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(config.SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.SEED)

    p = config.MLP_PARAMS
    residual_pred = np.zeros(len(val_pred))
    oof = oof.copy()
    oof["residual"] = oof["y_true"] - oof["global_pred"]
    # MLP input = base features + the global prediction for that horizon.
    mlp_cols = gcols + ["global_pred"]
    val_by_station = {sid: idx for sid, idx in val_pred.groupby(config.STATION_ID_COL).groups.items()}

    for sid, otr in oof.groupby(config.STATION_ID_COL):
        if sid not in val_by_station or len(otr) < p.min_station_rows:
            continue  # fallback residual = 0 + flag implicit
        vidx = val_by_station[sid]
        va = val_pred.loc[vidx]

        Xtr = otr[mlp_cols].to_numpy(dtype=np.float32)
        ytr = otr["residual"].to_numpy(dtype=np.float32).reshape(-1, 1)
        Xva = va[mlp_cols].to_numpy(dtype=np.float32)

        # The shared notebook features contain NaNs. XGBoost handles them;
        # the MLP needs a finite, fixed placeholder before train-only scaling.
        Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
        Xva = np.nan_to_num(Xva, nan=0.0, posinf=0.0, neginf=0.0)

        mu, sigma = Xtr.mean(0), Xtr.std(0)
        sigma[sigma == 0] = 1.0
        Xtr = (Xtr - mu) / sigma
        Xva = (Xva - mu) / sigma

        # small holdout tail of the training residuals for early stopping
        n = len(Xtr)
        cut = max(1, int(n * 0.85))
        Xt, yt = Xtr[:cut], ytr[:cut]
        Xe, ye = Xtr[cut:], ytr[cut:]

        model = _build_mlp(Xtr.shape[1], p).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay)
        loss_fn = nn.MSELoss()
        Xt_t = torch.tensor(Xt, device=device)
        yt_t = torch.tensor(yt, device=device)
        Xe_t = torch.tensor(Xe if len(Xe) else Xt, device=device)
        ye_t = torch.tensor(ye if len(ye) else yt, device=device)

        best, best_state, bad = float("inf"), None, 0
        for _ in range(p.max_epochs):
            model.train()
            perm = torch.randperm(len(Xt_t), device=device)
            for i in range(0, len(Xt_t), p.batch_size):
                bidx = perm[i:i + p.batch_size]
                opt.zero_grad()
                out = model(Xt_t[bidx])
                loss = loss_fn(out, yt_t[bidx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vloss = loss_fn(model(Xe_t), ye_t).item()
            if vloss < best - 1e-6:
                best, bad = vloss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= p.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            rp = model(torch.tensor(Xva, device=device)).cpu().numpy().reshape(-1)
        if records is not None:
            records.append({"estimator": model.cpu(), "group": str(sid),
                            "horizon": int(horizon), "role": "local_residual_mlp",
                            "mean": mu, "std": sigma, "feature_columns": mlp_cols})
        residual_pred[val_pred.index.get_indexer(vidx)] = rp
    return residual_pred


def _build_mlp(in_dim: int, p: config.FixedMLPParams):
    from torch import nn
    layers: list = []
    prev = in_dim
    for units in p.hidden_units:
        layers += [nn.Linear(prev, units), nn.ReLU(), nn.Dropout(p.dropout)]
        prev = units
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Walk-forward evaluation (fold-level primary metric) + finalisation
# ---------------------------------------------------------------------------
def _finalize(spec: RunSpec, predictions: pd.DataFrame,
              importances: list[pd.DataFrame], feature_cols: list[str],
              model_records: list | None = None) -> dict:
    reports = metrics.build_reports(predictions) if len(predictions) else {
        "overall": {"micro": {}, "macro": {}},
        "by_horizon": pd.DataFrame(), "by_station": pd.DataFrame(),
        "station_horizon": pd.DataFrame(),
    }
    imp = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    return {
        "predictions": predictions,
        "reports": reports,
        "importance": imp,
        "feature_list": feature_cols,
        "model_records": model_records or [],
    }
