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
from typing import Callable, Optional

import numpy as np
import pandas as pd

from . import config, features, metrics, splits

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estimator factories (fixed reasonable defaults, plan section 5 & 11)
# ---------------------------------------------------------------------------
def make_xgb(prefer_gpu: bool = True):
    import xgboost as xgb
    params = config.XGB_PARAMS.as_dict()
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
    params = config.GBR_PARAMS.as_dict()
    return GradientBoostingRegressor(random_state=config.SEED, **params)


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
) -> tuple[np.ndarray, object]:
    """Fit a fresh estimator on training rows, predict validation rows."""
    est = factory(prefer_gpu=prefer_gpu)
    est.fit(X_tr, y_tr)
    return est.predict(X_val), est


def _importance_frame(est, feature_cols: list[str], tag: str) -> pd.DataFrame:
    imp = getattr(est, "feature_importances_", None)
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
        tr = train_df.dropna(subset=feature_cols + tcols)
        va = val_df.dropna(subset=feature_cols)
        if len(tr) == 0 or len(va) == 0:
            return pd.DataFrame(columns=id_cols + ["horizon", "y_true", "y_pred"]), importances
        from sklearn.multioutput import MultiOutputRegressor
        base = factory(prefer_gpu=spec.prefer_gpu)
        model = MultiOutputRegressor(base)
        model.fit(tr[feature_cols], tr[tcols])
        yhat = np.asarray(model.predict(va[feature_cols]))
        for j, h in enumerate(config.FORECAST_HORIZONS):
            block = va[id_cols].copy()
            block["horizon"] = h
            block["y_true"] = va[f"target_t{h}"].to_numpy()
            block["y_pred"] = yhat[:, j]
            preds.append(block)
        # Importance from the first sub-estimator (representative).
        try:
            importances.append(_importance_frame(model.estimators_[0], feature_cols, "multi_h1"))
        except Exception:  # noqa: BLE001
            pass
        return pd.concat(preds, ignore_index=True), importances

    # Direct: one estimator per horizon.
    for h in config.FORECAST_HORIZONS:
        tcol = f"target_t{h}"
        tr = train_df.dropna(subset=feature_cols + [tcol])
        va = val_df.dropna(subset=feature_cols)
        if len(tr) == 0 or len(va) == 0:
            continue
        yhat, est = _fit_predict_estimator(
            factory, spec.prefer_gpu, tr[feature_cols], tr[tcol], va[feature_cols]
        )
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
def run_holdout(spec: RunSpec, df: pd.DataFrame, stations: list) -> dict:
    """Fit on the training window and predict the validation holdout.

    Returns predictions, reports, feature importances, and the feature list.
    Dispatches on spec.training_strategy.
    """
    factory = ESTIMATOR_FACTORIES[spec.algorithm]
    feature_cols = assemble_feature_columns(spec)
    masks = splits.date_masks(df)

    df = df[df[config.STATION_ID_COL].isin(stations)].copy()
    masks = splits.date_masks(df)

    if spec.training_strategy == "local":
        return _run_local(spec, df, factory, feature_cols, masks)
    if spec.training_strategy in ("global", "regional"):
        return _run_pooled(spec, df, factory, feature_cols, masks)
    if spec.training_strategy in ("global_local_tree", "global_local_mlp"):
        return _run_residual(spec, df, factory, feature_cols, masks)
    raise ValueError(f"Unknown training strategy: {spec.training_strategy}")


def _run_local(spec, df, factory, feature_cols, masks) -> dict:
    """One independent model set per station (E1.1)."""
    all_preds, all_imp = [], []
    for sid, sub in df.groupby(config.STATION_ID_COL):
        sub_masks = splits.date_masks(sub)
        tr, va = sub[sub_masks["train"]], sub[sub_masks["validation"]]
        preds, imps = _fit_partition(spec, factory, tr, va, feature_cols)
        all_preds.append(preds)
        for im in imps:
            if len(im):
                im = im.copy()
                im[config.STATION_ID_COL] = sid
                all_imp.append(im)
    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return _finalize(spec, predictions, all_imp, feature_cols)


def _run_pooled(spec, df, factory, feature_cols, masks) -> dict:
    """Global (all stations) or Regional (per region) pooled models."""
    df2, cols = _add_categorical_codes(df, spec, feature_cols, masks["train"])
    all_preds, all_imp = [], []

    if spec.training_strategy == "global":
        groups = [("global", df2)]
    else:  # regional: split by region_id, fallback handled by 'Central'
        groups = list(df2.groupby("region_id"))

    for gname, gdf in groups:
        gmasks = splits.date_masks(gdf)
        tr, va = gdf[gmasks["train"]], gdf[gmasks["validation"]]
        preds, imps = _fit_partition(spec, factory, tr, va, cols)
        all_preds.append(preds)
        for im in imps:
            if len(im):
                im = im.copy()
                im["group"] = gname
                all_imp.append(im)
    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return _finalize(spec, predictions, all_imp, cols)


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
    folds = splits.walk_forward_folds(df2)

    # --- Global model, direct per-horizon (residual is defined per horizon) ---
    global_val_preds: dict[int, pd.DataFrame] = {}
    global_oof: dict[int, pd.DataFrame] = {}
    all_imp = []

    for h in config.FORECAST_HORIZONS:
        tcol = f"target_t{h}"
        train_rows = df2[masks["train"]].dropna(subset=gcols + [tcol])
        val_rows = df2[masks["validation"]].dropna(subset=gcols)

        # Global fit on full training window -> validation predictions.
        if len(train_rows) and len(val_rows):
            yhat, est = _fit_predict_estimator(
                factory, spec.prefer_gpu, train_rows[gcols], train_rows[tcol], val_rows[gcols]
            )
            vp = val_rows[[config.STATION_ID_COL, config.DATE_COL]].copy()
            vp["global_pred"] = yhat
            vp["y_true"] = val_rows[tcol].to_numpy()
            global_val_preds[h] = vp
            all_imp.append(_importance_frame(est, gcols, f"global_h{h}"))

        # Out-of-fold global predictions on training rows.
        oof_parts = []
        for fold in folds:
            tmask, vmask = splits.fold_masks(df2, fold)
            ftr = df2[tmask].dropna(subset=gcols + [tcol])
            fva = df2[vmask].dropna(subset=gcols + [tcol])
            if len(ftr) == 0 or len(fva) == 0:
                continue
            yhat, _ = _fit_predict_estimator(
                factory, spec.prefer_gpu, ftr[gcols], ftr[tcol], fva[gcols]
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
            residual_pred = corrector(spec, factory, oof, vp, gcols)
        vp["y_pred"] = vp["global_pred"].to_numpy() + residual_pred
        vp["horizon"] = h
        final_preds.append(vp[[config.STATION_ID_COL, config.DATE_COL, "horizon", "y_true", "y_pred"]])

    predictions = pd.concat(final_preds, ignore_index=True) if final_preds else pd.DataFrame()
    return _finalize(spec, predictions, all_imp, gcols)


def _tree_corrector(spec, factory, oof, val_pred, gcols) -> np.ndarray:
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
        est = factory(prefer_gpu=spec.prefer_gpu)
        est.fit(otr[gcols], otr["residual"])
        residual_pred[val_pred.index.get_indexer(vidx)] = est.predict(va[gcols])
    return residual_pred


def _mlp_corrector(spec, factory, oof, val_pred, gcols) -> np.ndarray:
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
              importances: list[pd.DataFrame], feature_cols: list[str]) -> dict:
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
    }
