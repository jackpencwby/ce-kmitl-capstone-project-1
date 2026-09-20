"""Metrics and multi-level reporting (Experimental_Plan.md section 13).

Every metric is computed on the ug/m3 scale, on rows where the true target
is present. Reporting levels: by horizon, by station, station x horizon,
micro-average, macro-average (station-equal weight), and high-PM episodes.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import config


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom <= 0:
        return float("nan")
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, MSE, RMSE, R2, Bias on aligned, non-NaN arrays."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"n": 0, "mae": float("nan"), "mse": float("nan"),
                "rmse": float("nan"), "r2": float("nan"), "bias": float("nan")}
    err = yp - yt
    mse = float(np.mean(err ** 2))
    return {
        "n": int(len(yt)),
        "mae": float(np.mean(np.abs(err))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": _safe_r2(yt, yp),
        "bias": float(np.mean(err)),
    }


def high_pm_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    thresholds: Iterable[float] = config.HIGH_PM_THRESHOLDS) -> dict[str, float]:
    """MAE/RMSE restricted to days where the true value exceeds a threshold."""
    out: dict[str, float] = {}
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    for thr in thresholds:
        sel = yt > thr
        key = f"high_{thr:g}"
        if sel.sum() == 0:
            out[f"{key}_n"] = 0
            out[f"{key}_mae"] = float("nan")
            out[f"{key}_rmse"] = float("nan")
        else:
            err = yp[sel] - yt[sel]
            out[f"{key}_n"] = int(sel.sum())
            out[f"{key}_mae"] = float(np.mean(np.abs(err)))
            out[f"{key}_rmse"] = float(np.sqrt(np.mean(err ** 2)))
    return out


def build_reports(pred_df: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
    """Compute all reporting levels from a long prediction frame.

    pred_df must have columns: station_id, date, horizon, y_true, y_pred.
    Returns a dict with overall/by_horizon/by_station/station_horizon frames.
    """
    pred_df = pred_df.dropna(subset=["y_true"]).copy()

    # station x horizon
    sh_rows = []
    for (sid, h), sub in pred_df.groupby([config.STATION_ID_COL, "horizon"]):
        m = point_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        m.update({config.STATION_ID_COL: sid, "horizon": h})
        sh_rows.append(m)
    station_horizon = pd.DataFrame(sh_rows)

    # by horizon (micro across stations)
    bh_rows = []
    for h, sub in pred_df.groupby("horizon"):
        m = point_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        m["horizon"] = h
        bh_rows.append(m)
    by_horizon = pd.DataFrame(bh_rows)

    # by station (micro across horizons)
    bs_rows = []
    for sid, sub in pred_df.groupby(config.STATION_ID_COL):
        m = point_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        m.update(high_pm_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()))
        m[config.STATION_ID_COL] = sid
        bs_rows.append(m)
    by_station = pd.DataFrame(bs_rows)

    # micro-average (all rows)
    micro = point_metrics(pred_df["y_true"].to_numpy(), pred_df["y_pred"].to_numpy())
    micro.update(high_pm_metrics(pred_df["y_true"].to_numpy(), pred_df["y_pred"].to_numpy()))

    # macro-average: mean of per-station metrics (station-equal weight),
    # and mean of per-(station,horizon) rmse = primary metric.
    macro = {
        "macro_rmse_station": float(np.nanmean(by_station["rmse"])) if len(by_station) else float("nan"),
        "macro_mae_station": float(np.nanmean(by_station["mae"])) if len(by_station) else float("nan"),
        "macro_r2_station": float(np.nanmean(by_station["r2"])) if len(by_station) else float("nan"),
        # Primary metric: macro RMSE averaged across stations AND horizons.
        "primary_macro_rmse": float(np.nanmean(station_horizon["rmse"])) if len(station_horizon) else float("nan"),
        "worst_station_rmse": float(np.nanmax(by_station["rmse"])) if len(by_station) else float("nan"),
    }
    if len(by_horizon):
        bh = by_horizon.set_index("horizon")["rmse"]
        if 1 in bh.index and max(config.FORECAST_HORIZONS) in bh.index:
            macro["rmse_degradation_t1_to_t7"] = float(bh[max(config.FORECAST_HORIZONS)] - bh[1])

    overall = {"micro": micro, "macro": macro}
    return {
        "overall": overall,
        "by_horizon": by_horizon,
        "by_station": by_station,
        "station_horizon": station_horizon,
    }


def persistence_prediction(df_rows: pd.DataFrame) -> pd.DataFrame:
    """Naive baseline y_hat(t+h) = pm25(t). Expects columns pm25 + target_t{h}.

    Returns a long prediction frame comparable to model outputs.
    """
    records = []
    for h in config.FORECAST_HORIZONS:
        tcol = f"target_t{h}"
        sub = df_rows[[config.STATION_ID_COL, config.DATE_COL, config.TARGET_COL, tcol]].copy()
        sub = sub.rename(columns={config.TARGET_COL: "y_pred", tcol: "y_true"})
        sub["horizon"] = h
        records.append(sub)
    return pd.concat(records, ignore_index=True)
