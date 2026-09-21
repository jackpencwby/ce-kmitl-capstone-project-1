"""Causal feature engineering and target construction.

All lag/rolling/spatial features are built *within* a station and use only
past information (Experimental_Plan.md sections 2, 3.1, 15). Targets are the
forward-shifted pm25 for horizons t+1..t+7.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import config

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def add_targets(df: pd.DataFrame, horizons: Iterable[int] = config.FORECAST_HORIZONS) -> pd.DataFrame:
    """Add target_t{h} = pm25 shifted -h within each station (no fill)."""
    df = df.copy()
    grp = df.groupby(config.STATION_ID_COL)[config.TARGET_COL]
    for h in horizons:
        df[f"target_t{h}"] = grp.shift(-h)
    return df


def target_cols(horizons: Iterable[int] = config.FORECAST_HORIZONS) -> list[str]:
    return [f"target_t{h}" for h in horizons]


# ---------------------------------------------------------------------------
# Target-station baseline features (causal)
# ---------------------------------------------------------------------------
def add_pm25_lag_roll(df: pd.DataFrame) -> pd.DataFrame:
    """PM2.5 lags and rolling stats computed on the shifted series.

    Rolling stats use the lag-1 series so day t never sees pm25 of day t
    (plan section 3.1 example).
    """
    df = df.copy()
    g = df.groupby(config.STATION_ID_COL)[config.TARGET_COL]

    for lag in config.PM25_LAGS:
        df[f"pm25_lag_{lag}"] = g.shift(lag)

    lagged = g.shift(1)
    lagged_by_station = lagged.groupby(df[config.STATION_ID_COL])
    for w in config.PM25_ROLL_WINDOWS:
        df[f"pm25_roll_mean_{w}"] = (
            lagged_by_station.rolling(w, min_periods=max(2, w // 2)).mean()
            .reset_index(level=0, drop=True)
        )
        df[f"pm25_roll_std_{w}"] = (
            lagged_by_station.rolling(w, min_periods=max(2, w // 2)).std()
            .reset_index(level=0, drop=True)
        )
    return df


def ensure_baseline_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure every source baseline feature column exists.

    Weather/fire/calendar features already exist in the master table; this is
    a guard so an experiment fails loudly rather than silently dropping a
    feature that was renamed upstream.
    """
    needed = (set(config.WEATHER_FEATURES) | set(config.FIRE_FEATURES)
              | set(config.CALENDAR_FEATURES) | set(config.AIR_QUALITY_FEATURES))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Master table is missing expected feature columns: {missing}")
    return df


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full non-spatial baseline: targets + pm25 lag/roll + guard checks."""
    df = ensure_baseline_columns(df)
    df = df.copy()
    for col in config.AIR_QUALITY_FEATURES:
        # A constant placeholder plus a flag preserves sparse satellite rows
        # without learning from validation/test or using future observations.
        df[f"{col}_missing"] = df[col].isna().astype("int8")
        df[col] = df[col].fillna(0.0)
    df = add_pm25_lag_roll(df)
    df = add_targets(df)
    return df


# ---------------------------------------------------------------------------
# Region assignment (E1.4)
# ---------------------------------------------------------------------------
# Thai provinces grouped into 4 broad regions. Central bundles East/West
# with Central to keep every region above a usable sample size.
_NORTH = {
    "เชียงใหม่", "เชียงราย", "ลำปาง", "ลำพูน", "แม่ฮ่องสอน", "น่าน", "พะเยา",
    "แพร่", "อุตรดิตถ์", "ตาก", "สุโขทัย", "พิษณุโลก", "เพชรบูรณ์", "พิจิตร",
    "กำแพงเพชร", "นครสวรรค์", "อุทัยธานี",
}
_NORTHEAST = {
    "เลย", "หนองคาย", "หนองบัวลำภู", "อุดรธานี", "บึงกาฬ", "นครพนม", "สกลนคร",
    "มุกดาหาร", "กาฬสินธุ์", "ขอนแก่น", "มหาสารคาม", "ร้อยเอ็ด", "ยโสธร",
    "อำนาจเจริญ", "อุบลราชธานี", "ศรีสะเกษ", "สุรินทร์", "บุรีรัมย์",
    "นครราชสีมา", "ชัยภูมิ",
}
_SOUTH = {
    "ชุมพร", "ระนอง", "สุราษฎร์ธานี", "พังงา", "ภูเก็ต", "กระบี่",
    "นครศรีธรรมราช", "ตรัง", "พัทลุง", "สตูล", "สงขลา", "ปัตตานี", "ยะลา",
    "นราธิวาส",
}


def assign_region(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `region_id` column from the province name (geography, not results).

    Anything not in North/Northeast/South falls into "Central" which also
    serves as the fallback bucket (plan E1.4 fallback rule).
    """
    df = df.copy()
    province_col = "province" if "province" in df.columns else None
    if province_col is None:
        LOGGER.warning("No province column; assigning all stations to region 'Central'")
        df["region_id"] = "Central"
        return df

    def _region(p: object) -> str:
        if not isinstance(p, str):
            return "Central"
        if p in _NORTH:
            return "North"
        if p in _NORTHEAST:
            return "Northeast"
        if p in _SOUTH:
            return "South"
        return "Central"

    df["region_id"] = df[province_col].map(_region)
    return df


# ---------------------------------------------------------------------------
# Spatial neighbor features (E4)
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compass bearing (degrees, 0-360) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def angular_difference_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two compass angles (0-180)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def build_neighbor_table(
    coords: pd.DataFrame,
    spatial: config.SpatialConfig = config.SPATIAL,
) -> pd.DataFrame:
    """For each station, its up-to-K nearest neighbours within the radius.

    Returns columns: station_id, neighbor_id, distance_km, bearing_deg.
    """
    rows = []
    recs = coords.to_dict("records")
    for tgt in recs:
        cand = []
        for nb in recs:
            if nb[config.STATION_ID_COL] == tgt[config.STATION_ID_COL]:
                continue
            d = haversine_km(tgt[config.LAT_COL], tgt[config.LON_COL],
                             nb[config.LAT_COL], nb[config.LON_COL])
            if d <= spatial.max_radius_km:
                b = initial_bearing_deg(tgt[config.LAT_COL], tgt[config.LON_COL],
                                       nb[config.LAT_COL], nb[config.LON_COL])
                cand.append((nb[config.STATION_ID_COL], d, b))
        cand.sort(key=lambda t: t[1])
        for nb_id, d, b in cand[: spatial.max_neighbors]:
            rows.append({
                config.STATION_ID_COL: tgt[config.STATION_ID_COL],
                "neighbor_id": nb_id,
                "distance_km": d,
                "bearing_deg": b,
            })
    return pd.DataFrame(rows)


def _neighbor_pm_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Wide (date x station_id) table of same-day pm25, for lookups."""
    wide = df.pivot_table(
        index=config.DATE_COL, columns=config.STATION_ID_COL,
        values=config.TARGET_COL, aggfunc="first",
    )
    return wide


def add_neighbor_features(
    df: pd.DataFrame,
    neighbor_table: pd.DataFrame,
    mode: str,
    spatial: config.SpatialConfig = config.SPATIAL,
) -> pd.DataFrame:
    """Add neighbor PM2.5 aggregate features, then lag them (no look-ahead).

    mode is one of: "unweighted", "distance", "wind".
    A daily neighbor aggregate is computed from *same-day* neighbor pm25,
    then lagged by 1/3/7 days so day t only sees neighbor values from the
    past (plan E4 leakage rules 4-5). Missing neighbors are renormalised out;
    calm wind / zero weight falls back to the distance-weighted value.
    """
    df = df.copy()
    wide = _neighbor_pm_wide(df)
    dates = wide.index

    # Precompute per-station neighbor lists.
    nbrs: dict[object, list[tuple[object, float, float]]] = {}
    for sid, sub in neighbor_table.groupby(config.STATION_ID_COL):
        nbrs[sid] = list(zip(sub["neighbor_id"], sub["distance_km"], sub["bearing_deg"]))

    # Daily wind-from per station (only needed for wind mode).
    wind_wide = None
    if mode == "wind":
        wind_wide = df.pivot_table(
            index=config.DATE_COL, columns=config.STATION_ID_COL,
            values=config.WIND_FROM_COL, aggfunc="first",
        )

    # Accumulate per-station aggregates in a dict, then build the frame once
    # (assigning columns one at a time fragments the DataFrame).
    agg_cols: dict[object, np.ndarray] = {}
    eps = spatial.epsilon_km
    nan_col = np.full(len(dates), np.nan)
    for sid in wide.columns:
        neighbor_list = nbrs.get(sid, [])
        nb_ids = [n[0] for n in neighbor_list if n[0] in wide.columns]
        if not nb_ids:
            agg_cols[sid] = nan_col
            continue
        nb_pm = wide[nb_ids]  # dates x neighbors
        dist = np.array([n[1] for n in neighbor_list if n[0] in wide.columns])
        bearing = np.array([n[2] for n in neighbor_list if n[0] in wide.columns])

        if mode == "unweighted":
            agg_cols[sid] = nb_pm.mean(axis=1, skipna=True).to_numpy()
            continue

        dist_w = 1.0 / (dist + eps)  # (n_neighbors,)
        if mode == "distance":
            w = np.broadcast_to(dist_w, nb_pm.shape).copy()
            agg_cols[sid] = _weighted_row_mean(nb_pm.values, w)
            continue

        if mode == "wind":
            # wind_from at target station per day.
            wf = wind_wide[sid].reindex(dates).values if sid in wind_wide.columns else np.full(len(dates), np.nan)
            wmat = np.zeros(nb_pm.shape)
            for j in range(len(nb_ids)):
                delta = np.array([
                    angular_difference_deg(w_i, bearing[j]) if not np.isnan(w_i) else np.nan
                    for w_i in wf
                ])
                cos_term = np.cos(np.radians(delta))
                cos_term = np.clip(cos_term, 0.0, None) ** spatial.wind_power
                wmat[:, j] = cos_term * dist_w[j]
            wind_agg = _weighted_row_mean(nb_pm.values, wmat)
            # Fallback to distance-weighted where wind weights sum to ~0.
            dist_full = np.broadcast_to(dist_w, nb_pm.shape).copy()
            dist_agg = _weighted_row_mean(nb_pm.values, dist_full)
            zero_wind = np.nan_to_num(wmat).sum(axis=1) <= 0
            wind_agg[zero_wind] = dist_agg[zero_wind]
            agg_cols[sid] = wind_agg
            continue

        raise ValueError(f"Unknown neighbor mode: {mode}")

    agg = pd.DataFrame(agg_cols, index=dates)

    # Melt aggregate back to long form keyed by (date, station_id). The pivot
    # turns station ids into column labels (object dtype); cast them back to
    # the original station_id dtype so the merge matches and does not upcast
    # df's station_id column to object.
    agg_long = agg.reset_index().melt(
        id_vars=config.DATE_COL, var_name=config.STATION_ID_COL, value_name="neighbor_pm_agg"
    )
    agg_long[config.STATION_ID_COL] = agg_long[config.STATION_ID_COL].astype(
        df[config.STATION_ID_COL].dtype
    )
    df = df.merge(agg_long, on=[config.DATE_COL, config.STATION_ID_COL], how="left")

    # Lag the aggregate (never same-day) within station.
    g = df.groupby(config.STATION_ID_COL)["neighbor_pm_agg"]
    for lag in spatial.neighbor_lags:
        df[f"neighbor_pm_lag_{lag}"] = g.shift(lag)
    df = df.drop(columns=["neighbor_pm_agg"])
    return df


def _weighted_row_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Row-wise weighted mean ignoring NaN pm values (renormalise weights)."""
    mask = ~np.isnan(values)
    w = np.where(mask, weights, 0.0)
    num = np.nansum(np.where(mask, values, 0.0) * w, axis=1)
    den = w.sum(axis=1)
    out = np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)
    return out


def neighbor_feature_cols(spatial: config.SpatialConfig = config.SPATIAL) -> list[str]:
    return [f"neighbor_pm_lag_{lag}" for lag in spatial.neighbor_lags]
