"""Central, frozen configuration for every Stage 1 experiment.

Experimental_Plan.md section 3 requires that a fixed set of rules stay
constant across every run so results are comparable. Those rules live here
and must not be changed per-experiment. If a dataset revision needs a
different split date, change it here once, not inside an experiment script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
STAGE1_DIR: Final[Path] = REPO_ROOT / "experiment-stage-1"
ARTIFACTS_DIR: Final[Path] = STAGE1_DIR / "artifacts"

# Local copy of the master preprocessed table (mirror of the GCS object).
LOCAL_MASTER_CSV: Final[Path] = REPO_ROOT / "clean-data_preprocess_all_stations_daily.csv"

# The same table in Cloud Storage (S3-compatible interoperability endpoint).
# The object path below matches preprocessing_summary.json -> output_location.
GCS_MASTER_OBJECT: Final[str] = "clean-data/preprocess/all_stations_daily.csv"

# ---------------------------------------------------------------------------
# Frozen experiment rules (Experimental_Plan.md section 3)
# ---------------------------------------------------------------------------
SEED: Final[int] = 42

# Time split. The plan lists these as the current configuration; if the
# dataset revision differs, override here only (never per experiment).
VALIDATION_START: Final[str] = "2026-05-08"
TEST_START: Final[str] = "2026-07-07"

# Forecast horizons t+1 .. t+7.
FORECAST_HORIZONS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7)

# Walk-forward expanding-window folds inside Train+Validation.
N_WALK_FORWARD_FOLDS: Final[int] = 4

# Embargo/gap between end of a training label window and the start of the
# next validation prediction origin (plan section 15: >= max horizon).
EMBARGO_DAYS: Final[int] = max(FORECAST_HORIZONS)

# Drop groups whose pm25 target is missing more than this fraction
# (weather-only stations). Matches FEATURE_SELECTION.md.
MAX_TARGET_NAN_FRAC: Final[float] = 0.30

# High-PM2.5 episode thresholds (ug/m3) for episode metrics (plan 13.2).
HIGH_PM_THRESHOLDS: Final[tuple[float, ...]] = (37.5, 75.0)

# Identifier columns.
STATION_ID_COL: Final[str] = "station_id"
STATION_NAME_COL: Final[str] = "station_name"
DATE_COL: Final[str] = "date"
SEGMENT_ID_COL: Final[str] = "segment_id"
TARGET_COL: Final[str] = "pm25"
LAT_COL: Final[str] = "lat"
LON_COL: Final[str] = "long"


# ---------------------------------------------------------------------------
# Feature baseline (Experimental_Plan.md section 3.1)
# ---------------------------------------------------------------------------
# PM2.5 lags of the target station.
PM25_LAGS: Final[tuple[int, ...]] = (1, 3, 7, 30)

# Rolling windows built on the *shifted* pm25 series (no look-ahead).
PM25_ROLL_WINDOWS: Final[tuple[int, ...]] = (7,)

# Same-day weather features (measured at day t, allowed as of day t).
WEATHER_FEATURES: Final[tuple[str, ...]] = (
    "temperature_avg",
    "temperature_range",
    "humidity_avg",
    "humidity_range",
    "pressure_avg",
    "pressure_range",
    "wind_speed_avg",
    "wind_direction_avg_sin",
    "wind_direction_avg_cos",
    "log1p_rainfall",
    "rain_event",
)

# Fire / hotspot activity (measured at day t).
FIRE_FEATURES: Final[tuple[str, ...]] = (
    "hotspot_present",
    "log1p_hotspot_count",
    "log1p_hotspot_frp_sum",
)

# Calendar (known future) features.
CALENDAR_FEATURES: Final[tuple[str, ...]] = (
    "year_index",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
)

# Wind direction column used for spatial wind weighting (degrees, "wind from").
WIND_FROM_COL: Final[str] = "wind_direction_avg"


@dataclass(frozen=True)
class SpatialConfig:
    """Neighbor construction settings (Experiment 4). Frozen defaults."""

    max_radius_km: float = 100.0
    max_neighbors: int = 5
    epsilon_km: float = 1.0
    wind_power: float = 2.0  # exponent p in cos(delta)^p
    neighbor_lags: tuple[int, ...] = (1, 3, 7)


SPATIAL: Final[SpatialConfig] = SpatialConfig()


# ---------------------------------------------------------------------------
# Region mapping for Regional model (E1.4). Derived from geography, not
# validation results (plan section 6 E1.4). Thailand provinces -> region.
# ---------------------------------------------------------------------------
REGION_BY_PROVINCE: Final[dict[str, str]] = {}  # populated in features.assign_region


@dataclass
class FixedXGBParams:
    """Fixed reasonable defaults for Stage 1 screening (plan section 5, 11.1).

    Not tuned in Stage 1. GPU is used when available; callers fall back to
    CPU without changing anything else.
    """

    n_estimators: int = 600
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 5.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    def as_dict(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "gamma": self.gamma,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }


@dataclass
class FixedLGBMParams:
    """Fixed reasonable defaults for LightGBM (plan section 11.2)."""

    n_estimators: int = 600
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = 8
    min_child_samples: int = 30
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0

    def as_dict(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
        }


@dataclass
class FixedGBRParams:
    """Fixed reasonable defaults for GradientBoostingRegressor (plan 11.3)."""

    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 3
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    subsample: float = 0.8
    max_features: float = 0.8

    def as_dict(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "subsample": self.subsample,
            "max_features": self.max_features,
        }


@dataclass
class FixedMLPParams:
    """Fixed reasonable defaults for the MLP residual corrector (plan 11.4)."""

    hidden_units: tuple[int, ...] = (64, 32)
    dropout: float = 0.2
    weight_decay: float = 1e-4
    learning_rate: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 15
    min_station_rows: int = 120  # fallback to residual=0 below this


XGB_PARAMS: Final[FixedXGBParams] = FixedXGBParams()
LGBM_PARAMS: Final[FixedLGBMParams] = FixedLGBMParams()
GBR_PARAMS: Final[FixedGBRParams] = FixedGBRParams()
MLP_PARAMS: Final[FixedMLPParams] = FixedMLPParams()


def baseline_feature_list() -> list[str]:
    """The non-spatial baseline feature columns produced by features.py.

    Order matters only for reproducibility of feature_list.txt; the model
    is order-invariant.
    """
    feats: list[str] = []
    feats += [f"pm25_lag_{lag}" for lag in PM25_LAGS]
    feats += [f"pm25_roll_mean_{w}" for w in PM25_ROLL_WINDOWS]
    feats += [f"pm25_roll_std_{w}" for w in PM25_ROLL_WINDOWS]
    feats += list(WEATHER_FEATURES)
    feats += list(FIRE_FEATURES)
    feats += list(CALENDAR_FEATURES)
    return feats
