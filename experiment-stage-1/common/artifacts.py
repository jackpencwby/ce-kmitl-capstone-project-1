"""Artifact writing per Experimental_Plan.md section 16.

Each run writes to artifacts/<run_id>/ with config, manifests, metrics,
predictions, feature list, importances and a training log.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import config

LOGGER = logging.getLogger(__name__)


def _lib_version(name: str) -> Optional[str]:
    try:
        module = __import__(name)
        return getattr(module, "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return None
    return None


def _file_hash(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_device(prefer_gpu: bool = True) -> dict[str, Optional[str]]:
    """Report the compute device and library/GPU info for the run."""
    info: dict[str, Optional[str]] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": _lib_version("torch"),
        "xgboost_version": _lib_version("xgboost"),
        "lightgbm_version": _lib_version("lightgbm"),
        "sklearn_version": _lib_version("sklearn"),
        "cuda_version": None,
        "gpu_model": None,
        "device": "cpu",
    }
    try:
        import torch
        if prefer_gpu and torch.cuda.is_available():
            info["device"] = "cuda"
            info["cuda_version"] = torch.version.cuda
            info["gpu_model"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return info


def make_run_dir(run_id: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = config.ARTIFACTS_DIR / f"{run_id}__{ts}"
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, obj: dict) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    path.write_text(json.dumps(obj, indent=2, default=_default, ensure_ascii=False), encoding="utf-8")


def dataset_manifest(df: pd.DataFrame, stations: list) -> dict:
    return {
        "master_local_path": str(config.LOCAL_MASTER_CSV),
        "dataset_gcs_prefix": config.GCS_DATA_PREFIX,
        "station_file_pattern": f"<station>/{config.GCS_STATION_FILENAME}",
        "co_aod_missing_policy": "zero placeholder with per-feature missing indicator",
        "master_sha256": _file_hash(config.LOCAL_MASTER_CSV),
        "n_rows": int(len(df)),
        "date_min": str(df[config.DATE_COL].min().date()),
        "date_max": str(df[config.DATE_COL].max().date()),
        "n_stations_total": int(df[config.STATION_ID_COL].nunique()),
        "n_stations_used": len(stations),
        "stations_used": [str(s) for s in stations],
        "git_commit": _git_commit(),
    }


def save_run(
    run_dir: Path,
    run_id: str,
    run_config: dict,
    df: pd.DataFrame,
    stations: list,
    folds_manifest: pd.DataFrame,
    reports: dict,
    predictions: pd.DataFrame,
    feature_list: list[str],
    feature_importance: Optional[pd.DataFrame] = None,
    training_log: str = "",
) -> None:
    """Persist the full artifact set for one run."""
    device = detect_device()
    full_config = {"run_id": run_id, **run_config, **device,
                   "seed": config.SEED,
                   "validation_start": config.VALIDATION_START,
                   "test_start": config.TEST_START,
                   "forecast_horizons": list(config.FORECAST_HORIZONS)}
    write_json(run_dir / "config.json", full_config)
    write_json(run_dir / "dataset_manifest.json", dataset_manifest(df, stations))
    folds_manifest.to_csv(run_dir / "fold_manifest.csv", index=False)

    write_json(run_dir / "metrics_overall.json", reports["overall"])
    reports["by_horizon"].to_csv(run_dir / "metrics_by_horizon.csv", index=False)
    reports["by_station"].to_csv(run_dir / "metrics_by_station.csv", index=False)
    reports["station_horizon"].to_csv(run_dir / "metrics_station_horizon.csv", index=False)

    # Predictions: parquet if pyarrow available, else CSV.
    pred_path = run_dir / "predictions_validation.parquet"
    try:
        predictions.to_parquet(pred_path, index=False)
    except Exception:  # noqa: BLE001
        pred_path = run_dir / "predictions_validation.csv"
        predictions.to_csv(pred_path, index=False)

    (run_dir / "feature_list.txt").write_text("\n".join(feature_list), encoding="utf-8")
    if feature_importance is not None and len(feature_importance):
        feature_importance.to_csv(run_dir / "feature_importance.csv", index=False)
    (run_dir / "training_log.txt").write_text(training_log, encoding="utf-8")
    LOGGER.info("Saved artifacts to %s", run_dir)
