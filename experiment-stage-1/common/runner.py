"""End-to-end orchestration shared by every experiment script.

An experiment script builds a ``RunSpec`` and calls :func:`execute`. This
module loads the data (local or GCS), builds the causal features (adding
spatial neighbour features only when the spec asks for them), resolves the
requested stations, runs the model, computes reports, and writes artifacts.
"""

from __future__ import annotations

import logging
import random
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import artifacts, config, data, features, metrics, models, splits

LOGGER = logging.getLogger(__name__)


def _seed_everything() -> None:
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    try:
        import torch
        torch.manual_seed(config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.SEED)
    except Exception:  # noqa: BLE001
        pass


def prepare_dataframe(spec: models.RunSpec, source: str) -> pd.DataFrame:
    """Load the master table and build all features the spec needs."""
    df = data.load_master(source=source)
    df = features.build_base_features(df)
    if spec.include_region_id or spec.training_strategy == "regional":
        df = features.assign_region(df)
    if spec.spatial_mode != "none":
        coords = data.station_coordinates(df)
        nbr = features.build_neighbor_table(coords)
        df = features.add_neighbor_features(df, nbr, mode=spec.spatial_mode)
    return df


def describe(spec: models.RunSpec, df: pd.DataFrame, stations: list) -> str:
    masks = splits.date_masks(df)
    folds = splits.walk_forward_folds(df)
    lines = [
        f"Run id            : {spec.run_id}",
        f"Algorithm         : {spec.algorithm}",
        f"Training strategy : {spec.training_strategy}",
        f"Forecast strategy : {spec.forecast_strategy}",
        f"Spatial mode      : {spec.spatial_mode}",
        f"Stations selected : {len(stations)}",
        f"  first ids       : {stations[:10]}",
        f"Rows total        : {len(df)}",
        f"  train rows      : {int(masks['train'].sum())}",
        f"  validation rows : {int(masks['validation'].sum())}",
        f"  test rows(locked): {int(masks['test'].sum())}",
        f"Validation start  : {config.VALIDATION_START}",
        f"Test start        : {config.TEST_START}",
        f"Walk-forward folds: {len(folds)}",
        f"Feature count     : {len(models.assemble_feature_columns(spec))}"
        + (" (+ station/region codes for pooled models)"
           if spec.include_station_id or spec.include_region_id else ""),
    ]
    return "\n".join(lines)


def execute(spec: models.RunSpec, args) -> int:
    """Run one experiment end-to-end. Returns a process exit code."""
    _seed_everything()
    spec.prefer_gpu = not args.prefer_cpu

    if args.check_gcs:
        ok = data.check_gcs_connection()
        LOGGER.info("GCS connection: %s", "OK" if ok else "FAILED (continuing)")

    df = prepare_dataframe(spec, source=args.source)
    stations = data.resolve_stations(df, args.stations)
    if args.max_stations is not None:
        stations = stations[: args.max_stations]

    summary = describe(spec, df, stations)
    print(summary)

    if not args.train:
        print("\n[dry run] Re-run with --train to fit models and write artifacts.")
        return 0

    device = artifacts.detect_device(prefer_gpu=spec.prefer_gpu)
    LOGGER.info("Device: %s (%s)", device["device"], device.get("gpu_model"))
    started = datetime.now(timezone.utc)

    result = models.run_holdout(spec, df, stations)
    predictions = result["predictions"]

    # Naive persistence baseline on the same validation cohort for reference.
    masks = splits.date_masks(df[df[config.STATION_ID_COL].isin(stations)])
    val_rows = df[df[config.STATION_ID_COL].isin(stations)][masks["validation"]]
    naive = metrics.persistence_prediction(val_rows)
    naive_reports = metrics.build_reports(naive)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    reports = result["reports"]
    macro = reports["overall"].get("macro", {})
    print("\n=== Validation results ===")
    print(f"Primary macro RMSE (station x horizon): {macro.get('primary_macro_rmse'):.4f}"
          if macro.get("primary_macro_rmse") == macro.get("primary_macro_rmse") else "Primary macro RMSE: n/a")
    print(f"Macro RMSE (station)                  : {macro.get('macro_rmse_station')}")
    print(f"Macro MAE  (station)                  : {macro.get('macro_mae_station')}")
    print(f"Worst-station RMSE                    : {macro.get('worst_station_rmse')}")
    print(f"Naive persistence primary macro RMSE  : "
          f"{naive_reports['overall']['macro'].get('primary_macro_rmse')}")
    print(f"Elapsed seconds                       : {elapsed:.1f}")

    run_dir = artifacts.make_run_dir(spec.run_id)
    run_config = {
        "algorithm": spec.algorithm,
        "training_strategy": spec.training_strategy,
        "forecast_strategy": spec.forecast_strategy,
        "spatial_mode": spec.spatial_mode,
        "include_station_id": spec.include_station_id,
        "include_region_id": spec.include_region_id,
        "elapsed_seconds": elapsed,
        "naive_primary_macro_rmse": naive_reports["overall"]["macro"].get("primary_macro_rmse"),
        "fixed_params": {
            "xgboost": config.XGB_PARAMS.as_dict(),
            "lightgbm": config.LGBM_PARAMS.as_dict(),
            "gbr": config.GBR_PARAMS.as_dict(),
        }.get(spec.algorithm, {}),
    }
    artifacts.save_run(
        run_dir=run_dir,
        run_id=spec.run_id,
        run_config=run_config,
        df=df,
        stations=stations,
        folds_manifest=splits.fold_manifest(splits.walk_forward_folds(df)),
        reports=reports,
        predictions=predictions,
        feature_list=result["feature_list"],
        feature_importance=result["importance"],
        training_log=summary,
    )
    print(f"\nArtifacts (local): {run_dir}")

    if getattr(args, "upload_gcs", False):
        base_prefix = getattr(args, "gcs_artifacts_prefix", None) or config.GCS_ARTIFACTS_PREFIX
        dest_prefix = f"{base_prefix.strip('/')}/{run_dir.name}"
        try:
            folder_uri = data.upload_dir_to_gcs(run_dir, dest_prefix)
            print(f"Artifacts (GCS) : {folder_uri}")
            if not getattr(args, "keep_local", True):
                shutil.rmtree(run_dir, ignore_errors=True)
                print(f"Removed local copy: {run_dir}")
        except Exception as error:  # noqa: BLE001 - upload must not lose a run
            LOGGER.error("GCS upload failed (local artifacts kept): %s", error)
            print(f"[warn] GCS upload failed, local artifacts kept at {run_dir}: {error}")

    return 0
