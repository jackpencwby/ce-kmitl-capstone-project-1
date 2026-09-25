"""End-to-end orchestration shared by every experiment script.

An experiment script builds a ``RunSpec`` and calls :func:`execute`. This
module loads the data (local or GCS), builds the causal features (adding
spatial neighbour features only when the spec asks for them), resolves the
requested stations, runs the model, computes reports, and writes artifacts.
"""

from __future__ import annotations

import logging
import json
import random
import shutil
from dataclasses import asdict
from time import perf_counter
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
    df = features.add_exact_day_targets(df)
    if "xgboost_history_valid" not in df.columns:
        raise KeyError("Master table is missing xgboost_history_valid")
    if spec.include_region_id or spec.training_strategy == "regional":
        df = features.assign_region(df)
    if spec.spatial_mode != "none":
        coords = data.station_coordinates(df)
        nbr = features.build_neighbor_table(coords)
        df = features.add_neighbor_features(df, nbr, mode=spec.spatial_mode)
    # Eligibility applies to forecast origins, not observed neighbor sources.
    eligible_history = (df["xgboost_history_valid"].astype(str).str.strip()
                        .str.lower().isin(("true", "1", "1.0")))
    df = df.loc[eligible_history].copy()
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
        f"Walk-forward folds: {len(folds) if spec.walk_forward else 0} (executed when training)",
        f"Feature count     : {len(models.assemble_feature_columns(spec))}"
        + (" (+ station/region codes for pooled models)"
           if spec.include_station_id or spec.include_region_id else ""),
    ]
    return "\n".join(lines)


def _save_persistence_comparison(run_dir, split, comparison):
    artifacts.write_json(run_dir / f'persistence_comparison_{split}.json', {
        'n':comparison['n'], 'model':comparison['model']['overall'],
        'persistence':comparison['persistence']['overall'],
        'cohort':'paired finite station/date/horizon support'})


def _save_cv(run_dir, result, df):
    result['fold_metrics'].to_csv(run_dir / 'cv_fold_metrics.csv', index=False)
    result['model_selection'].to_csv(run_dir / 'cv_model_selection.csv', index=False)
    artifacts.write_json(run_dir / 'metrics_overall_cv.json', result['reports']['overall'])
    for key, name in [('by_horizon','metrics_by_horizon_cv.csv'),
                      ('by_station','metrics_by_station_cv.csv'),
                      ('station_horizon','metrics_station_horizon_cv.csv')]:
        result['reports'][key].to_csv(run_dir / name, index=False)
    try:
        result['predictions'].to_parquet(run_dir / 'predictions_cv.parquet', index=False)
    except ImportError:
        result['predictions'].to_csv(run_dir / 'predictions_cv.csv', index=False)
    _save_persistence_comparison(run_dir, 'cv',
        metrics.paired_persistence_reports(result['predictions'], df))


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
    if not stations:
        raise ValueError('No eligible stations remain; check missingness and station selection.')

    summary = describe(spec, df, stations)
    print(summary)

    if not args.train:
        print("\n[dry run] Re-run with --train to fit models and write artifacts.")
        return 0

    device = artifacts.detect_device(prefer_gpu=spec.prefer_gpu)
    LOGGER.info("Device: %s (%s)", device["device"], device.get("gpu_model"))
    started = datetime.now(timezone.utc)

    cv_result = None
    cv_started = perf_counter()
    if spec.walk_forward:
        cv_result = models.run_walk_forward(spec, df, stations)
    cv_seconds = perf_counter() - cv_started if spec.walk_forward else 0.
    validation_started = perf_counter()

    result = models.run_holdout(spec, df, stations)
    predictions = result["predictions"]

    # Naive persistence baseline on the same validation cohort for reference.
    masks = splits.date_masks(df[df[config.STATION_ID_COL].isin(stations)])
    val_rows = df[df[config.STATION_ID_COL].isin(stations)][masks["validation"]]
    validation_pair = metrics.paired_persistence_reports(predictions, val_rows)
    naive_reports = validation_pair['persistence']
    validation_seconds = perf_counter() - validation_started

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    reports = result["reports"]
    macro = reports["overall"].get("macro", {})
    print("\n=== Validation results ===")
    primary_rmse = macro.get('primary_macro_rmse')
    print(f"Primary macro RMSE (station x horizon): {primary_rmse:.4f}"
          if primary_rmse is not None and np.isfinite(primary_rmse) else "Primary macro RMSE: n/a")
    print(f"Macro RMSE (station)                  : {macro.get('macro_rmse_station')}")
    print(f"Macro MAE  (station)                  : {macro.get('macro_mae_station')}")
    print(f"Worst-station RMSE                    : {macro.get('worst_station_rmse')}")
    print(f"Naive persistence primary macro RMSE  : "
          f"{naive_reports['overall']['macro'].get('primary_macro_rmse')}")
    print(f"Paired model primary macro RMSE       : "
          f"{validation_pair['model']['overall']['macro'].get('primary_macro_rmse')} "
          f"(n={validation_pair['n']})")
    print(f"Elapsed seconds                       : {elapsed:.1f}")

    run_dir = artifacts.make_run_dir(spec.run_id)
    if cv_result is not None:
        _save_cv(run_dir, cv_result, df)
    _save_persistence_comparison(run_dir, 'validation', validation_pair)
    run_config = {
        "algorithm": spec.algorithm,
        "training_strategy": spec.training_strategy,
        "forecast_strategy": spec.forecast_strategy,
        "spatial_mode": spec.spatial_mode,
        "include_station_id": spec.include_station_id,
        "include_region_id": spec.include_region_id,
        "elapsed_seconds": elapsed,
        "cv_seconds": cv_seconds,
        "validation_seconds": validation_seconds,
        "evaluation_splits": ['cv','validation','test'] if spec.walk_forward else ['validation','test'],
        "walk_forward_executed": spec.walk_forward,
        "complete_target_evaluation": spec.complete_target_evaluation,
        "selection_policy": ('training_tail_85_15_refit' if spec.selection_patience
                             else 'validation_early_stopping' if spec.baseline_xgb else 'fixed_budget'),
        "selection_patience": spec.selection_patience,
        "paired_validation_n": validation_pair['n'],
        "paired_validation_model_primary_macro_rmse": validation_pair['model']['overall']['macro'].get('primary_macro_rmse'),
        "spatial_config": asdict(config.SPATIAL) if spec.spatial_mode != 'none' else None,
        "naive_primary_macro_rmse": naive_reports["overall"]["macro"].get("primary_macro_rmse"),
        "fixed_params": ({"xgboost": config.xgb_params_for_horizon(1),
            "lightgbm": config.LGBM_PARAMS.as_dict(),
            "gbr": config.GBR_PARAMS.as_dict(),
        }.get(spec.algorithm, {})),
        "params_by_horizon": ({str(h): config.xgb_params_for_horizon(h)
                               for h in config.FORECAST_HORIZONS}
                              if spec.algorithm == "xgboost" else {}),
        "params_source": ("best_params_t1.json ... best_params_t7.json"
                          if spec.algorithm == "xgboost" else None),
        "early_stopping_rounds": (config.BASELINE_EARLY_STOPPING_ROUNDS
                                  if spec.baseline_xgb else None),
    }
    artifacts.save_run(
        run_dir=run_dir,
        run_id=spec.run_id,
        run_config=run_config,
        df=df,
        stations=stations,
        folds_manifest=splits.fold_manifest(splits.walk_forward_folds(df) if spec.walk_forward else []),
        reports=reports,
        predictions=predictions,
        feature_list=result["feature_list"],
        feature_importance=result["importance"],
        training_log=summary,
        model_records=result["model_records"],
        prefer_gpu=spec.prefer_gpu,
    )
    # Preserve actual validation device for E1's reused estimators at test.
    run_config['actual_device'] = json.loads((run_dir / 'config.json').read_text())['actual_device']
    # E1 uses validation for early stopping, so reuse the selected estimators
    # on test. Other experiment families fit a separate train+validation model.
    test_started = perf_counter()
    if spec.run_id.startswith("E1_") and spec.baseline_xgb:
        test_result = models.predict_e1_test_from_validation_models(
            spec, df, stations, result["model_records"])
    else:
        test_result = models.run_holdout(spec, df, stations, evaluation="test")
    del result
    test_reports = test_result["reports"]
    test_rows = df[df[config.STATION_ID_COL].isin(stations)][masks["test"]]
    test_pair = metrics.paired_persistence_reports(test_result['predictions'], test_rows)
    naive_test = test_pair['persistence']
    test_seconds = perf_counter() - test_started
    _save_persistence_comparison(run_dir, 'test', test_pair)
    test_macro = test_reports["overall"].get("macro", {})
    print("\n=== Test results ===")
    print(f"Primary macro RMSE (station x horizon): {test_macro.get('primary_macro_rmse')}")
    print(f"Naive persistence primary macro RMSE  : "
          f"{naive_test['overall']['macro'].get('primary_macro_rmse')}")
    print(f"Paired model primary macro RMSE       : "
          f"{test_pair['model']['overall']['macro'].get('primary_macro_rmse')} (n={test_pair['n']})")
    artifacts.save_run(
        run_dir=run_dir, run_id=spec.run_id,
        run_config={**run_config,
                    "test_seconds": test_seconds,
                    "elapsed_seconds": (datetime.now(timezone.utc)-started).total_seconds(),
                    "paired_test_n": test_pair['n'],
                    "paired_test_model_primary_macro_rmse": test_pair['model']['overall']['macro'].get('primary_macro_rmse'),
                    "test_naive_primary_macro_rmse": naive_test["overall"]["macro"].get("primary_macro_rmse")},
        df=df, stations=stations,
        folds_manifest=splits.fold_manifest(splits.walk_forward_folds(df) if spec.walk_forward else []),
        reports=test_reports, predictions=test_result["predictions"],
        feature_list=test_result["feature_list"],
        feature_importance=test_result["importance"],
        training_log=summary, split="test",
        model_records=test_result["model_records"],
        prefer_gpu=spec.prefer_gpu,
    )
    if spec.run_id.startswith("E1_") and spec.baseline_xgb:
        validation_manifest = run_dir / "model" / "validation" / "manifest.json"
        manifest = json.loads(validation_manifest.read_text(encoding="utf-8"))
        for item in manifest["models"]:
            item["file"] = f"../validation/{item['file']}"
            if "training_history_file" in item:
                item["training_history_file"] = (
                    f"../validation/{item['training_history_file']}")
        manifest["test_uses_validation_selected_models"] = True
        artifacts.write_json(run_dir / "model" / "test" / "manifest.json", manifest)
    if spec.run_id.startswith("E1_"):
        from . import notebook_outputs
        notebook_outputs.export(
            run_dir, df,
            {"validation": predictions, "test": test_result["predictions"]},
            test_result["feature_list"],
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
