"""Resume only E2.3 test, retaining validation and checkpointing each horizon."""
from __future__ import annotations

import argparse
import json
import os
import gc
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd

from common import artifacts, cli, config, config_e2, data, metrics, models, runner
from common.training_progress import GBRProgress, summarize, write_atomic


def train_horizon(h, spec, df, stations, feature_list, checkpoint, log_every=10):
    """Run inside an isolated process; never mutate the parent's horizon list."""
    from threadpoolctl import threadpool_limits
    path = checkpoint / f"h{h}.joblib"
    if path.exists():
        return h
    print(f"Training test horizon {h} on CPU (PID {os.getpid()})", flush=True)
    original_horizons = config.FORECAST_HORIZONS
    progress = GBRProgress(checkpoint, h, original_horizons,
                           config_e2.params_for("gbr")["n_estimators"], log_every)
    original_monitor = spec.gbr_monitor
    try:
        config.FORECAST_HORIZONS = [h]
        spec.gbr_monitor = progress
        with threadpool_limits(limits=1):
            result = models.run_holdout(spec, df, stations, evaluation="test")
    except BaseException as exc:
        progress.finish("failed", exc)
        raise
    finally:
        config.FORECAST_HORIZONS = original_horizons
        spec.gbr_monitor = original_monitor
    if result["predictions"].empty or result["predictions"]["date"].min() < pd.Timestamp(config.TEST_START):
        raise ValueError("Invalid test prediction period")
    if result["feature_list"] != feature_list or len(result["model_records"]) != 1:
        raise ValueError("Unexpected test model structure")
    temp = path.with_suffix(".tmp")
    joblib.dump(result, temp)
    temp.replace(path)
    progress.finish("completed")
    print(f"Test horizon {h} checkpoint saved", flush=True)
    return h


def positive_workers(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("workers must be at least 1")
    return number


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--workers", type=positive_workers, default=2,
                        help="Independent CPU horizon processes (default: 2; RAM grows per worker).")
    parser.add_argument("--log-every", type=positive_workers, default=10,
                        help="Print progress every N trees (JSON updates every tree).")
    args = parser.parse_args()
    cli.configure_logging("INFO")
    root = args.run_dir.resolve()
    saved = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    spec = models.RunSpec(
        run_id="E2_GLOBAL__GBR__DIRECT__NO_NEIGHBOR__SEED42",
        algorithm="gbr", training_strategy="global", forecast_strategy="direct",
        spatial_mode="none", include_station_id=True, parameter_profile="e2",
        prefer_gpu=False,
    )
    expected = {key: getattr(spec, key) for key in (
        "run_id", "algorithm", "training_strategy", "forecast_strategy",
        "spatial_mode", "include_station_id", "include_region_id", "parameter_profile")}
    expected.update(seed=config.SEED, validation_start=config.VALIDATION_START,
                    test_start=config.TEST_START, forecast_horizons=list(config.FORECAST_HORIZONS),
                    fixed_params=config_e2.params_for("gbr"))
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Saved configuration differs: {key}")
    if artifacts._file_hash(config.LOCAL_MASTER_CSV) != manifest["master_sha256"]:
        raise ValueError("Dataset hash differs from validation run")
    if artifacts._lib_version("sklearn") != saved["sklearn_version"]:
        raise ValueError("scikit-learn version differs from validation run")
    horizons = list(config.FORECAST_HORIZONS)
    model_manifest = json.loads((root / "model/validation/manifest.json").read_text(encoding="utf-8"))
    if sorted(item["horizon"] for item in model_manifest["models"]) != horizons:
        raise ValueError("Validation model set is incomplete")
    for item in model_manifest["models"]:
        if not (root / "model/validation" / item["file"]).is_file():
            raise ValueError("Missing validation model")
    runner._seed_everything()
    df = runner.prepare_dataframe(spec, source="local")
    stations = data.resolve_stations(df, ["all"])
    current = artifacts.dataset_manifest(df, stations)
    for key in ("n_rows", "stations_used", "station_code_map", "date_min", "date_max"):
        if current[key] != manifest[key]:
            raise ValueError(f"Prepared dataset differs: {key}")
    feature_list = models.assemble_feature_columns(spec) + ["station_id_code"]
    if feature_list != (root / "feature_list.txt").read_text(encoding="utf-8").splitlines():
        raise ValueError("Feature list differs from validation run")
    print(f"Verified dataset, parameters, features and {len(stations)} stations.", flush=True)
    if args.check_only:
        return
    # Workers need only model inputs, identifiers and targets, not the full export.
    columns = list(dict.fromkeys(models.assemble_feature_columns(spec) +
                   [config.STATION_ID_COL, config.DATE_COL] + [f"target_t{h}" for h in horizons]))
    df = df.loc[df[config.STATION_ID_COL].isin(stations), columns].copy()
    gc.collect()
    if (root / "metrics_overall_test.json").exists():
        raise ValueError("Test results already exist; refusing to overwrite")
    checkpoint = root / "test_checkpoints"
    checkpoint.mkdir(exist_ok=True)
    status_path = root / "test_resume_status.json"
    workers = min(args.workers, len(horizons), os.cpu_count() or 1)
    status = {"state": "running", "pid": os.getpid(), "device": "cpu", "workers": workers,
              "started_at": datetime.now(timezone.utc).isoformat(), "completed_horizons": []}
    def update():
        status["progress"] = summarize(checkpoint, horizons, saved["fixed_params"]["n_estimators"])
        write_atomic(status_path, status)
    update()
    try:
        status["completed_horizons"] = [h for h in horizons if (checkpoint / f"h{h}.joblib").exists()]
        pending = [h for h in horizons if h not in status["completed_horizons"]]
        status["pending_horizons"] = pending.copy()
        update()
        print(f"Using {workers} parallel workers; pending horizons: {pending}", flush=True)
        # Spawned numerical libraries must not start their own large thread pools.
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[variable] = "1"
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(train_horizon, h, spec, df, stations, feature_list, checkpoint, args.log_every)
                       for h in pending]
            remaining = set(futures)
            while remaining:
                finished, remaining = wait(remaining, timeout=5, return_when=FIRST_COMPLETED)
                for future in finished:
                    h = future.result()
                    status["completed_horizons"] = sorted(status["completed_horizons"] + [h])
                    status["pending_horizons"].remove(h)
                update()
        del df
        gc.collect()
        results = [joblib.load(checkpoint / f"h{h}.joblib") for h in horizons]
        predictions = pd.concat([r["predictions"] for r in results], ignore_index=True)
        reports = metrics.build_reports(predictions)
        # Write test outputs only: validation provenance and artifacts stay intact.
        artifacts.write_json(root / "metrics_overall_test.json", reports["overall"])
        for key, filename in (("by_horizon", "metrics_by_horizon_test.csv"),
                              ("by_station", "metrics_by_station_test.csv"),
                              ("station_horizon", "metrics_station_horizon_test.csv")):
            reports[key].to_csv(root / filename, index=False)
        predictions.to_parquet(root / "predictions_test.parquet", index=False)
        pd.concat([r["importance"] for r in results], ignore_index=True).to_csv(
            root / "feature_importance_test.csv", index=False)
        artifacts.save_models(root, [rec for r in results for rec in r["model_records"]], "test", feature_list)
        status.update(state="completed", finished_at=datetime.now(timezone.utc).isoformat(),
                      primary_macro_rmse=reports["overall"]["macro"]["primary_macro_rmse"])
        update()
        print(f"Test complete: {status['primary_macro_rmse']=}", flush=True)
    except BaseException as exc:
        status.update(state="failed", error=repr(exc))
        update()
        raise


if __name__ == "__main__":
    main()
