#!/usr/bin/env python3
"""Stage 2 LightGBM tuning on CPU.

The four experiments defined in ``experiment-stage-2 - Sheet1.csv`` are:

1. Global single-output forecasting (one target/horizon per estimator)
2. Global multi-output forecasting (one forecaster returns t+1...t+7 together)
3. Regional single-output forecasting
4. Regional multi-output forecasting

Hyperparameters are sampled without replacement from the supplied finite grid.
Trials use the validation holdout only.  The selected configuration is then
refit on train+validation and evaluated exactly once on the locked test split.
LightGBM is explicitly configured with ``device_type='cpu'`` for a portable
Windows training workflow.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from sklearn.multioutput import MultiOutputRegressor


# Stage 2 intentionally reuses Stage 1's frozen data contract, causal feature
# construction, splits, metrics, and artifact schema.  The directory name has
# a hyphen, so it must be placed on sys.path rather than imported as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_DIR = REPO_ROOT / "experiment-stage-1"
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

from common import artifacts, config, data, metrics, models, runner, splits  # noqa: E402


LOGGER = logging.getLogger(__name__)

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

EXPERIMENTS = {
    1: dict(scope="global", forecast="direct", label="GLOBAL__SINGLE_OUTPUT"),
    2: dict(scope="global", forecast="multi", label="GLOBAL__MULTI_OUTPUT"),
    3: dict(scope="regional", forecast="direct", label="REGIONAL__SINGLE_OUTPUT"),
    4: dict(scope="regional", forecast="multi", label="REGIONAL__MULTI_OUTPUT"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), type=int, required=True)
    parser.add_argument("--train", action="store_true", help="Run tuning and write artifacts.")
    parser.add_argument("--source", choices=["auto", "local", "gcs"], default="auto")
    parser.add_argument("--stations", nargs="+", default=["all"])
    parser.add_argument("--max-stations", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=40,
                        help="Randomly sample this many unique configurations (default: 40).")
    parser.add_argument("--random-seed", type=int, default=config.SEED)
    parser.add_argument("--skip-test", action="store_true",
                        help="Tune on validation and save the winner without touching the test split.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.n_trials <= 0:
        parser.error("--n-trials must be positive")
    if args.n_trials > GRID_SIZE:
        parser.error(f"--n-trials cannot exceed the finite grid size ({GRID_SIZE:,})")
    return args


def require_lightgbm() -> None:
    """Verify that LightGBM is importable and can run a small CPU fit."""
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise RuntimeError("LightGBM is missing; install experiment-stage-2/requirements.txt") from error

    try:
        probe = lgb.LGBMRegressor(
            objective="regression",
            device_type="cpu",
            n_estimators=1,
            min_child_samples=1,
            random_state=config.SEED,
            n_jobs=1,
            verbosity=-1,
        )
        probe.fit(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0.0, 1.0, 0.0, 1.0]))
    except Exception as error:  # noqa: BLE001 - preserve the underlying error
        raise RuntimeError(
            "Stage 2 requires a working CPU LightGBM installation. The installed build could not "
            "fit the preflight model."
        ) from error


def make_cpu_lgbm(params: dict[str, Any], prefer_gpu: bool = True):
    """Build a LightGBM regressor that always uses CPU.

    ``prefer_gpu`` is accepted because Stage 1's shared fitting helper passes
    it to factories; it is ignored because Stage 2 is explicitly CPU-only.
    """
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        objective="regression",
        device_type="cpu",
        random_state=config.SEED,
        n_jobs=-1,
        subsample_freq=1,  # enables row sampling when subsample < 1
        verbosity=-1,
        **params,
    )


def make_spec(experiment: int) -> models.RunSpec:
    setting = EXPERIMENTS[experiment]
    scope = setting["scope"]
    return models.RunSpec(
        run_id=f"S2_{experiment}__{setting['label']}__LGBM_CPU__SEED{config.SEED}",
        algorithm="lightgbm",
        training_strategy=scope,
        forecast_strategy=setting["forecast"],
        spatial_mode="none",
        prefer_gpu=False,
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


def _multi_output_partition(
    spec: models.RunSpec,
    params: dict[str, Any],
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: list[str],
    group_name: str,
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[dict[str, Any]]]:
    """Fit one multi-output forecaster and return its seven-horizon predictions.

    LightGBM's sklearn regressor accepts one target at a time.  The
    ``MultiOutputRegressor`` wrapper gives Stage 2 one forecasting object with
    a seven-column target/prediction interface.  Its fitted LightGBM estimators
    are deliberately kept inside that single wrapper rather than being fitted
    in a horizon loop by this experiment.
    """
    target_columns = [f"target_t{h}" for h in config.FORECAST_HORIZONS]
    id_columns = [config.STATION_ID_COL, config.DATE_COL]
    max_horizon = max(config.FORECAST_HORIZONS)
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

    train_rows = train_frame.dropna(subset=target_columns)
    train_rows = train_rows[
        train_rows[config.DATE_COL] + pd.Timedelta(days=max_horizon) < train_label_end
    ]
    validation_rows = validation_frame.dropna(subset=target_columns)
    validation_rows = validation_rows[
        validation_rows[config.DATE_COL] + pd.Timedelta(days=max_horizon) < boundary
    ]
    if train_rows.empty or validation_rows.empty:
        return (
            pd.DataFrame(columns=id_columns + ["horizon", "y_true", "y_pred"]),
            [],
            [],
        )

    forecaster = MultiOutputRegressor(make_cpu_lgbm(dict(params)), n_jobs=1)
    forecaster.fit(train_rows[feature_columns], train_rows[target_columns])
    prediction_matrix = np.asarray(forecaster.predict(validation_rows[feature_columns]))
    if prediction_matrix.shape != (len(validation_rows), len(target_columns)):
        raise RuntimeError(
            "Multi-output forecaster returned an unexpected prediction shape: "
            f"{prediction_matrix.shape}; expected "
            f"({len(validation_rows)}, {len(target_columns)})"
        )

    prediction_blocks: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    for output_index, horizon in enumerate(config.FORECAST_HORIZONS):
        block = validation_rows[id_columns].copy()
        block["horizon"] = horizon
        block["y_true"] = validation_rows[target_columns[output_index]].to_numpy()
        block["y_pred"] = prediction_matrix[:, output_index]
        prediction_blocks.append(block)

        estimator = forecaster.estimators_[output_index]
        importance = models._importance_frame(estimator, feature_columns, f"output_t{horizon}")
        if not importance.empty:
            importance["group"] = str(group_name)
            importances.append(importance)

    record = {
        "estimator": forecaster,
        "group": str(group_name),
        "horizon": 0,
        "role": "multi_output",
        "feature_columns": feature_columns,
        "target_columns": target_columns,
    }
    return pd.concat(prediction_blocks, ignore_index=True), importances, [record]


def run_pooled(spec: models.RunSpec, df: pd.DataFrame, stations: list,
               params: dict[str, Any], evaluation: str) -> dict[str, Any]:
    """Train pooled LightGBM models for one split using a supplied trial config."""
    frame = df[df[config.STATION_ID_COL].isin(stations)].copy()
    masks = _evaluation_masks(frame, evaluation)
    eval_spec = replace(
        spec,
        evaluation_start=(config.TEST_START if evaluation == "test" else config.VALIDATION_START),
        evaluation_end=(None if evaluation == "test" else config.TEST_START),
        selection_patience=None,
    )
    base_columns = models.assemble_feature_columns(eval_spec)
    frame, feature_columns = models._add_categorical_codes(frame, eval_spec, base_columns, masks["train"])
    groups = [("global", frame)] if eval_spec.training_strategy == "global" else list(frame.groupby("region_id"))
    factory = partial(make_cpu_lgbm, dict(params))
    predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []

    for group_name, group_frame in groups:
        train_frame = group_frame.loc[masks["train"].reindex(group_frame.index)]
        validation_frame = group_frame.loc[masks["validation"].reindex(group_frame.index)]
        if eval_spec.forecast_strategy == "multi":
            group_predictions, group_importances, group_records = _multi_output_partition(
                eval_spec, params, train_frame, validation_frame,
                feature_columns, str(group_name),
            )
            records.extend(group_records)
        else:
            group_predictions, group_importances = models._fit_partition(
                eval_spec, factory, train_frame, validation_frame, feature_columns,
                records, str(group_name),
            )
        predictions.append(group_predictions)
        for importance in group_importances:
            if not importance.empty:
                importance = importance.copy()
                importance["group"] = str(group_name)
                importances.append(importance)

    combined = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return models._finalize(eval_spec, combined, importances, feature_columns, records)


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
        folds_manifest=pd.DataFrame(),
        reports=result["reports"],
        predictions=result["predictions"],
        feature_list=result["feature_list"],
        feature_importance=result["importance"],
        training_log=run_config["description"],
        split=split,
        model_records=result["model_records"],
        prefer_gpu=False,
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    require_lightgbm()

    spec = make_spec(args.experiment)
    if not args.train:
        print("[dry run] CPU LightGBM probe passed. Re-run with --train to load data and tune.")
        return 0

    df = runner.prepare_dataframe(spec, source=args.source)
    stations = data.resolve_stations(df, args.stations)
    if args.max_stations is not None:
        stations = stations[:args.max_stations]
    if not stations:
        raise ValueError("No eligible stations remain after selection")

    run_dir = artifact_directory(spec.run_id)
    candidates = list(ParameterSampler(HYPERPARAMETER_GRID, n_iter=args.n_trials, random_state=args.random_seed))
    LOGGER.info("Stage 2 experiment %d: %s; stations=%d; trials=%d; output=%s",
                args.experiment, EXPERIMENTS[args.experiment]["label"], len(stations), len(candidates), run_dir)
    trials: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_score = float("inf")

    for trial_number, params in enumerate(candidates, start=1):
        started = perf_counter()
        row: dict[str, Any] = {"trial": trial_number, **params}
        try:
            result = run_pooled(spec, df, stations, params, evaluation="validation")
            score = primary_score(result)
            row.update(status="ok", primary_macro_rmse=score, elapsed_seconds=perf_counter() - started)
            if score < best_score:
                best_score, best_params = score, dict(params)
                row["is_best"] = True
                LOGGER.info("Trial %d/%d is best: RMSE %.6f", trial_number, len(candidates), score)
            else:
                row["is_best"] = False
                LOGGER.info("Trial %d/%d: RMSE %.6f", trial_number, len(candidates), score)
        except Exception as error:  # noqa: BLE001 - retain failures for reproducibility
            row.update(status="failed", error=str(error), elapsed_seconds=perf_counter() - started, is_best=False)
            LOGGER.exception("Trial %d/%d failed", trial_number, len(candidates))
        trials.append(row)
        pd.DataFrame(trials).to_csv(run_dir / "tuning_trials.csv", index=False)

    if best_params is None:
        raise RuntimeError("No hyperparameter trial completed successfully; see tuning_trials.csv")

    # Refit on the same validation split with the selected configuration to
    # write a complete validation artifact.  Test is not involved in tuning.
    validation_result = run_pooled(spec, df, stations, best_params, evaluation="validation")
    validation_rows = df[df[config.STATION_ID_COL].isin(stations) & splits.date_masks(df)["validation"]]
    validation_pair = write_persistence_comparison(
        run_dir, "validation", validation_result["predictions"], validation_rows
    )
    output_mode = "multi-output" if spec.forecast_strategy == "multi" else "single-output/direct"
    description = (
        f"Stage 2 experiment {args.experiment}: {EXPERIMENTS[args.experiment]['scope']} LightGBM "
        f"with {output_mode} forecasting; CPU training. "
        "Parameters were selected by validation primary macro RMSE from the supplied grid."
    )
    run_config = {
        "description": description,
        "experiment_id": args.experiment,
        "scope": EXPERIMENTS[args.experiment]["scope"],
        "algorithm": "lightgbm",
        "training_strategy": spec.training_strategy,
        "forecast_strategy": "multi_output" if spec.forecast_strategy == "multi" else "direct",
        "prediction_output": "multi_output" if spec.forecast_strategy == "multi" else "single_output",
        "multi_output_implementation": (
            "sklearn.multioutput.MultiOutputRegressor(LGBMRegressor): one forecaster "
            "object accepts seven targets and returns seven predictions."
            if spec.forecast_strategy == "multi" else None
        ),
        "spatial_mode": "none",
        "include_station_id": True,
        "include_region_id": spec.include_region_id,
        "requested_device": "cpu",
        "actual_device_requirement": "cpu",
        "selection_split": "validation",
        "test_locked_during_tuning": True,
        "hyperparameter_grid": HYPERPARAMETER_GRID,
        "n_trials_requested": args.n_trials,
        "n_trials_completed": sum(t["status"] == "ok" for t in trials),
        "random_seed": args.random_seed,
        "best_params": best_params,
        "best_validation_primary_macro_rmse": best_score,
        "paired_validation_n": validation_pair["n"],
        "paired_validation_model_primary_macro_rmse": validation_pair["model"]["overall"]["macro"].get("primary_macro_rmse"),
        "naive_validation_primary_macro_rmse": validation_pair["persistence"]["overall"]["macro"].get("primary_macro_rmse"),
        "evaluation_splits": ["validation"] if args.skip_test else ["validation", "test"],
        "feature_count": len(validation_result["feature_list"]),
        "spatial_config": None,
    }
    save_split(run_dir, spec, df, stations, validation_result, run_config, "validation")
    artifacts.write_json(run_dir / "best_params.json", best_params)

    if args.skip_test:
        LOGGER.info("Validation winner saved. Test was skipped by request.")
        return 0

    test_started = perf_counter()
    test_result = run_pooled(spec, df, stations, best_params, evaluation="test")
    test_rows = df[df[config.STATION_ID_COL].isin(stations) & splits.date_masks(df)["test"]]
    test_pair = write_persistence_comparison(run_dir, "test", test_result["predictions"], test_rows)
    run_config.update({
        "test_seconds": perf_counter() - test_started,
        "paired_test_n": test_pair["n"],
        "paired_test_model_primary_macro_rmse": test_pair["model"]["overall"]["macro"].get("primary_macro_rmse"),
        "naive_test_primary_macro_rmse": test_pair["persistence"]["overall"]["macro"].get("primary_macro_rmse"),
    })
    save_split(run_dir, spec, df, stations, test_result, run_config, "test")
    LOGGER.info("Complete. Validation RMSE=%.6f; test RMSE=%s",
                best_score, test_pair["model"]["overall"]["macro"].get("primary_macro_rmse"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
