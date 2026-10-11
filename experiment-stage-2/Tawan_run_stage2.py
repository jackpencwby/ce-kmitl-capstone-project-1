#!/usr/bin/env python3
"""Stage 2 XGBoost experiments with hierarchical validation-only tuning."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import re
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_DIR = REPO_ROOT / "experiment-stage-1"
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

from common import artifacts, config, data, features, metrics, models, runner, splits  # noqa: E402

LOGGER = logging.getLogger("stage2")
TARGET_COLS = [f"target_t{h}" for h in config.FORECAST_HORIZONS]
TARGET_LABELS = [f"t+{h}" for h in config.FORECAST_HORIZONS]

ORIGINAL_DOMAIN: dict[str, list[Any]] = {
    "n_estimators": [300, 500, 800, 1200, 2000],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "min_child_weight": [1, 2, 5, 10, 20],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.01, 0.1, 0.5, 1],
    "reg_alpha": [0, 0.001, 0.01, 0.1, 1, 5],
    "reg_lambda": [0.1, 0.5, 1, 2, 5, 10],
}
PHASE1_GRID = {
    "learning_rate": [0.03, 0.1],
    "max_depth": [4, 6, 8],
    "min_child_weight": [1, 5, 20],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}
PHASE2_GRID = {
    "gamma": ORIGINAL_DOMAIN["gamma"],
    "reg_alpha": ORIGINAL_DOMAIN["reg_alpha"],
    "reg_lambda": ORIGINAL_DOMAIN["reg_lambda"],
}
FIXED_REGULARIZATION = {"gamma": 0, "reg_alpha": 0, "reg_lambda": 1}
MAX_BOOSTING_ROUNDS = 3000
PHASE1_BUDGETS = (600, 1400, MAX_BOOSTING_ROUNDS)
PHASE2_BUDGETS = (800, MAX_BOOSTING_ROUNDS)
PHASE3_BUDGETS = (800, MAX_BOOSTING_ROUNDS)
PHASE2_KEEP = 36
PHASE3_KEEP = 36

EXPERIMENTS: dict[int, dict[str, str]] = {
    1: {"scope": "global", "forecast": "direct", "label": "GLOBAL_SINGLE_HEAD"},
    2: {"scope": "global", "forecast": "multi", "label": "GLOBAL_MULTI_HEAD"},
    7: {"scope": "regional", "forecast": "direct", "label": "REGIONAL_SINGLE_HEAD"},
    8: {"scope": "regional", "forecast": "multi", "label": "REGIONAL_MULTI_HEAD"},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=int, choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--train", action="store_true", help="Run hierarchical tuning and save artifacts.")
    parser.add_argument("--source", choices=("auto", "local", "gcs"), default="auto")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--stations", nargs="+", default=["all"])
    parser.add_argument("--max-stations", type=int)
    parser.add_argument("--skip-test", action="store_true", help="Stop after validation selection/refit.")
    parser.add_argument("--tuning-mode", choices=("hierarchical-grid",), default="hierarchical-grid")
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--phase1-pruning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-phase1-candidates", type=int)
    parser.add_argument("--max-phase2-candidates", type=int)
    parser.add_argument("--max-phase3-candidates", type=int)
    parser.add_argument("--development", action="store_true", help="Use a small deterministic candidate subset.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.early_stopping_rounds < 1:
        parser.error("--early-stopping-rounds must be positive")
    for name in ("max_stations", "max_phase1_candidates", "max_phase2_candidates", "max_phase3_candidates"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.development:
        args.max_phase1_candidates = min(args.max_phase1_candidates or 3, 3)
        args.max_phase2_candidates = min(args.max_phase2_candidates or 3, 3)
        args.max_phase3_candidates = min(args.max_phase3_candidates or 3, 3)
        args.skip_test = True
    return args


def _gpu_diagnostics() -> dict[str, Any]:
    info: dict[str, Any] = {"device": "cuda:0"}
    try:
        import torch
        info.update({
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_bytes"] = int(props.total_memory)
            info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:  # diagnostic only; XGBoost probe remains authoritative
        info["torch_diagnostic_error"] = f"{type(exc).__name__}: {exc}"
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        info["nvidia_smi"] = completed.stdout.strip() if completed.returncode == 0 else completed.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        info["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _synchronize_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def cuda_preflight(require_multi: bool) -> dict[str, Any]:
    """Fail before dataset loading unless scalar CUDA and requested vector trees fit."""
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("Stage 2 requires XGBoost. Install experiment-stage-2/Tawan_requirements.txt.") from exc
    info = {"xgboost_version": xgb.__version__, "requested_device": "cuda:0",
            "tree_method": "hist", "diagnostics": _gpu_diagnostics()}
    if not xgb.build_info().get("USE_CUDA", False):
        raise RuntimeError(
            f"Installed XGBoost {xgb.__version__} was built without CUDA; device=cuda:0; "
            f"build_info={xgb.build_info()}; CUDA diagnostics={info['diagnostics']}"
        )

    X = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1], [1, 2], [2, 2], [3, 1]], dtype=np.float32)
    y = np.asarray([0, 1, 1, 2, 3, 3, 4, 4], dtype=np.float32)
    Y = np.column_stack([y + h for h in range(7)]).astype(np.float32)
    checks: list[str] = []
    for label, extra, target in [("single_output", {}, y)]:
        model = xgb.XGBRegressor(
            n_estimators=8, max_depth=2, learning_rate=0.1, tree_method="hist",
            device="cuda:0", objective="reg:squarederror", eval_metric="rmse",
            early_stopping_rounds=2, random_state=config.SEED, **extra,
        )
        try:
            model.fit(X, target, eval_set=[(X, target)], verbose=False)
        except Exception as exc:
            raise RuntimeError(
                f"CUDA preflight failed for {label}; XGBoost={xgb.__version__}; "
                f"device=cuda:0; CUDA diagnostics={info['diagnostics']}; "
                f"exception={type(exc).__name__}: {exc}"
            ) from exc
        prediction = np.asarray(model.predict(X))
        if prediction.shape != target.shape:
            raise RuntimeError(f"CUDA {label} probe returned shape {prediction.shape}, expected {target.shape}")
        if not hasattr(model, "best_iteration"):
            raise RuntimeError(f"CUDA {label} probe did not expose best_iteration; early stopping is unavailable")
        checks.append(label)
        del model
        _synchronize_cuda()
    if require_multi:
        model = xgb.XGBRegressor(
            n_estimators=8, max_depth=2, learning_rate=0.1, tree_method="hist",
            device="cuda:0", objective="reg:squarederror", eval_metric="rmse",
            multi_strategy="multi_output_tree", early_stopping_rounds=2,
            random_state=config.SEED,
        )
        try:
            model.fit(X, Y, eval_set=[(X, Y)], verbose=False)
        except Exception as exc:
            raise RuntimeError(
                f"CUDA multi_output_tree preflight failed; XGBoost={xgb.__version__}; "
                f"device=cuda:0; CUDA diagnostics={info['diagnostics']}; "
                f"exception={type(exc).__name__}: {exc}"
            ) from exc
        prediction = np.asarray(model.predict(X))
        if prediction.shape != Y.shape:
            raise RuntimeError(f"multi_output_tree returned {prediction.shape}; expected {Y.shape}")
        if model.get_params().get("multi_strategy") != "multi_output_tree":
            raise RuntimeError("XGBoost did not retain multi_strategy='multi_output_tree'")
        if not hasattr(model, "best_iteration"):
            raise RuntimeError("Native multi_output_tree early stopping is unsupported by this XGBoost build")
        checks.append("multi_output_tree + early_stopping")
        info["multi_output_tree"] = "PASS"
        info["multi_prediction_shape"] = list(prediction.shape)
        info["multi_feature_importance"] = probe_multi_importance(model)
        # Confirm JSON serialization preserves one native vector-leaf model and output shape.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="stage2-xgb-probe-") as temp_dir:
            model_path = Path(temp_dir) / "probe.json"
            model.save_model(model_path)
            loaded = xgb.XGBRegressor()
            loaded.load_model(model_path)
            if np.asarray(loaded.predict(X)).shape != Y.shape:
                raise RuntimeError("Reloaded multi_output_tree model changed prediction shape")
        del model
        _synchronize_cuda()
    info.update({"cuda_training": "PASS", "early_stopping": "PASS", "checks": checks})
    return info


def probe_multi_importance(model: Any) -> dict[str, Any]:
    try:
        values = model.feature_importances_
        return {"status": "supported", "shape": list(np.asarray(values).shape)}
    except Exception as exc:  # vector-leaf importance is version-dependent
        return {"status": "unsupported", "reason": f"{type(exc).__name__}: {exc}"}


def make_spec(experiment: int, seed: int) -> models.RunSpec:
    setting = EXPERIMENTS[experiment]
    regional = setting["scope"] == "regional"
    return models.RunSpec(
        run_id=f"S2_{experiment}__{setting['label']}__XGB_CUDA__SEED{seed}",
        algorithm="xgboost", training_strategy=setting["scope"],
        forecast_strategy=setting["forecast"], spatial_mode="none", prefer_gpu=True,
        include_station_id=True, include_region_id=regional,
        complete_target_evaluation=True, selection_patience=None,
    )


def _all_candidates(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    names = list(grid)
    return [dict(zip(names, values)) for values in itertools.product(*(grid[name] for name in names))]


def _cap(candidates: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    return candidates if limit is None else candidates[:limit]


def phase1_candidates() -> list[dict[str, Any]]:
    fixed = FIXED_REGULARIZATION
    return [dict(candidate, **fixed) for candidate in _all_candidates(PHASE1_GRID)]


def phase2_candidates(best_structure: dict[str, Any]) -> list[dict[str, Any]]:
    anchor = {key: best_structure[key] for key in PHASE1_GRID}
    return [dict(anchor, **candidate) for candidate in _all_candidates(PHASE2_GRID)]


def _neighbor_values(domain: list[Any], center: Any, count: int = 3) -> list[Any]:
    index = domain.index(center)
    start = max(0, min(index - count // 2, len(domain) - count))
    stop = min(len(domain), start + count)
    return domain[start:stop]


def phase3_candidates(best_params: dict[str, Any]) -> list[dict[str, Any]]:
    keys = list(PHASE1_GRID)
    local = {key: _neighbor_values(ORIGINAL_DOMAIN[key], best_params[key]) for key in keys}
    result = []
    for structural in _all_candidates(local):
        result.append({**best_params, **structural})
    return result


def _candidate_key(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _keep_best(scores: list[tuple[float, str, dict[str, Any]]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(scores, key=lambda item: (item[0], item[1]))
    return [item[2] for item in ordered[:max(1, min(count, len(ordered)))]]


def _valid_rows(frame: pd.DataFrame, dates: pd.Series, boundary: pd.Timestamp) -> pd.DataFrame:
    candidates = frame.loc[dates.reindex(frame.index)].dropna(subset=TARGET_COLS)
    horizon_delta = pd.Timedelta(int(max(config.FORECAST_HORIZONS)), unit="D")
    return candidates[candidates[config.DATE_COL] + horizon_delta < boundary]


def make_context(df: pd.DataFrame, stations: list[Any], spec: models.RunSpec,
                 evaluation: str) -> dict[str, Any]:
    frame = df[df[config.STATION_ID_COL].isin(stations)].copy()
    if evaluation == "validation":
        # Keep the locked test rows out of the tuning context entirely.
        frame = frame[frame[config.DATE_COL] < pd.Timestamp(config.TEST_START)].copy()
    masks = splits.date_masks(frame)
    if evaluation == "validation":
        fit_mask, score_mask = masks["train"], masks["validation"]
        fit_boundary, score_boundary = pd.Timestamp(config.VALIDATION_START), pd.Timestamp(config.TEST_START)
    elif evaluation == "test":
        fit_mask = masks["train"] | masks["validation"]
        score_mask = masks["test"]
        fit_boundary, score_boundary = pd.Timestamp(config.TEST_START), pd.Timestamp.max.normalize()
    else:
        raise ValueError(f"Unknown evaluation split {evaluation!r}")
    feature_columns = models.assemble_feature_columns(spec)
    frame, feature_columns = models._add_categorical_codes(frame, spec, feature_columns, fit_mask)
    train_rows = _valid_rows(frame, fit_mask, fit_boundary)
    validation_rows = _valid_rows(frame, score_mask, score_boundary)
    if train_rows.empty or validation_rows.empty:
        raise ValueError(f"No complete seven-target rows for {evaluation}: train={len(train_rows)}, eval={len(validation_rows)}")
    if spec.training_strategy == "regional":
        if "region_id" not in frame or frame["region_id"].isna().any():
            raise ValueError("Stage 1 did not assign a region_id to every eligible row")
        groups = sorted(frame["region_id"].dropna().unique().tolist(), key=str)
    else:
        groups = ["global"]
    return {"frame": frame, "train_rows": train_rows, "validation_rows": validation_rows,
            "features": feature_columns, "groups": groups, "evaluation": evaluation,
            "fit_boundary": fit_boundary, "score_boundary": score_boundary, "spec": spec}


def make_final_train_context(df: pd.DataFrame, stations: list[Any],
                             spec: models.RunSpec) -> dict[str, Any]:
    """Build only Train + Validation inputs for final refit; do not select test rows."""
    frame = df[df[config.STATION_ID_COL].isin(stations)].copy()
    masks = splits.date_masks(frame)
    fit_mask = masks["train"] | masks["validation"]
    feature_columns = models.assemble_feature_columns(spec)
    frame, feature_columns = models._add_categorical_codes(frame, spec, feature_columns, fit_mask)
    train_rows = _valid_rows(frame, fit_mask, pd.Timestamp(config.TEST_START))
    if train_rows.empty:
        raise ValueError("No complete seven-target rows remain for final Train + Validation refit")
    if spec.training_strategy == "regional":
        if "region_id" not in frame or frame["region_id"].isna().any():
            raise ValueError("Stage 1 did not assign a region_id to every eligible row")
        groups = sorted(frame["region_id"].dropna().unique().tolist(), key=str)
    else:
        groups = ["global"]
    return {"frame": frame, "train_rows": train_rows, "features": feature_columns,
            "groups": groups, "spec": spec}


def _fit_groups(train_rows: pd.DataFrame, score_rows: pd.DataFrame | None,
                spec: models.RunSpec, groups: list[Any]):
    if spec.training_strategy == "global":
        if score_rows is None:
            yield "global", train_rows, None
        else:
            yield "global", train_rows, score_rows
        return
    for group in groups:
        tr = train_rows[train_rows["region_id"] == group]
        va = score_rows[score_rows["region_id"] == group] if score_rows is not None else None
        if tr.empty or (va is not None and va.empty):
            raise ValueError(f"Region {group!r} has no complete train or evaluation rows")
        yield str(group), tr, va


def _model_groups(context: dict[str, Any]):
    return list(_fit_groups(context["train_rows"], context["validation_rows"],
                            context["spec"], context["groups"]))


def _xgb(params: dict[str, Any], seed: int, budget: int, early_stopping_rounds: int | None,
         multi: bool):
    import xgboost as xgb
    kwargs = {
        "objective": "reg:squarederror", "tree_method": "hist", "device": "cuda:0",
        "n_estimators": int(budget), "random_state": int(seed), "n_jobs": -1,
        "eval_metric": "rmse", "verbosity": 0, **params,
    }
    if early_stopping_rounds is not None:
        # XGBoost 3.x takes early_stopping_rounds in the estimator constructor.
        kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
    if multi:
        kwargs["multi_strategy"] = "multi_output_tree"
    return xgb.XGBRegressor(**kwargs)


def _prediction_blocks(rows: pd.DataFrame, matrix: np.ndarray) -> list[pd.DataFrame]:
    matrix = np.asarray(matrix)
    expected = (len(rows), len(TARGET_COLS))
    if matrix.shape != expected:
        raise RuntimeError(f"Model prediction shape {matrix.shape}; expected {expected}")
    blocks = []
    for j, horizon in enumerate(config.FORECAST_HORIZONS):
        block = rows[[config.STATION_ID_COL, config.DATE_COL]].copy()
        block["horizon"] = horizon
        block["y_true"] = rows[TARGET_COLS[j]].to_numpy()
        block["y_pred"] = matrix[:, j]
        blocks.append(block)
    return blocks


def fit_evaluate(context: dict[str, Any], params: dict[str, Any], budget: int,
                 early_stopping_rounds: int | None, seed: int,
                 fit_counter: dict[str, int], keep_models: bool = False) -> dict[str, Any]:
    importances: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    prediction_blocks: list[pd.DataFrame] = []
    fit_best_iterations: dict[str, int] = {}
    fit_best_scores: dict[str, float | None] = {}
    multi = context["spec"].forecast_strategy == "multi"
    xcols = context["features"]

    for group, train_rows, validation_rows in _model_groups(context):
        if multi:
            estimator = _xgb(params, seed, budget, early_stopping_rounds, multi=True)
            fit_counter["total"] += 1
            estimator.fit(
                train_rows[xcols], train_rows[TARGET_COLS],
                eval_set=[(validation_rows[xcols], validation_rows[TARGET_COLS])], verbose=False,
            )
            prediction = estimator.predict(validation_rows[xcols])
            prediction_blocks.extend(_prediction_blocks(validation_rows, prediction))
            key = f"{group}:multi_output"
            fit_best_iterations[key] = int(getattr(estimator, "best_iteration", budget - 1))
            try: fit_best_scores[key] = float(estimator.best_score)
            except (AttributeError, TypeError, ValueError): fit_best_scores[key] = None
            try:
                imp = models._importance_frame(estimator, xcols, "multi_output")
                if not imp.empty:
                    imp["group"] = group
                    importances.append(imp)
            except Exception as exc:
                if keep_models:
                    importances.append(pd.DataFrame([{"status": "unsupported", "reason": f"{type(exc).__name__}: {exc}"}]))
            if keep_models:
                records.append({"estimator": estimator, "group": group, "horizon": 0,
                                "role": "multi_output", "feature_columns": xcols,
                                "target_columns": TARGET_COLS})
            else:
                del estimator
            continue

        for j, horizon in enumerate(config.FORECAST_HORIZONS):
            estimator = _xgb(params, seed, budget, early_stopping_rounds, multi=False)
            fit_counter["total"] += 1
            estimator.fit(
                train_rows[xcols], train_rows[TARGET_COLS[j]],
                eval_set=[(validation_rows[xcols], validation_rows[TARGET_COLS[j]])], verbose=False,
            )
            yhat = np.asarray(estimator.predict(validation_rows[xcols])).reshape(-1)
            block = validation_rows[[config.STATION_ID_COL, config.DATE_COL]].copy()
            block["horizon"] = horizon
            block["y_true"] = validation_rows[TARGET_COLS[j]].to_numpy()
            block["y_pred"] = yhat
            prediction_blocks.append(block)
            key = f"{group}:t+{horizon}"
            fit_best_iterations[key] = int(getattr(estimator, "best_iteration", budget - 1))
            try: fit_best_scores[key] = float(estimator.best_score)
            except (AttributeError, TypeError, ValueError): fit_best_scores[key] = None
            imp = models._importance_frame(estimator, xcols, f"t+{horizon}")
            if not imp.empty:
                imp["group"] = group
                importances.append(imp)
            if keep_models:
                records.append({"estimator": estimator, "group": group, "horizon": horizon,
                                "role": "direct", "feature_columns": xcols})
            else:
                del estimator

    prediction_frame = pd.concat(prediction_blocks, ignore_index=True)
    reports = metrics.build_reports(prediction_frame)
    score = reports["overall"]["macro"].get("primary_macro_rmse")
    if score is None or not np.isfinite(score):
        raise ValueError("No finite validation primary_macro_rmse was produced")
    per_horizon = reports["by_horizon"].set_index("horizon")["rmse"].to_dict()
    return {
        "score": float(score), "predictions": prediction_frame, "reports": reports,
        "importance": pd.concat(importances, ignore_index=True) if importances else pd.DataFrame(),
        "model_records": records, "best_iterations": fit_best_iterations,
        "best_scores": fit_best_scores,
        "rmse_by_horizon": {f"rmse_t{h}": float(per_horizon.get(h, float("nan"))) for h in config.FORECAST_HORIZONS},
    }


def _run_round(phase: int, round_id: int, candidates: list[dict[str, Any]], budget: int,
               context: dict[str, Any], args: argparse.Namespace, run_dir: Path,
               trial_rows: list[dict[str, Any]], fit_counter: dict[str, int], phase_seconds: dict[str, float]) -> list[dict[str, Any]]:
    results: list[tuple[float, str, dict[str, Any]]] = []
    phase_key = f"phase{phase}"
    started_phase = perf_counter()
    for index, params in enumerate(candidates, start=1):
        started = perf_counter()
        trial_id = f"P{phase}R{round_id}T{index:04d}"
        row: dict[str, Any] = {
            "phase": phase, "round": round_id, "trial_id": trial_id,
            "experiment_id": args.experiment, "parameters": _candidate_key(params),
            "boosting_budget": budget, "best_iteration": None, "best_score": None,
            "validation_primary_macro_rmse": None, **{f"rmse_t{h}": None for h in config.FORECAST_HORIZONS},
            "elapsed_seconds": None, "status": "running", "failure_reason": "",
        }
        try:
            result = fit_evaluate(
                context, params, budget, args.early_stopping_rounds, args.seed,
                fit_counter, keep_models=False,
            )
            row.update({
                "best_iteration": json.dumps(result["best_iterations"], sort_keys=True),
                "best_score": json.dumps(result["best_scores"], sort_keys=True),
                "validation_primary_macro_rmse": result["score"],
                **result["rmse_by_horizon"], "status": "ok",
            })
            results.append((result["score"], _candidate_key(params), params))
        except Exception as exc:  # keep candidate-level failures auditable
            row.update(status="failed", failure_reason=f"{type(exc).__name__}: {exc}")
            LOGGER.exception("%s candidate %s failed", trial_id, _candidate_key(params))
        finally:
            _synchronize_cuda()
            row["elapsed_seconds"] = perf_counter() - started
            trial_rows.append(row)
            pd.DataFrame(trial_rows).to_csv(run_dir / "tuning_trials.csv", index=False)
    phase_seconds[phase_key] = phase_seconds.get(phase_key, 0.0) + perf_counter() - started_phase
    results.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in results]


def _limit(candidates: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    limited = _cap(candidates, limit)
    if not limited:
        raise ValueError("Candidate grid is empty")
    return limited


def hierarchical_search(context: dict[str, Any], args: argparse.Namespace,
                        run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, float], dict[str, int]]:
    if context.get("evaluation") != "validation":
        raise ValueError("Hyperparameter search accepts validation context only; test context is locked")
    tuning_started = perf_counter()
    trial_rows: list[dict[str, Any]] = []
    fit_counter = {"total": 0}
    phase_seconds: dict[str, float] = {}
    p1 = _limit(phase1_candidates(), args.max_phase1_candidates)
    p1_survivors = p1
    p1_stage_counts = []
    for round_id, budget in enumerate(PHASE1_BUDGETS if args.phase1_pruning else (MAX_BOOSTING_ROUNDS,), start=1):
        scored = _run_round(1, round_id, p1_survivors, budget, context, args, run_dir,
                            trial_rows, fit_counter, phase_seconds)
        keep = max(1, math.ceil(len(scored) / 3)) if args.phase1_pruning else len(scored)
        p1_survivors = scored[:keep]
        p1_stage_counts.append({"round": round_id, "boosting_budget": budget,
                                "candidates_evaluated": len(scored), "survivors": len(p1_survivors)})
        if not p1_survivors:
            raise RuntimeError("All Phase 1 candidates failed")
    phase1_winner = p1_survivors[0]
    pd.DataFrame([r for r in trial_rows if r["phase"] == 1]).to_csv(run_dir / "phase1_results.csv", index=False)

    p2 = _limit(phase2_candidates(phase1_winner), args.max_phase2_candidates)
    p2_scored = _run_round(2, 1, p2, PHASE2_BUDGETS[0], context, args, run_dir,
                           trial_rows, fit_counter, phase_seconds)
    p2_survivors = p2_scored[:min(PHASE2_KEEP, len(p2_scored))]
    p2_final = _run_round(2, 2, p2_survivors, PHASE2_BUDGETS[1], context, args, run_dir,
                          trial_rows, fit_counter, phase_seconds)
    if not p2_final:
        raise RuntimeError("All Phase 2 candidates failed")
    phase2_winner = p2_final[0]
    pd.DataFrame([r for r in trial_rows if r["phase"] == 2]).to_csv(run_dir / "phase2_results.csv", index=False)

    p3 = _limit(phase3_candidates(phase2_winner), args.max_phase3_candidates)
    p3_scored = _run_round(3, 1, p3, PHASE3_BUDGETS[0], context, args, run_dir,
                           trial_rows, fit_counter, phase_seconds)
    p3_survivors = p3_scored[:min(PHASE3_KEEP, len(p3_scored))]
    p3_final = _run_round(3, 2, p3_survivors, PHASE3_BUDGETS[1], context, args, run_dir,
                          trial_rows, fit_counter, phase_seconds)
    if not p3_final:
        raise RuntimeError("All Phase 3 candidates failed")
    best_params = p3_final[0]
    best_trial = next(r for r in reversed(trial_rows) if r["phase"] == 3 and r["round"] == 2
                      and r["parameters"] == _candidate_key(best_params) and r["status"] == "ok")
    pd.DataFrame([r for r in trial_rows if r["phase"] == 3]).to_csv(run_dir / "phase3_results.csv", index=False)
    summary = {
        "phase1_candidate_count": len(phase1_candidates()),
        "phase1_candidates_used": len(p1), "phase1_survivors": len(p1_survivors),
        "phase1_rounds": p1_stage_counts,
        "phase2_candidate_count": len(phase2_candidates(phase1_winner)),
        "phase2_candidates_used": len(p2), "phase2_survivors": len(p2_survivors),
        "phase3_candidate_count": len(phase3_candidates(phase2_winner)),
        "phase3_candidates_used": len(p3), "phase3_survivors": len(p3_survivors),
        "total_candidate_evaluations": sum(r["status"] in ("ok", "failed") for r in trial_rows),
        "total_models_trained": fit_counter["total"],
        "best_validation_score": float(best_trial["validation_primary_macro_rmse"]),
        "best_parameters": best_params,
        "best_iteration": json.loads(best_trial["best_iteration"]),
        "phase_seconds": phase_seconds,
        "total_tuning_seconds": perf_counter() - tuning_started,
    }
    artifacts.write_json(run_dir / "tuning_summary.json", summary)
    return best_params, summary, trial_rows, phase_seconds, fit_counter


def _best_rounds(validation_models: list[dict[str, Any]]) -> dict[str, int]:
    rounds = {}
    for record in validation_models:
        group = str(record["group"])
        if record["role"] == "multi_output":
            estimator = record["estimator"]
            rounds[f"{group}:multi_output"] = int(getattr(estimator, "best_iteration", MAX_BOOSTING_ROUNDS - 1)) + 1
        else:
            estimator = record["estimator"]
            rounds[f"{group}:t+{record['horizon']}"] = int(getattr(estimator, "best_iteration", MAX_BOOSTING_ROUNDS - 1)) + 1
    return rounds


def _final_refit_params(best_params: dict[str, Any], rounds: dict[str, int], group: str,
                        horizon: int | None, multi: bool) -> dict[str, Any]:
    # The estimator budget is passed separately so tuned parameters can never
    # accidentally override the validation-selected iteration count.
    return dict(best_params)


def _artifact_dir(run_id: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = Path(__file__).resolve().parent / "artifacts" / f"{run_id}__{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _save_named_final_models(run_dir: Path, records: list[dict[str, Any]],
                             training_context: dict[str, Any]) -> list[dict[str, Any]]:
    import xgboost as xgb
    model_dir = run_dir / "models"
    model_manifest = []
    for record in records:
        group = str(record["group"])
        safe_group = re.sub(r"[^A-Za-z0-9_.-]", "_", group)
        if record["role"] == "multi_output":
            name = "multi_output.json" if group == "global" else f"{safe_group}_multi_output.json"
            path = model_dir / name
        elif group == "global":
            path = model_dir / f"horizon_{record['horizon']}.json"
        else:
            path = model_dir / safe_group / f"horizon_{record['horizon']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        record["estimator"].save_model(path)
        item = {"group": group, "role": record["role"], "horizon": record["horizon"],
                "file": str(path.relative_to(run_dir)), "device": "cuda:0",
                "best_iteration": int(record["estimator"].get_params().get("n_estimators", 0)) - 1}
        if record["role"] == "multi_output":
            reloaded = xgb.XGBRegressor()
            reloaded.load_model(path)
            sample = training_context["train_rows"]
            if group != "global":
                sample = sample[sample["region_id"].astype(str) == group]
            sample_x = sample[training_context["features"]].iloc[:min(8, len(sample))]
            prediction_shape = list(np.asarray(reloaded.predict(sample_x)).shape)
            expected_shape = [len(sample_x), len(TARGET_COLS)]
            if prediction_shape != expected_shape:
                raise RuntimeError(f"Reloaded model {path.name} predicted {prediction_shape}; expected {expected_shape}")
            item["reload_verified_prediction_shape"] = prediction_shape
        model_manifest.append(item)
    artifacts.write_json(run_dir / "model_manifest.json", {"models": model_manifest})
    return model_manifest


def _write_metric_aliases(run_dir: Path, result: dict[str, Any], split: str,
                          source_frame: pd.DataFrame) -> None:
    artifacts.write_json(run_dir / f"{split}_metrics.json", result["reports"]["overall"])
    flat_metrics = [{"section": section, "metric": key, "value": value}
                    for section, values in result["reports"]["overall"].items()
                    for key, value in values.items() if isinstance(value, (int, float))]
    pd.DataFrame(flat_metrics).to_csv(run_dir / f"{split}_metrics.csv", index=False)
    result["reports"]["by_horizon"].to_csv(run_dir / f"{split}_metrics_by_horizon.csv", index=False)
    result["reports"]["by_station"].to_csv(run_dir / f"{split}_metrics_by_station.csv", index=False)
    result["reports"]["station_horizon"].to_csv(run_dir / f"{split}_metrics_station_horizon.csv", index=False)
    region = _region_metrics(result["predictions"], source_frame)
    if not region.empty:
        region.to_csv(run_dir / f"{split}_metrics_by_region.csv", index=False)
    for name in (f"predictions_{split}.parquet", f"predictions_{split}.csv"):
        source = run_dir / name
        if source.exists():
            destination = run_dir / f"{split}_predictions{source.suffix}"
            destination.write_bytes(source.read_bytes())
            break


def _region_metrics(predictions: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if "region_id" not in df.columns:
        return pd.DataFrame()
    region_map = df[[config.STATION_ID_COL, "region_id"]].drop_duplicates(config.STATION_ID_COL)
    joined = predictions.merge(region_map, on=config.STATION_ID_COL, how="left", validate="many_to_one")
    rows = []
    for region, region_frame in joined.groupby("region_id", sort=True, dropna=False):
        groups = [("overall", region_frame)]
        groups.extend((f"t+{int(h)}", part) for h, part in region_frame.groupby("horizon", sort=True))
        for horizon, part in groups:
            values = metrics.point_metrics(part["y_true"].to_numpy(), part["y_pred"].to_numpy())
            rows.append({"region_id": region, "horizon": horizon, **values})
    return pd.DataFrame(rows)


def _preflight_print(info: dict[str, Any], df: pd.DataFrame | None, stations: list[Any] | None,
                     spec: models.RunSpec, args: argparse.Namespace) -> None:
    import xgboost as xgb
    print("[Stage 2 Preflight]")
    print(f"Python: {sys.version.split()[0]}")
    print(f"XGBoost: {xgb.__version__}")
    print(f"CUDA device: {info['diagnostics'].get('gpu_name') or info['diagnostics'].get('nvidia_smi', 'unknown')}")
    print(f"CUDA training: {info.get('cuda_training', 'PASS')}")
    print(f"multi_output_tree: {info.get('multi_output_tree', 'not required')}")
    if df is not None:
        scope = df[df[config.STATION_ID_COL].isin(stations or [])]
        masks = splits.date_masks(scope)
        feature_count = len(models.assemble_feature_columns(spec)) + int(spec.include_station_id) + int(spec.include_region_id)
        print(f"Dataset: {len(scope):,} rows; eligible stations: {len(stations or [])}; features: {feature_count}; targets: {len(TARGET_COLS)}")
        print(f"Train: {int(masks['train'].sum()):,}; Validation: {int(masks['validation'].sum()):,}; Test: {int(masks['test'].sum()):,}")
    print(f"Experiment: {args.experiment} / {EXPERIMENTS[args.experiment]['scope']}")
    print(f"Forecast strategy: {EXPERIMENTS[args.experiment]['forecast']}")
    print("Tuning: hierarchical-grid; early stopping rounds:", args.early_stopping_rounds)
    print("Seed:", args.seed)


def _device_count_for_context(context: dict[str, Any]) -> int:
    scope_count = len(context["groups"])
    heads = 1 if context["spec"].forecast_strategy == "multi" else len(config.FORECAST_HORIZONS)
    return scope_count * heads


def main(argv: list[str] | None = None) -> int:
    run_started = perf_counter()
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    np.random.seed(args.seed)
    spec = make_spec(args.experiment, args.seed)
    gpu_info = cuda_preflight(require_multi=spec.forecast_strategy == "multi")
    _preflight_print(gpu_info, None, None, spec, args)
    if not args.train:
        print("[dry run] CUDA and requested XGBoost architecture probes passed. Add --train to load data and tune.")
        return 0

    df = runner.prepare_dataframe(spec, source=args.source)
    if "region_id" not in df.columns:
        # Metadata only for per-region reporting in global runs. The region code
        # is not appended to global model features.
        df = features.assign_region(df)
    stations = data.resolve_stations(df, args.stations)
    if args.max_stations is not None:
        stations = stations[:args.max_stations]
    if not stations:
        raise ValueError("No eligible stations remain after selection")
    context = make_context(df, stations, spec, "validation")
    if args.development:
        run_id = f"DEV_S2_{args.experiment}__{EXPERIMENTS[args.experiment]['label']}__XGB_CUDA__SEED{args.seed}"
    else:
        run_id = spec.run_id
    run_dir = _artifact_dir(run_id)
    _preflight_print(gpu_info, df, stations, spec, args)
    LOGGER.info("Output: %s; groups=%s; fits per candidate=%d", run_dir, context["groups"], _device_count_for_context(context))

    tuning_start = perf_counter()
    best_params, tuning_summary, _, phase_seconds, fit_counter = hierarchical_search(context, args, run_dir)
    tuning_seconds = perf_counter() - tuning_start

    # Refit selected architecture on TRAIN + VALIDATION using iteration counts
    # selected by validation early stopping. This fit does not inspect TEST.
    final_validation = fit_evaluate(context, best_params, MAX_BOOSTING_ROUNDS,
                                    args.early_stopping_rounds, args.seed,
                                    {"total": 0}, keep_models=True)
    best_rounds = _best_rounds(final_validation["model_records"])
    # The validation artifacts above are selection predictions. The final
    # models below use only Train + Validation; test rows are not selected yet.
    validation_result = final_validation
    final_fit_start = perf_counter()
    final_train = make_final_train_context(df, stations, spec)
    final_models: list[dict[str, Any]] = []
    model_lookup: dict[tuple[str, int], Any] = {}
    for group, train_rows, _ in _fit_groups(final_train["train_rows"], None,
                                            spec, final_train["groups"]):
        if spec.forecast_strategy == "multi":
            estimator = _xgb(_final_refit_params(best_params, best_rounds, group, None, True),
                             args.seed, best_rounds[f"{group}:multi_output"], None, True)
            estimator.fit(train_rows[final_train["features"]], train_rows[TARGET_COLS])
            model_lookup[(group, 0)] = estimator
            final_models.append({"estimator": estimator, "group": group, "horizon": 0,
                                 "role": "multi_output", "feature_columns": final_train["features"],
                                 "target_columns": TARGET_COLS})
        else:
            for j, horizon in enumerate(config.FORECAST_HORIZONS):
                key = f"{group}:t+{horizon}"
                rounds = best_rounds[key]
                estimator = _xgb(_final_refit_params(best_params, best_rounds, group, horizon, False),
                                 args.seed, rounds, None, False)
                estimator.fit(train_rows[final_train["features"]], train_rows[TARGET_COLS[j]])
                model_lookup[(group, horizon)] = estimator
                final_models.append({"estimator": estimator, "group": group, "horizon": horizon,
                                     "role": "direct", "feature_columns": final_train["features"]})
    _synchronize_cuda()
    final_refit_seconds = perf_counter() - final_fit_start
    test_prediction_seconds = None
    test_result = None
    if not args.skip_test:
        # Only now construct the locked test cohort and generate predictions.
        test_context = make_context(df, stations, spec, "test")
        test_blocks: list[pd.DataFrame] = []
        test_started = perf_counter()
        for group, _, test_rows in _model_groups(test_context):
            if spec.forecast_strategy == "multi":
                estimator = model_lookup[(group, 0)]
                prediction = np.asarray(estimator.predict(test_rows[test_context["features"]]))
                test_blocks.extend(_prediction_blocks(test_rows, prediction))
            else:
                for horizon in config.FORECAST_HORIZONS:
                    estimator = model_lookup[(group, horizon)]
                    prediction = np.asarray(estimator.predict(test_rows[test_context["features"]])).reshape(-1)
                    block = test_rows[[config.STATION_ID_COL, config.DATE_COL]].copy()
                    block["horizon"] = horizon
                    block["y_true"] = test_rows[f"target_t{horizon}"].to_numpy()
                    block["y_pred"] = prediction
                    test_blocks.append(block)
        _synchronize_cuda()
        test_predictions = pd.concat(test_blocks, ignore_index=True)
        test_reports = metrics.build_reports(test_predictions)
        test_prediction_seconds = perf_counter() - test_started
        test_result = {"predictions": test_predictions, "reports": test_reports,
                       "importance": pd.DataFrame(), "feature_list": context["features"],
                       "model_records": final_models}

    full_config = {
        "description": f"Stage 2 XGBoost CUDA experiment {args.experiment}: {spec.training_strategy} / {spec.forecast_strategy}; hierarchical grid, validation-only selection.",
        "experiment_id": args.experiment, "algorithm": "xgboost", "scope": spec.training_strategy,
        "training_strategy": spec.training_strategy, "forecast_strategy": spec.forecast_strategy,
        "prediction_output": "native_multi_output_tree" if spec.forecast_strategy == "multi" else "seven_independent_scalar_models",
        "multi_output_implementation": "native XGBoost vector-leaf trees" if spec.forecast_strategy == "multi" else None,
        "multi_strategy": "multi_output_tree" if spec.forecast_strategy == "multi" else None,
        "device": "cuda:0", "xgboost_version": gpu_info["xgboost_version"], "gpu_diagnostics": gpu_info["diagnostics"],
        "seed": args.seed, "source": args.source, "feature_count": len(context["features"]),
        "feature_columns": context["features"], "target_order": TARGET_LABELS,
        "selection_split": "validation", "test_locked_during_tuning": True,
        "tuning_mode": args.tuning_mode, "early_stopping_rounds": args.early_stopping_rounds,
        "max_boosting_rounds": MAX_BOOSTING_ROUNDS, "phase_grids": {"phase1": PHASE1_GRID, "phase2": PHASE2_GRID},
        "phase_budgets": {"phase1": PHASE1_BUDGETS, "phase2": PHASE2_BUDGETS, "phase3": PHASE3_BUDGETS},
        "pruning_enabled": args.phase1_pruning, "development_run": args.development,
        "selected_phase": 3, "best_params": best_params, "best_validation_primary_macro_rmse": tuning_summary["best_validation_score"],
        "best_iteration": tuning_summary["best_iteration"],
        "selected_boosting_rounds": best_rounds, "n_regions": len(context["groups"]),
        "model_fits_during_tuning": fit_counter["total"], "tuning_seconds": tuning_seconds,
        "phase_seconds": phase_seconds, "final_refit_seconds": final_refit_seconds,
        "test_prediction_seconds": test_prediction_seconds,
        "evaluation_splits": ["validation"] if args.skip_test else ["validation", "test"],
    }
    # Stage 1's artifact writer preserves the common metric/prediction/model schema.
    artifacts.save_run(
        run_dir=run_dir, run_id=spec.run_id, run_config=full_config, df=df, stations=stations,
        folds_manifest=pd.DataFrame(), reports=validation_result["reports"],
        predictions=validation_result["predictions"], feature_list=context["features"],
        feature_importance=validation_result["importance"], training_log="Validation-selected models; test was not used during tuning.",
        split="validation", model_records=validation_result["model_records"], prefer_gpu=True,
    )
    _write_metric_aliases(run_dir, validation_result, "validation", df)
    best_params_payload = {"selected_phase": 3, "primary_macro_rmse": tuning_summary["best_validation_score"],
                           "best_iteration": tuning_summary["best_iteration"],
                           "selected_boosting_rounds": best_rounds, "params": best_params}
    artifacts.write_json(run_dir / "best_params.json", best_params_payload)
    importance_status = gpu_info.get("multi_feature_importance", {"status": "not_applicable"})
    artifacts.write_json(run_dir / "feature_importance_status.json", importance_status)

    _save_named_final_models(run_dir, final_models, final_train)
    if not args.skip_test:
        artifacts.save_run(
            run_dir=run_dir, run_id=spec.run_id, run_config=full_config, df=df, stations=stations,
            folds_manifest=pd.DataFrame(), reports=test_result["reports"],
            predictions=test_result["predictions"], feature_list=context["features"],
            feature_importance=test_result["importance"], training_log="Final models refit on train + validation; locked test evaluated once.",
            split="test", model_records=final_models, prefer_gpu=True,
        )
        _write_metric_aliases(run_dir, test_result, "test", df)
    runtime = {
        "phase1_seconds": phase_seconds.get("phase1", 0.0),
        "phase2_seconds": phase_seconds.get("phase2", 0.0),
        "phase3_seconds": phase_seconds.get("phase3", 0.0),
        "tuning_total_seconds": tuning_seconds,
        "final_refit_seconds": final_refit_seconds,
        "test_prediction_seconds": test_prediction_seconds,
        "total_run_seconds": perf_counter() - run_started,
        "total_model_fits_during_tuning": fit_counter["total"],
        "validation_refit_fits": len(validation_result["model_records"]),
        "final_model_fits": len(final_models),
    }
    artifacts.write_json(run_dir / "runtime.json", runtime)
    test_message = ""
    if test_result is not None:
        test_score = test_result["reports"]["overall"]["macro"].get("primary_macro_rmse")
        test_message = f"; test primary macro RMSE={test_score:.6f}"
    LOGGER.info("Completed Stage 2 experiment %s; validation primary macro RMSE=%.6f%s",
                args.experiment, tuning_summary["best_validation_score"], test_message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
