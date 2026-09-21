"""Notebook-compatible result tree shared by E1.1 through E1.5.

Metric names, comparison pairing and CSV columns follow xgboost_1to7_tuned.ipynb.
Model manifests retain each experiment's actual number of fitted estimators.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import artifacts, config


PREDICTION_COLUMNS = ["forecast_origin", "target_date", "station_id",
                      "actual_pm25", "persistence_baseline", "horizon",
                      "predicted_pm25", "error", "absolute_error"]
METRIC_COLUMNS = ["n_samples", "mae", "rmse", "r2", "bias",
                  "comparison_n_samples", "xgb_comparison_mae",
                  "baseline_mae", "mae_improvement", "mae_improvement_pct",
                  "xgb_comparison_rmse", "baseline_rmse",
                  "rmse_improvement", "rmse_improvement_pct"]


def metrics_for(prediction: pd.DataFrame) -> dict:
    """Use the notebook's exact prediction and persistence pairing rules."""
    valid = np.isfinite(prediction.actual_pm25) & np.isfinite(prediction.predicted_pm25)
    actual = prediction.loc[valid, "actual_pm25"]
    estimated = prediction.loc[valid, "predicted_pm25"]
    if not len(actual):
        raise ValueError("No finite predictions to evaluate")
    result = {
        "n_samples": len(actual),
        "mae": float(mean_absolute_error(actual, estimated)),
        "rmse": float(np.sqrt(mean_squared_error(actual, estimated))),
        "r2": float(r2_score(actual, estimated)) if len(actual) >= 2 and actual.nunique() > 1 else np.nan,
        "bias": float((estimated - actual).mean()),
    }
    paired = prediction.loc[valid & np.isfinite(prediction.persistence_baseline)]
    result["comparison_n_samples"] = len(paired)
    for metric, function in (
        ("mae", mean_absolute_error),
        ("rmse", lambda a, b: np.sqrt(mean_squared_error(a, b))),
    ):
        model_score = float(function(paired.actual_pm25, paired.predicted_pm25)) if len(paired) else np.nan
        baseline_score = float(function(paired.actual_pm25, paired.persistence_baseline)) if len(paired) else np.nan
        result[f"xgb_comparison_{metric}"] = model_score
        result[f"baseline_{metric}"] = baseline_score
        result[f"{metric}_improvement"] = baseline_score - model_score
        result[f"{metric}_improvement_pct"] = (
            100 * (baseline_score - model_score) / baseline_score
            if baseline_score > 0 else np.nan)
    return result


def enrich_predictions(prediction: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the runner's long predictions to the notebook's CSV schema."""
    lookup = frame[[config.STATION_ID_COL, config.DATE_COL, config.TARGET_COL]].drop_duplicates(
        [config.STATION_ID_COL, config.DATE_COL])
    result = prediction.merge(lookup, on=[config.STATION_ID_COL, config.DATE_COL],
                              how="left", validate="many_to_one")
    result = result.rename(columns={config.DATE_COL: "forecast_origin",
                                    config.STATION_ID_COL: "station_id",
                                    config.TARGET_COL: "persistence_baseline",
                                    "y_true": "actual_pm25", "y_pred": "predicted_pm25"})
    result["target_date"] = result.forecast_origin + pd.to_timedelta(result.horizon, unit="D")
    result["error"] = result.predicted_pm25 - result.actual_pm25
    result["absolute_error"] = result.error.abs()
    return result[PREDICTION_COLUMNS].sort_values(
        ["horizon", "station_id", "target_date"]).reset_index(drop=True)


def _plot(prediction: pd.DataFrame, folder: Path, title: str,
          time_series: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    if time_series:
        fig, ax = plt.subplots(figsize=(15, 6))
        for column, label, style in (
            ("actual_pm25", "Actual", "-"),
            ("predicted_pm25", "Prediction", "-"),
            ("persistence_baseline", "Persistence baseline", "--"),
        ):
            ax.plot(prediction.target_date, prediction[column], style,
                    label=label, linewidth=1.3)
        ax.set(xlabel="Target date", ylabel="PM2.5 (µg/m³)", title=title)
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(folder / "time_series.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(prediction.target_date, prediction.error)
        ax.axhline(0, color="#555555", linestyle="--")
        ax.set(xlabel="Target date", ylabel="Predicted − actual (µg/m³)",
               title=f"{title} | Residuals")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(folder / "residuals.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(prediction.actual_pm25, prediction.predicted_pm25,
               alpha=0.3, s=12)
    low = float(min(prediction.actual_pm25.min(), prediction.predicted_pm25.min()))
    high = float(max(prediction.actual_pm25.max(), prediction.predicted_pm25.max()))
    if high == low:
        high = low + 1.0
    ax.plot([low, high], [low, high], "--", color="#555555")
    ax.set(xlabel="Actual PM2.5 (µg/m³)", ylabel="Predicted PM2.5 (µg/m³)",
           title=title, xlim=(low, high), ylim=(low, high))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def _model_index(run_dir: Path, horizon: int) -> list[dict]:
    index = []
    for split in ("validation", "test"):
        path = run_dir / "model" / split / "manifest.json"
        if not path.exists():
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for model in manifest["models"]:
            if int(model["horizon"]) != horizon:
                continue
            entry = dict(model)
            entry["split"] = split
            entry["file"] = str((Path("..") / "model" / split / model["file"]).as_posix())
            if "training_history_file" in model:
                entry["training_history_file"] = str((Path("..") / "model" / split /
                                                      model["training_history_file"]).as_posix())
            index.append(entry)
    return index


def export(run_dir: Path, frame: pd.DataFrame, predictions: dict[str, pd.DataFrame],
           feature_list: list[str]) -> None:
    """Write the notebook's horizon/station result structure for one E1 run."""
    config_copy = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    dataset_manifest = json.loads((run_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    selected = set(dataset_manifest["stations_used"])
    frame = frame[frame[config.STATION_ID_COL].astype(str).isin(selected)]
    config_copy["notebook_output_schema"] = "xgboost_1to7_tuned"
    config_copy["features"] = feature_list
    config_copy["horizons"] = list(config.FORECAST_HORIZONS)
    config_copy["source"] = dataset_manifest.get("master_local_path")
    config_copy["source_sha256"] = dataset_manifest.get("master_sha256")
    artifacts.write_json(run_dir / "run_config.json", config_copy)
    pd.Series({column: int(frame[column].isna().sum()) if column in frame else 0
               for column in feature_list}, name="missing_count").to_csv(
                   run_dir / "feature_missingness.csv")
    target_audit = [{"horizon": h, "source": "exact_day_lookup",
                     "available_rows": int(frame[f"target_t{h}"].notna().sum())}
                    for h in config.FORECAST_HORIZONS]
    pd.DataFrame(target_audit).to_csv(run_dir / "target_audit.csv", index=False)

    enriched = {split: enrich_predictions(pred, frame) for split, pred in predictions.items()}
    all_metrics, all_station_metrics, split_rows = [], [], []
    for h in config.FORECAST_HORIZONS:
        train_rows = frame[frame[f"target_t{h}"].notna()].copy()
        train_rows["target_date"] = train_rows[config.DATE_COL] + pd.Timedelta(days=h)
        train_rows = train_rows[(train_rows[config.DATE_COL] < pd.Timestamp(config.VALIDATION_START))
                                & (train_rows.target_date < pd.Timestamp(config.VALIDATION_START))]
        split_rows.append({"horizon": h, "split": "train", "n_samples": len(train_rows),
                           "origin_start": train_rows[config.DATE_COL].min(),
                           "origin_end": train_rows[config.DATE_COL].max(),
                           "target_start": train_rows.target_date.min(),
                           "target_end": train_rows.target_date.max()})
        folder = run_dir / f"horizon_{h:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        model_index = _model_index(run_dir, h)
        artifacts.write_json(folder / "model_manifest.json", {"horizon": h, "models": model_index})
        primary = [item for item in model_index if item["split"] == "validation"
                   and item["role"] == "main"]
        validation_prediction = enriched["validation"]
        validation_prediction = validation_prediction[
            (validation_prediction.horizon == h) & validation_prediction.actual_pm25.notna()]
        validation_metric = (metrics_for(validation_prediction)["rmse"]
                             if len(validation_prediction) else None)
        horizon_params = config_copy.get("params_by_horizon", {}).get(
            str(h), config_copy.get("fixed_params", {}))
        selected_params = {key: value for key, value in horizon_params.items()
                           if key != "n_estimators"}
        artifacts.write_json(folder / "best_params.json", {
            "params": selected_params,
            "validation_rmse": validation_metric,
            "best_iteration": primary[0].get("best_iteration") if len(primary) == 1 else None,
            "prediction_trees": (primary[0]["best_iteration"] + 1
                                 if len(primary) == 1 and "best_iteration" in primary[0] else None),
            "full_model_params": primary[0].get("full_model_params") if len(primary) == 1 else None,
            "selected_models": [{"group": item["group"], "role": item["role"],
                                 "best_iteration": item.get("best_iteration"),
                                 "full_model_params": item.get("full_model_params")}
                                for item in model_index if item["split"] == "validation"],
        })
        # A single pooled XGBoost model can occupy the notebook's model.ubj slot.
        if (config_copy.get("training_strategy") == "global" and len(primary) == 1
                and primary[0]["file"].endswith(".json")):
            import xgboost as xgb
            source = (folder / primary[0]["file"]).resolve()
            model = xgb.XGBRegressor()
            model.load_model(source)
            model.save_model(folder / "model.ubj")
        histories = []
        for item in model_index:
            if item["split"] != "validation" or "training_history_file" not in item:
                continue
            history_path = (folder / item["training_history_file"]).resolve()
            history = pd.read_csv(history_path)
            history.insert(0, "group", item["group"])
            histories.append(history)
        history_table = (pd.concat(histories, ignore_index=True) if histories else
                         pd.DataFrame(columns=["group", "iteration", "train_rmse", "validation_rmse"]))
        history_table.to_csv(folder / "training_history.csv", index=False)
        if not history_table.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            for group, data in history_table.groupby("group"):
                ax.plot(data.iteration, data.validation_rmse, alpha=0.45,
                        label=str(group) if history_table.group.nunique() <= 8 else None)
            ax.set(xlabel="Boosting iteration", ylabel="Validation RMSE (µg/m³)",
                   title=f"Day +{h} | validation history")
            if history_table.group.nunique() <= 8:
                ax.legend()
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(folder / "training_curve.png", dpi=150)
            plt.close(fig)
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.text(0.5, 0.5, "No per-iteration validation history recorded",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            fig.savefig(folder / "training_curve.png", dpi=150)
            plt.close(fig)
        importance = run_dir / "feature_importance.csv"
        if importance.exists():
            table = pd.read_csv(importance)
            if "model" in table:
                table = table[table.model.astype(str).str.endswith(f"h{h}")]
            table.to_csv(folder / "feature_importance.csv", index=False)
        station_rows = []
        for split, prediction in enriched.items():
            part = prediction[(prediction.horizon == h) & prediction.actual_pm25.notna()].copy()
            if part.empty:
                continue
            destination = folder / split
            destination.mkdir(exist_ok=True)
            part.to_csv(destination / "predictions.csv", index=False, encoding="utf-8-sig")
            result = {"horizon": h, "split": split, **metrics_for(part)}
            pd.DataFrame([result]).to_csv(destination / "metrics.csv", index=False)
            all_metrics.append(result)
            _plot(part, destination, f"Day +{h} | {split} | RMSE={result['rmse']:.3f}", False)
            split_rows.append({"horizon": h, "split": split, "n_samples": len(part),
                               "origin_start": part.forecast_origin.min(),
                               "origin_end": part.forecast_origin.max(),
                               "target_start": part.target_date.min(),
                               "target_end": part.target_date.max()})
            if split == "test":
                for station_id, sub in part.groupby("station_id", sort=True):
                    station_dir = destination / str(station_id)
                    station_dir.mkdir(exist_ok=True)
                    row = {"horizon": h, "station_id": station_id,
                           "start_date": sub.target_date.min(),
                           "end_date": sub.target_date.max(), **metrics_for(sub)}
                    sub.to_csv(station_dir / "predictions.csv", index=False, encoding="utf-8-sig")
                    pd.DataFrame([row]).to_csv(station_dir / "metrics.csv", index=False)
                    _plot(sub, station_dir,
                          f"Station {station_id} | Day +{h} | Test | RMSE={row['rmse']:.3f}", True)
                    station_rows.append(row)
        station_summary = pd.DataFrame(station_rows)
        if len(station_summary):
            station_summary.sort_values("rmse").to_csv(folder / "station_summary.csv", index=False)
            all_station_metrics.extend(station_summary.sort_values("rmse").to_dict("records"))
        else:
            pd.DataFrame(columns=["horizon", "station_id", "start_date", "end_date"] + METRIC_COLUMNS).to_csv(
                folder / "station_summary.csv", index=False)
        validation_rows = [row for row in all_metrics
                           if row["horizon"] == h and row["split"] == "validation"]
        fixed = selected_params
        pd.DataFrame([{
            "trial": 0, "validation_rmse": validation_rows[0]["rmse"] if validation_rows else np.nan,
            "best_iteration": (primary[0].get("best_iteration") if len(primary) == 1 else np.nan),
            "seconds": np.nan, "selection_method": "fixed_no_search", **fixed,
        }]).to_csv(folder / "tuning_results.csv", index=False)
    summary = pd.DataFrame(all_metrics)
    expected = {(h, split) for h in config.FORECAST_HORIZONS
                for split in ("validation", "test")}
    actual = {(int(row["horizon"]), row["split"]) for row in all_metrics}
    if actual != expected:
        raise RuntimeError(f"Incomplete notebook-compatible results: {sorted(expected - actual)}")
    summary.to_csv(run_dir / "horizon_summary.csv", index=False)
    pd.DataFrame(all_station_metrics).to_csv(run_dir / "all_station_summary.csv", index=False)
    pd.DataFrame(split_rows).to_csv(run_dir / "split_summary.csv", index=False)
    test_summary = summary[summary.split.eq("test")].sort_values("horizon")
    if not test_summary.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, metric in zip(axes, ("mae", "rmse")):
            ax.plot(test_summary.horizon, test_summary[f"xgb_comparison_{metric}"],
                    "o-", label="Model")
            ax.plot(test_summary.horizon, test_summary[f"baseline_{metric}"],
                    "s--", label="Persistence baseline")
            ax.set(xlabel="Forecast horizon (days)", ylabel=metric.upper(),
                   title=f"Test {metric.upper()} by horizon")
            ax.grid(alpha=0.2)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(run_dir / "horizon_comparison.png", dpi=150)
        plt.close(fig)
    artifacts.write_json(run_dir / "completion.json", {
        "status": "complete", "smoke_test": False,
        "horizons": list(config.FORECAST_HORIZONS),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "xgboost_1to7_tuned",
        "metric_rows": len(summary), "test_station_rows": len(all_station_metrics),
    })
