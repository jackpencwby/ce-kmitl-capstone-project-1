# Stage 1 experiments — PM2.5 1–7 day forecasting

This folder implements **Stage 1 (one-factor-at-a-time screening)** of
[`Experimental_Plan.md`](../Experimental_Plan.md). Each experiment from the
plan is a separate runnable script; a shared `common/` package holds the data
loading, feature engineering, splitting, metrics and artifact code so every
run uses identical, frozen rules and only the one factor under test changes.

## Experiment scripts

| Script | Plan | What varies | Fixed axes |
|---|---|---|---|
| `E1_1.py` | E1.1 | Training = **Local** (one model per station and horizon) | XGB / Direct / No neighbor |
| `E1_2.py` | E1.2 | Training = **Global** (all stations pooled + station_id) | XGB / Direct / No neighbor |
| `E1_3.py` | E1.3 | Training = **Global + local tree residual** (OOF residuals) | XGB / Direct / No neighbor |
| `E1_4.py` | E1.4 | Training = **Regional** (North/NE/Central/South by province) | XGB / Direct / No neighbor |
| `E1_5.py` | E1.5 | Training = **Global + local MLP residual** (PyTorch) | XGB / Direct / No neighbor |
| `E2_1.py` | E2.1 | Algorithm = **XGBoost** | Local / Direct / No neighbor |
| `E2_2.py` | E2.2 | Algorithm = **LightGBM** (GPU build if available, else CPU) | Local / Direct / No neighbor |
| `E2_3.py` | E2.3 | Algorithm = **GradientBoostingRegressor** (CPU only) | Local / Direct / No neighbor |
| `E3_1.py` | E3.1 | Forecast = **Direct** (7 models, one per horizon) | Local / XGB / No neighbor |
| `E3_2.py` | E3.2 | Forecast = **Multi-output** (independent estimators on common complete-case rows) | Local / XGB / No neighbor |
| `E4_1.py` | E4.1 | Spatial = **No neighbor** | Local / XGB / Direct |
| `E4_2.py` | E4.2 | Spatial = **Unweighted neighbor** mean | Local / XGB / Direct |
| `E4_3.py` | E4.3 | Spatial = **Distance-weighted** (1/(d+ε)) | Local / XGB / Direct |
| `E4_4.py` | E4.4 | Spatial = **Wind + distance** weighted | Local / XGB / Direct |

## Command-line flags (shared by every script)

```
--train                    Actually fit models and write artifacts.
                           Without it, the script does a DRY RUN (prints the
                           resolved configuration, station list and split).
                           A training run evaluates validation and test
                           separately and saves the models used for each.
--stations S [S ...]       Station id(s) to train/score, or 'all' (default).
                           e.g. --stations 72 36 108
                           Ineligible (weather-only) ids are dropped with a
                           warning.
--source {auto,local,gcs}  Where to read the master table. 'auto' (default)
                           uses the local CSV if present, else downloads from
                           GCS. 'gcs' forces a fresh download.
--check-gcs                Verify the GCP connection from .env, then continue.
--max-stations N           Cap the number of stations (quick smoke tests).
--prefer-cpu               Force CPU even when a CUDA GPU is available.
--upload-gcs               Upload the completed artifact folder to GCS (requires --train).
--gcs-artifacts-prefix P   Override the bucket folder used by --upload-gcs.
--keep-local               Keep the local artifacts after upload (default).
--no-keep-local            Delete local artifacts after a successful upload.
--log-level LEVEL          DEBUG / INFO / WARNING.
```

### Examples

```powershell
# From the repository root, using the project's venv.

# Dry run: see the config + which stations are eligible.
.venv/Scripts/python.exe experiment-stage-1/E1_1.py

# Train the Local baseline on one station.
.venv/Scripts/python.exe experiment-stage-1/E1_1.py --train --stations 72

# Train the Global model on every eligible station, data pulled from GCS.
.venv/Scripts/python.exe experiment-stage-1/E1_2.py --train --stations all --source gcs

# Wind+distance spatial run on a handful of stations.
.venv/Scripts/python.exe experiment-stage-1/E4_4.py --train --stations 72 36 108

# Verify the GCP connection defined in .env.
.venv/Scripts/python.exe experiment-stage-1/E1_1.py --check-gcs
```

## GCP connection (`.env`)

The repo `.env` holds Cloud Storage **HMAC** credentials (the access key
starts with `GOOG1E…`). `common/data.py` reads `.env` and talks to the
S3-compatible interoperability endpoint `https://storage.googleapis.com`
via `boto3`, using:

```
AWS_ACCESS_KEY_ID      HMAC access key
AWS_SECRET_ACCESS_KEY  HMAC secret
GCS_BUCKET             kmitl-capstone-project-data-bucket
```

The dataset is read from
`gs://<GCS_BUCKET>/clean-data/preprocess-09-19/<station>/daily_dataset.csv`,
for example `1003_Nakhon Nayok Weather Observing Station/daily_dataset.csv`.
`--source gcs` discovers all station folders and combines their daily CSVs
into `clean-data_preprocess-09-19_all_stations_daily.csv` in the repo root.
Summary files and combined exports in the bucket are excluded to avoid
duplicate rows. A failed download leaves the previous local cache intact.
Runs prefer this local copy by default; `--source local` requires it.

## Frozen rules (do not change per experiment)

Set once in `common/config.py` (plan section 3):

- Seed `42`
- `validation_start = 2025-11-19`, `test_start = 2026-04-13`, matching the
  original notebook's 70/15/15 split boundaries. The September 19 dataset
  extends farther in time, so these fixed dates no longer give exact 70/15/15
  proportions on the new file.
- Horizons `t+1 … t+7`
- 4 expanding walk-forward folds with a 7-day embargo
- Weather-only stations (pm25 missing > 30% in training) are dropped as targets
- XGBoost parameters selected per horizon from `best_params_t1.json` through `best_params_t7.json` (no search during Stage 1 runs)
- LightGBM and GradientBoostingRegressor keep their own fixed parameters in `common/config.py`
- Metrics: MAE, MSE, RMSE, R², Bias; primary = macro RMSE over station × horizon

## Feature baseline

The shared baseline takes the notebook's 106 precomputed numeric features in
the same order, then adds 22 CO/AOD measurements, lag/rolling/difference
features, and missing indicators from the September 19 export. The full list
is frozen in `common/notebook_features.py`. Source NaNs remain NaNs for XGBoost
and LightGBM; GBR imputes them using a constant in its saved pipeline. Rows
must pass `xgboost_history_valid`, as in the notebook. Targets are exact future
calendar days within the same station and segment. E4 also adds neighbor
features. Old artifacts use a different split and feature set; rerun all
comparisons on the new configuration.

Every XGBoost experiment in E1–E4 uses `params` and
`full_model_params.n_estimators` from `best_params_t{h}.json` for horizon `h`.
The files currently specify a 2,000-tree ceiling. E1.1 fits one model per
station and horizon, using
100-round early stopping on validation. No parameter search runs during an
experiment. The historical `best_iteration` and `prediction_trees` in the
source files describe the earlier tuning runs; each new E1 fit selects its
own best iteration.
Targets are looked up on the exact future date in the same station segment.
The selected validation model is reused for test; test targets never enter
early stopping. Its saved model manifest includes `best_iteration` and a
training-history CSV for each station and horizon.
E1.2–E1.5 use the same per-horizon XGBoost parameters and validation early
stopping
for their main global or regional models. The E1.3 tree residual correctors
use the same per-horizon XGBoost parameters on out-of-fold residuals; auxiliary
out-of-fold fits use the tail of their own training window for early stopping,
with a horizon-length target cutoff. All E1 test predictions reuse the models
selected before test, including any local residual correctors.
E2–E4 XGBoost runs use the same per-horizon parameters without early stopping.
Their test models are fitted again on train plus validation data. E3.2 fits
independent horizon estimators on rows with all seven targets available.

## Artifacts

Each `--train` run writes `artifacts/<run_id>__<timestamp>/` with
`config.json`, `dataset_manifest.json` (incl. dataset hash + git commit),
`fold_manifest.csv`, `metrics_overall.json`, `metrics_by_horizon.csv`,
`metrics_by_station.csv`, `metrics_station_horizon.csv`,
`predictions_validation.parquet` (CSV fallback), `feature_list.txt`,
`feature_importance.csv`, and `training_log.txt`. Device, library versions
and GPU model are recorded in `config.json` (plan section 16).
For XGBoost, `config.json` records `params_by_horizon` and `params_source`;
`fixed_params` contains the t+1 recipe for compatibility with older readers.
The same metrics and prediction files are also written with a `_test` suffix
for the test period. `metrics_by_station_test.csv` contains each station's
MAE, MSE, RMSE, R² and bias across horizons; `metrics_overall_test.json`
contains pooled (micro) and station-equal (macro) metrics. Fitted estimators
are in `model/validation/` and `model/test/`, each with a `manifest.json`.
For E1, the test manifest points to the validation-selected models; no second
fit is made. For E2–E4, the test fit uses train plus validation origins, but
excludes any training target whose forecast date reaches the test period.
The validation fit uses only training origins with labels available before
validation starts.

For E1.1–E1.5, the runner also writes the notebook-compatible tree:
`horizon_01/` through `horizon_07/` with `validation/predictions.csv`,
`validation/metrics.csv`, `test/predictions.csv`, `test/metrics.csv`,
`test/<station_id>/predictions.csv`, `test/<station_id>/metrics.csv`, plots,
`station_summary.csv`, `best_params.json`, `tuning_results.csv`,
`training_history.csv`, `training_curve.png`, `feature_importance.csv`, and
`model_manifest.json`. The run root gets `horizon_summary.csv`,
`all_station_summary.csv`, `run_config.json`, `split_summary.csv`,
`target_audit.csv`, `feature_missingness.csv`, `horizon_comparison.png`,
and `completion.json`. CSV prediction and metric column names follow
`xgboost_1to7_tuned.ipynb`; persistence comparisons use only paired rows.
All E1 targets are looked up on the exact future date within the same station
segment, matching the notebook's target audit.
`model.ubj` is present only when a horizon truly has one pooled XGBoost
model (E1.2). Other E1 strategies use the manifest for their multiple models.
The `tuning_results.csv` records the selected per-horizon configuration as
`selection_method=fixed_no_search`; it does not imply a parameter search.
Where a strategy has no per-iteration validation history, `training_history.csv`
has only headers and `training_curve.png` states that no history was recorded.

## Install & test

```powershell
.venv/Scripts/python.exe -m pip install -r experiment-stage-1/requirements.txt

# Unit tests for dataset loading, CO/AOD inputs, and spatial helpers.
.venv/Scripts/python.exe -m unittest discover -s experiment-stage-1/tests -v
```

## Notes

- Stage 1 screens options; it does not declare a final winner. Comparisons
  use the same split, features, seed and evaluation cohort per plan section 3.
- LightGBM GPU is used only when the installed build supports it; otherwise it
  falls back to CPU automatically (accuracy is comparable; do not compare CPU
  vs GPU timing directly, plan section 3.3).
- Global/local residual models (E1.3, E1.5) use expanding **out-of-fold** global
  predictions within the fit period for training residuals; the MLP's scaler is fit on training
  rows only, with early stopping and a per-station fallback to residual = 0
  when a station has too few rows.
