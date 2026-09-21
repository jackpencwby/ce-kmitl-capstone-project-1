# Stage 1 experiments — PM2.5 1–7 day forecasting

This folder implements **Stage 1 (one-factor-at-a-time screening)** of
[`Experimental_Plan.md`](../Experimental_Plan.md). Each experiment from the
plan is a separate runnable script; a shared `common/` package holds the data
loading, feature engineering, splitting, metrics and artifact code so every
run uses identical, frozen rules and only the one factor under test changes.

## Experiment scripts

| Script | Plan | What varies | Fixed axes |
|---|---|---|---|
| `E1_1.py` | E1.1 | Training = **Local** (one model set per station) — Stage 1 baseline | XGB / Direct / No neighbor |
| `E1_2.py` | E1.2 | Training = **Global** (all stations pooled + station_id) | XGB / Direct / No neighbor |
| `E1_3.py` | E1.3 | Training = **Global + local tree residual** (OOF residuals) | XGB / Direct / No neighbor |
| `E1_4.py` | E1.4 | Training = **Regional** (North/NE/Central/South by province) | XGB / Direct / No neighbor |
| `E1_5.py` | E1.5 | Training = **Global + local MLP residual** (PyTorch) | XGB / Direct / No neighbor |
| `E2_1.py` | E2.1 | Algorithm = **XGBoost** | Local / Direct / No neighbor |
| `E2_2.py` | E2.2 | Algorithm = **LightGBM** (GPU build if available, else CPU) | Local / Direct / No neighbor |
| `E2_3.py` | E2.3 | Algorithm = **GradientBoostingRegressor** (CPU only) | Local / Direct / No neighbor |
| `E3_1.py` | E3.1 | Forecast = **Direct** (7 models, one per horizon) | Local / XGB / No neighbor |
| `E3_2.py` | E3.2 | Forecast = **Multi-output** (`MultiOutputRegressor`) | Local / XGB / No neighbor |
| `E4_1.py` | E4.1 | Spatial = **No neighbor** | Local / XGB / Direct |
| `E4_2.py` | E4.2 | Spatial = **Unweighted neighbor** mean | Local / XGB / Direct |
| `E4_3.py` | E4.3 | Spatial = **Distance-weighted** (1/(d+ε)) | Local / XGB / Direct |
| `E4_4.py` | E4.4 | Spatial = **Wind + distance** weighted | Local / XGB / Direct |

## Command-line flags (shared by every script)

```
--train                    Actually fit models and write artifacts.
                           Without it, the script does a DRY RUN (prints the
                           resolved configuration, station list and split).
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
--log-level LEVEL          DEBUG / INFO / WARNING.
```

### Examples

```powershell
# From the repository root, using the CUDA-enabled venv.

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
- `validation_start = 2026-05-08`, `test_start = 2026-07-07` (test is locked)
- Horizons `t+1 … t+7`
- 4 expanding walk-forward folds with a 7-day embargo
- Weather-only stations (pm25 missing > 30% in training) are dropped as targets
- Fixed reasonable hyperparameters (no Optuna in Stage 1)
- Metrics: MAE, MSE, RMSE, R², Bias; primary = macro RMSE over station × horizon

## Feature baseline

Built causally in `common/features.py` (plan section 3.1, no look-ahead):

- Target-station PM2.5 lags 1/3/7/30 and rolling mean/std of the lagged series
- Same-day weather (temperature, humidity, pressure, wind speed, wind
  direction sin/cos, rainfall) and fire/hotspot activity
- Calendar cyclical features (known future)
- Same-day CO mean, CO 8-hour maximum (`ug/m3`), and AOD at 500 nm:
  `co_mean_ugm3`, `co_8h_max_ugm3`, `aod500_mean`. Each has a `_missing`
  indicator; missing measurements use a zero placeholder. This keeps rows
  with missing CO/AOD usable for every algorithm without future-value filling.
- Spatial neighbor PM2.5 aggregates (E4 only), always lagged 1/3/7 days

All experiments share the expanded 33-feature baseline. Source CO/AOD columns
must exist; older exports without them fail validation. The original
`example_dataset_1_station.csv` is a legacy schema example; use the new
root CSV and its `clean-data_preprocess-09-19_*` metadata for current runs.
Re-run comparisons together: results from the old dataset/baseline are not
directly comparable. Split dates and forecast horizons remain unchanged.

## Artifacts

Each `--train` run writes `artifacts/<run_id>__<timestamp>/` with
`config.json`, `dataset_manifest.json` (incl. dataset hash + git commit),
`fold_manifest.csv`, `metrics_overall.json`, `metrics_by_horizon.csv`,
`metrics_by_station.csv`, `metrics_station_horizon.csv`,
`predictions_validation.parquet` (CSV fallback), `feature_list.txt`,
`feature_importance.csv`, and `training_log.txt`. Device, library versions
and GPU model are recorded in `config.json` (plan section 16).

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
- Global/local residual models (E1.3, E1.5) use **out-of-fold** global
  predictions for the training residuals; the MLP's scaler is fit on training
  rows only, with early stopping and a per-station fallback to residual = 0
  when a station has too few rows.
