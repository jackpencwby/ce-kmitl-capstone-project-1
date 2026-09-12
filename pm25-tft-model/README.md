# PM2.5 forecasting with Darts TFTModel

Trains a single global Temporal Fusion Transformer across all GISTDA
stations in `../clean-data_preprocess_all_stations_daily.csv`, forecasting
`pm25` 1-7 days ahead.

- **Everything is in one notebook**: [`pm25_tft_training.ipynb`](pm25_tft_training.ipynb)
  — feature selection (condensed, with the data-quality findings), data
  prep, model config, training, and saving artifacts, in order.
- **Full feature-selection rationale + citations**: see
  [`FEATURE_SELECTION.md`](FEATURE_SELECTION.md) (the notebook's Section 2
  is a condensed version of this).
- **TFT background / Darts API notes**: see [`../REFERENCE.md`](../REFERENCE.md).

## Setup

```bash
cd pm25-tft-model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook pm25_tft_training.ipynb
```

## Run

Open `pm25_tft_training.ipynb` and run all cells top to bottom. Section 3
("Configuration") holds everything you'd otherwise pass as CLI flags —
edit the variables there before running:

| Variable | Default | Meaning |
|---|---|---|
| `INPUT_CHUNK_LENGTH` | 28 | days of history the encoder sees |
| `OUTPUT_CHUNK_LENGTH` | 7 | forecast horizon (matches the confirmed "1-7 days, one model" decision) |
| `VAL_START_DATE` / `VAL_HOLDOUT_DAYS` | last 60 days of the dataset | temporal holdout; everything before it is training data |
| `HIDDEN_SIZE` | 32 | TFT state size |
| `N_EPOCHS` | 30 | training epochs |
| `QUANTILES` | `[0.1, 0.5, 0.9]` | matches the TFT paper's quantile set |

Outputs land in `OUTPUT_DIR` (default `artifacts/`):

- `pm25_tft_model.pt` (+ Darts' companion files) — the trained model,
  loadable via `TFTModel.load("artifacts/pm25_tft_model.pt")`.
- `scalers.pkl` — a pickle with `target_scalers`, `past_scalers`,
  `future_scalers` (each a `dict[(station_id, segment_id), Scaler]`) and the
  fitted `static_covariates_transformer`, needed to inverse-transform
  predictions back to µg/m³ and to encode new data consistently at
  inference time.

## Known limitations / things to revisit

- Series are split into independent `(station_id, segment_id)` groups
  wherever the upstream pipeline detected a gap too long to trust as
  continuous (`segment_id` in the source CSV). A station with multiple
  segments gets one `Scaler` per segment rather than one per station, so its
  segments are normalized independently — usually harmless, but worth
  knowing if a station's segments look inconsistently scaled.
- Not every `station_id` has a PM2.5 sensor — some are pure meteorological
  stations. The notebook drops any `(station_id, segment_id)` group whose
  `pm25` is more than `MAX_TARGET_NAN_FRAC` (default 30%) missing before it
  can become a training target; see FEATURE_SELECTION.md, "Target
  availability."  A handful of remaining groups (2 in the full dataset) get
  dropped too because one of their weather columns is fully missing for
  that whole segment and can't be interpolated from nothing.
- Remaining short gaps in the selected covariates (median residual
  missingness is ~0.3% across the real PM2.5-bearing groups) are filled with
  `MissingValuesFiller` per group over the *entire* group span before the
  train/val split, so a small amount of interpolation could technically draw
  on a post-cutoff value to fill a pre-cutoff gap. Given how sparse these
  residual gaps are after the upstream pipeline's own short-gap imputation,
  this is a minor, undocumented-elsewhere simplification rather than a
  design decision to revisit first.
- No evaluation/backtesting cell is included yet (out of scope for this
  pass) — `model.historical_forecasts()` / `model.backtest()` from Darts are
  the natural next step, using `scalers.pkl` to inverse-transform back to
  µg/m³ before computing error metrics.
