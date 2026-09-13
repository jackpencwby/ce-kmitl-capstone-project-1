# Per-station evaluation step for the PM2.5 TFT notebook

Date: 2026-09-13
Status: approved (approach A)
Scope: `pm25-tft-model/pm25_tft_training.ipynb` only.

## Goal

After training, evaluate the fitted `TFTModel` on the 60-day validation
window and answer: how accurate is each station, and which station is most
accurate? Deliverables are a per-station R² chart (all stations, sorted) and
a printed top-5 / bottom-5 table.

## Evaluation protocol (user-confirmed)

Rolling backtest across the whole validation window via Darts'
pre-trained backtest:

- `model.historical_forecasts(series=<full-span scaled targets>,
  past_covariates=..., future_covariates=..., forecast_horizon=7, stride=7,
  start=val_start_date, start_format="value", retrain=False,
  last_points_only=False, overlap_end=False, num_samples=1)`.
- ~9 windows x 7 days = ~60 point predictions per station. With the
  quantile likelihood and `num_samples=1`, Darts returns the median (q0.5)
  as the point forecast.
- `start=val_start_date` on full-span series lets window 1 encode the last
  28 days of *training-period* actuals before the cutoff (realistic
  deployment semantics); no leakage since each window only uses data
  before its own start.
- `retrain=False`: the model is not retrained per window (predict-only).

## Changes

### 1. Prep changes (Section 4 cells — must re-run prep, ~15 s)

- Add `import matplotlib.pyplot as plt` to the setup cell (darts already
  depends on matplotlib; no `requirements.txt` change).
- Extend the `Datasets` dataclass and `prepare_datasets` with:
  - `val_keys: list[tuple[str, str]]` — the `(station_id, segment_id)` of
    each validation series, in order, so the eval cell can pick the right
    per-group `target_scaler` for inverse-transforming.
  - `full_targets: list[TimeSeries]` — each group's full-span scaled
    target (train + val portions), the series the backtest runs on.

### 2. New Section 8 cell: evaluation

- Run the backtest once over all val groups (full-span scaled targets +
  the existing full-span scaled past/future covariates).
- Per group: concatenate the per-window forecast chunks, inverse-transform
  forecast and actual values to µg/m³ with that group's `target_scaler`
  (MinMaxScaler inverse is exact), align on timestamps.
- Per group metrics: `sklearn.metrics.r2_score` and `mean_absolute_error`
  on µg/m³ (scikit-learn is already a dependency).
- Station aggregation: stations with multiple segments get the **mean of
  their segments' R²/MAE** (pooling segments would inflate R² via
  between-segment level differences).
- Edge cases: groups with < 2 forecast points or zero-variance actuals
  get R² = NaN, are excluded from the ranking and from the "most
  accurate" pick, and are reported by count.

### 3. Output

- One tall horizontal bar chart: every station sorted by R² (best at top,
  negative R² at the bottom, reference line at 0). The most accurate
  station is drawn in a contrasting color with a text annotation
  (station id + R²). Figure height scales with the station count.
- Printed table: top 5, bottom 5, overall mean R², and the winner.

Runtime: a few minutes on CPU, well under a minute on GPU (predict-only).

## Non-goals

- No additional artifact files (CSV export, saved PNG) — user chose the
  single in-notebook chart.
- No actual-vs-predicted overlay curves for individual stations.
- No backtesting over the training period.
