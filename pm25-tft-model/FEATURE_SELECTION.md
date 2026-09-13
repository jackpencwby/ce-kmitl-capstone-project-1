# Feature selection for the PM2.5 TFT model

The source table (`clean-data_preprocess_all_stations_daily.csv`, via
`clean-data_preprocess_all_stations_daily_feature_manifest.yaml`) has 257
columns: raw weather, hotspot/fire detections, calendar fields, cyclical
encodings, QC flags, imputation flags, lag features, rolling features,
difference features, and precomputed multi-horizon targets. Most of that is
either pipeline metadata (not a predictor) or redundant with something else
already selected. This document lists what was kept, what was dropped, and
why — so the choices can be checked and revisited rather than taken on
faith.

The manifest already ships curated `views` (`TFT_STATIC_FEATURES`,
`TFT_KNOWN_TIME_FEATURES`, `TFT_OBSERVED_FEATURES`) clearly meant for this
exact model. The selection below starts from those views and trims further —
see "Corrections to the manifest's TFT views" for the one place it was
overridden rather than followed.

## Final feature set

**R2 training revision (2026-09-13):** the feature columns below are unchanged.
The notebook now uses deterministic MSE, training-only standard scaling,
validation-based early stopping and best-checkpoint loading. For a fixed
evaluation target, maximizing R2 is equivalent to minimizing squared error;
quantile loss instead targets quantiles. This motivates trying MSE, but does
not establish a measured improvement. Per-group standardized MSE is only a
surrogate for the reported mean station R2. See the
[scikit-learn scoring guidance](https://scikit-learn.org/stable/modules/model_evaluation.html#which-scoring-function-should-i-use)
and [Darts TFT loss_fn documentation](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html).
We retain the 28-day encoder and seven-day direct horizon. The 64-unit hidden
state and 16-unit continuous representation follow the starting heuristic in
`REFERENCE.md` section 9; these are candidate settings, not experimentally
selected optima. No additional lag, geographic, weather, or fire features
have been justified by an ablation in this revision.

Residual notebook gaps are now forward-filled with leading incomplete rows
trimmed, replacing the interpolation described in the historical notes below.
Sensor eligibility uses training dates. Validation includes training encoder
context and ends before a separate final 60-day test period. Evaluation excludes
source labels marked missing or imputed. These flags are used for scoring only,
not as model inputs. Upstream CSV imputation remains an audit limitation.

| Role (Darts) | Columns | Count |
|---|---|---|
| Target | `pm25` | 1 |
| Static covariate | `station_id` | 1 |
| Known-future covariate | `year_index`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`, `doy_sin`, `doy_cos` | 7 |
| Past (observed-only) covariate | `temperature_avg`, `temperature_range`, `humidity_avg`, `humidity_range`, `pressure_avg`, `pressure_range`, `wind_speed_avg`, `wind_direction_avg_sin`, `wind_direction_avg_cos`, `log1p_rainfall`, `rain_event`, `hotspot_present`, `log1p_hotspot_count`, `log1p_hotspot_frp_sum` | 14 |

23 model inputs total, down from 257 raw columns / 33 columns in the
manifest's own `TFT_OBSERVED_FEATURES` view.

**Note on `sunshine_duration`**: it was in the original past-covariate list
below (kept for its link to boundary-layer mixing height/photochemical
activity) but was **removed** after checking its actual missingness
per-`(station_id, segment_id)` group against the CSV directly: 80 of the 537
groups that otherwise have a usable pm25 sensor (~15%) have
`sunshine_duration` missing for **100%** of their rows — not scattered gaps,
but stations with no sunshine sensor installed at all. Interpolation cannot
recover a column that's entirely missing for a group, and fabricating a
value for 15% of the training data would misrepresent real conditions
rather than fill a genuine gap, so the column was dropped instead of worked
around. `src/features.py` carries this same note next to `PAST_FEATURES`.

## Why each group looks the way it does

### Static: `station_id` only

`station_id` (227 stations) already uniquely identifies a physical
location, so it is a sufficient key for a per-entity embedding — this
mirrors how the TFT paper itself uses a single entity ID (store ID in the
Retail dataset, meter ID in Electricity) as the static covariate rather than
also feeding the entity's separately-known attributes. Adding `lat`/`long`/
`province`/`amphoe` on top would be redundant (they are a deterministic
function of `station_id`) and would only reintroduce the ~4% missingness
those columns carry in the raw table. Confirmed with the user directly
(2026-09-12): use `station_id` alone, no lat/long/province.

### Known-future: calendar only, no weather

The paper's contract for `future_covariates` is that they must be knowable
in advance for the full forecast horizon — TFT's decoder literally attends
over them as queries (Lim et al. 2019, §3.2; see `REFERENCE.md` §3). This
dataset has no weather-forecast feed, so the only columns that are honestly
"known future" are calendar-derived: `dow_sin/cos`, `month_sin/cos`,
`doy_sin/cos` for seasonality, plus `year_index` to let the model pick up
any multi-year trend/drift that pure Fourier terms cannot represent (e.g.
Northern Thailand's burning season has shown different intensity year to
year — see the FINNv2.5 Chiang Mai emissions study below).

Sine/cosine pairs are used instead of the raw integer calendar columns
(`month`, `day`, `day_of_week`, `day_of_year`, `week_of_year`, `quarter`) for
the standard reason cyclical encoding is preferred for neural forecasters:
raw integers imply a false distance between adjacent-but-wrapped values
(e.g. December and January look maximally far apart to a model reading raw
month numbers, when they are next to each other), while a sin/cos pair keeps
adjacent time points close in feature space. This is documented practice,
e.g. the `CyclicalFeatures` transformer in the `feature-engine` library
(https://feature-engine.trainindata.com/en/1.7.x/user_guide/creation/CyclicalFeatures.html).
Both raw and cyclical forms exist in the source table; only the cyclical
form is kept, since keeping both is pure collinearity.

### Past (observed-only): trimmed meteorology + fire activity

Every kept column has a documented physical link to PM2.5, and every
dropped column was dropped because it duplicates a kept one.

**Kept, with the evidence:**
- `wind_speed_avg` — wind speed is consistently negatively correlated with
  PM2.5 (stronger wind disperses pollutants); reported r ≈ −0.24 to −0.30 in
  a multi-city Chinese study
  (https://arxiv.org/pdf/1708.06072, also at
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5750928/).
- `wind_direction_avg_sin/cos` — transport direction determines whether a
  station sits downwind of a burning region; moved here from the manifest's
  `TFT_KNOWN_TIME_FEATURES` view (see correction below).
- `temperature_avg`, `temperature_range` — temperature is also inversely
  correlated with PM2.5 in the same study (r ≈ −0.09 to −0.36); the diurnal
  `*_range` is kept instead of `*_min`/`*_max` separately as a compact proxy
  for atmospheric stability/inversion strength — Northern Thailand PM2.5
  literature specifically calls out temperature inversions as a driver of
  haze accumulation
  (https://pmc.ncbi.nlm.nih.gov/articles/PMC11280843/, WRF-Chem assessment
  of transboundary PM2.5 from biomass burning in Northern Thailand).
- `humidity_avg`, `humidity_range` — humidity's relationship with PM2.5 is
  more site-dependent (positive in some studies, e.g. rs ≈ 0.38 in one
  cited result) but still consistently retained as a standard predictor
  across the meteorology–PM2.5 literature surveyed above.
- `pressure_avg`, `pressure_range` — weak but consistently reported
  correlation (r ≈ 0.03–0.30) tied to synoptic-scale stagnation events.
- `log1p_rainfall`, `rain_event` — precipitation scavenging (washout) is a
  well-documented PM2.5 removal mechanism, reported as a negative
  correlation with particulate matter in the same survey; `log1p` handles
  the heavy right-skew/zero-inflation typical of daily rainfall, and
  `rain_event` keeps a clean binary wet/dry signal that log1p smooths away.
- `hotspot_present`, `log1p_hotspot_count`, `log1p_hotspot_frp_sum` — fire
  activity is the dominant PM2.5 driver in this region specifically: local
  reporting attributes roughly 70% of Chiang Mai's April PM2.5 to biomass
  burning, and fire radiative power (FRP) — summed here as
  `log1p_hotspot_frp_sum` — is the standard remote-sensing intensity proxy
  used to drive burning-emission models such as FINNv2.5, whose Northern
  Thailand run tracked PM2.5 emission timing/intensity closely against
  MODIS/VIIRS FRP
  (https://pmc.ncbi.nlm.nih.gov/articles/PMC12846012/, "Near Real-Time
  Biomass Burning PM2.5 Emission Estimation ... in Northern Thailand Using
  FINNv2.5"). `hotspot_present` (any detection, binary) and
  `log1p_hotspot_count` (how many) separate "is there burning nearby" from
  "how much," which `log1p_hotspot_frp_sum` alone would blur.

**Dropped, and why:**
- `temperature_min`/`temperature_max`, `humidity_min`/`humidity_max`,
  `pressure_min`/`pressure_max` — algebraically redundant once `*_avg` and
  `*_range` are kept (`range = max − min`); keeping all of avg/min/max/range
  together only adds collinearity without new information (see e.g. James,
  Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*, on
  the cost of collinear predictors).
- `wind_speed_max`, `wind_direction_at_max_speed_(sin/cos)` — narrower,
  highly correlated restatements of `wind_speed_avg`/`wind_direction_avg`;
  dropped to avoid doubling the wind feature count for little marginal
  signal.
- `rainfall` (raw) — superseded by `log1p_rainfall` for the same
  skew/zero-inflation reason cyclical encoding beats raw calendar ints.
- `hotspot_count` (raw) — superseded by `log1p_hotspot_count`.
- `sunshine_duration` — checked and dropped; see the note under "Final
  feature set" above (100% missing for 15% of stations, not just scattered
  gaps).
- `hotspot_frp_mean`, `hotspot_frp_max`, `hotspot_bright_ti4_max`,
  `hotspot_bright_ti5_max`, `hotspot_min_distance_km` — all derived from
  the same underlying fire detections as `hotspot_frp_sum` and ~75%
  structurally missing (undefined on no-fire days), so they add both
  redundancy and a heavy missing-data burden that `hotspot_present` +
  `log1p_hotspot_count` + `log1p_hotspot_frp_sum` already cover more
  cleanly (those three have 0 missing values in the schema).
- `hotspot_day`, `hotspot_night`, `hotspot_day_ratio`, `hotspot_night_ratio`
  — a finer day/night split of the same counts; dropped to keep the fire
  feature block compact (3 features), especially since the ratio columns
  inherit the same ~75% structural missingness as the FRP/brightness group.
- All `*_lag_*`, `*_roll_*`, `*_diff_*` engineered columns — deliberately
  excluded, matching the manifest's own `TFT_OBSERVED_FEATURES` view, which
  already leaves these out (they belong to `XGBOOST_FEATURES` instead).
  This is the correct call for a TFT specifically: the LSTM encoder +
  interpretable attention block is designed to learn lag/rolling/trend
  structure directly from the raw `input_chunk_length` window (Lim et al.
  2019, §4.3–4.4); hand-feeding pre-computed lags/rolling stats on top would
  just duplicate what the sequence-to-sequence layer already extracts, and
  works against the model's interpretability story (variable-importance
  weights get split across `pm25` and five different `pm25_lag_*`/
  `pm25_roll_*` restatements of it instead of reading cleanly).
- `pm25` itself is not listed as a past covariate: in Darts, the target
  series is already fed to the encoder automatically, so duplicating it into
  `past_covariates` would be redundant.
- All QC / imputation / mapping-metadata columns (`qc_*`, `*_imputed`,
  `*_interpolated`, `*_missing_original`, `*_structural_missing`,
  `mapping_*`, `match_method`, `weather_duplicate_source`,
  `pm25_source_rows`, `pm25_duplicate_source`, `station_lat`/`station_lon`/
  `amphoe`/`province`/`tambon`, `segment_id`, `history_days_available`,
  `*_history_valid`, `missing_feature_count`, `imputed_feature_count`) —
  these describe how the pipeline built the row, not the physical state of
  the atmosphere; they are pipeline bookkeeping, not predictors.
  `segment_id` specifically is used only for grouping into gap-free series
  in `src/data.py`, never as a model input.
- `target_pm25_t1`/`t2`/`t3`/`t7` — precomputed for a tabular (e.g. XGBoost)
  setup. Darts' `TFTModel` generates every horizon step internally from the
  `pm25` target series via `output_chunk_length`, so these columns are
  unused; see the "Forecast horizon" decision below.

### Corrections to the manifest's TFT views

`feature_manifest.yaml`'s `views.TFT_KNOWN_TIME_FEATURES` lists
`wind_direction_avg_sin/cos` and `wind_direction_at_max_speed_sin/cos` as
known-future — but wind direction is a measured weather quantity with no
forecast feed in this dataset, so it cannot honestly be "known" ahead of
time (only the calendar-derived cyclical features can). Confirmed with the
user (2026-09-12): move the wind-direction cyclical features to the
observed/past-only group instead of following the manifest as-is. This
keeps `future_covariates` honest, which matters mechanically, not just
stylistically — `TFTModel`'s decoder is required to have these values for
the full forecast horizon at prediction time, which would be impossible for
a genuinely unknown quantity like wind direction.

**Update (2026-09-12)**: the source
`clean-data_preprocess_all_stations_daily_feature_manifest.yaml` itself has
now been corrected to match — `wind_direction_avg_sin/cos` and
`wind_direction_at_max_speed_sin/cos` moved from `views.TFT_KNOWN_TIME_FEATURES`
to `views.TFT_OBSERVED_FEATURES` in the manifest file, so this is no longer
a manifest-vs-model discrepancy to work around, just documented history of
why the split looks the way it does. `views.XGBOOST_FEATURES` was left
untouched (it's a flat tabular feature list with no known/observed
distinction, so the bug never applied there).

## Forecast horizon

Confirmed with the user (2026-09-12): one multi-step model with
`output_chunk_length=7`, predicting days 1–7 ahead in a single forward pass,
rather than four separate single-horizon models. This is also what the TFT
architecture is built for — direct multi-horizon forecasting from one
model (Lim et al. 2019, abstract/§1) — and it is why the precomputed
`target_pm25_t1/t2/t3/t7` columns aren't used: Darts' `TFTModel` slices
1..`output_chunk_length` labels out of the `pm25` series itself.

## Target availability: stations without a PM2.5 sensor

Not every `station_id` in this dataset actually measures PM2.5. A number of
them are pure meteorological stations — their names say so directly (e.g.
*"Narathiwat Weather Observing Station"*, *"Khirithan Dam Automatic Weather
Station"*) — and their `pm25` column is 90–100% `NaN` for their entire
history, not a handful of short measurement gaps. Checked directly against
the CSV: of the 576 `(station_id, segment_id)` groups with enough history to
form at least one training window, 39 have `pm25` missing for more than 30%
of their rows (almost all of those are ≥90% missing — there is a clean gap
in the distribution between ~5% and ~90%, no group sits in between).
`src/data.py` drops any group whose `pm25` exceeds `--max-target-nan-frac`
(default `0.3`) before it ever becomes a training target — these
weather-only stations are still free to appear as neighbors of real PM2.5
stations, but they cannot be forecast targets themselves. The remaining
~537 real PM2.5-bearing groups have very low residual missingness (median
0.3%), which `MissingValuesFiller` (linear interpolation, both directions)
comfortably covers.

## References

- Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2019/2021). *Temporal
  Fusion Transformers for Interpretable Multi-horizon Time Series
  Forecasting.* https://arxiv.org/abs/1912.09363
- "The relationships between PM2.5 and meteorological factors in China:
  seasonal and regional variations." https://arxiv.org/pdf/1708.06072 /
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5750928/
- "Assessment of Transboundary PM2.5 from Biomass Burning in Northern
  Thailand Using the WRF-Chem Model."
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11280843/
- "Near Real-Time Biomass Burning PM2.5 Emission Estimation to Support
  Environmental Health Risk Management in Northern Thailand Using
  FINNv2.5." https://pmc.ncbi.nlm.nih.gov/articles/PMC12846012/
- feature-engine `CyclicalFeatures` (cyclical encoding of calendar
  variables):
  https://feature-engine.trainindata.com/en/1.7.x/user_guide/creation/CyclicalFeatures.html
- James, G., Witten, D., Hastie, T., & Tibshirani, R. *An Introduction to
  Statistical Learning* — collinear-predictor rationale for dropping
  redundant min/max/avg/range combinations.
- `clean-data_preprocess_all_stations_daily_feature_manifest.yaml` and
  `clean-data_preprocess_all_stations_daily_preprocessing_schema.json` in
  the repo root — source of the raw column list, null counts, and the
  `TFT_*` views this selection starts from.
