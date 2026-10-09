# Experiment Stage 2 - CPU LightGBM tuning

Stage 2 implements the four rows in `experiment-stage-2 - Sheet1.csv`:

| ID | Scope | Forecast strategy |
|---|---|---|
| 1 | Global | Single-output / direct: one LightGBM estimator for each t+1...t+7 target |
| 2 | Global | Multi-output: one forecaster accepts seven targets and returns t+1...t+7 together |
| 3 | Regional | Single-output / direct |
| 4 | Regional | Multi-output |

The shared Stage 1 code remains the source of truth for feature engineering,
target construction, station eligibility, regional mapping, time splits,
metrics, and artifact formats. Stage 2 adds a reproducible random search over
the supplied LightGBM grid. It samples unique combinations; the complete grid
contains 16,200,000 candidates, so exhaustive search is deliberately not the
default.

For E2/E4, the forecaster is `MultiOutputRegressor(LGBMRegressor)`: callers fit
one object with a seven-column target matrix and receive a seven-column
prediction matrix in one call. LightGBM does not natively fit a vector-valued
target, so this sklearn wrapper owns one LightGBM estimator per output under
the hood; the estimators do not share tree parameters.

## CPU contract

Stage 2 runs with `device_type="cpu"`, so it works with the standard Windows
LightGBM package. Before loading data, `run_stage2.py` performs a tiny CPU
LightGBM fit and stops only if LightGBM itself is not usable:

```powershell
.venv\Scripts\python.exe experiment-stage-2\run_stage2.py --experiment 1
```

## Run

Use `--source gcs` when the ignored local master CSV is unavailable. Start
with a small station cohort and a low trial count to verify the environment:

```powershell
.venv\Scripts\python.exe experiment-stage-2\run_stage2.py --experiment 1 --train --source gcs --max-stations 10 --n-trials 3
```

Then run the full experiment one process at a time. CPU LightGBM uses all
available logical CPU cores by default:

```powershell
.venv\Scripts\python.exe experiment-stage-2\E2_1_global_single_head.py --train --source gcs --n-trials 40
.venv\Scripts\python.exe experiment-stage-2\E2_2_global_multi_head.py --train --source gcs --n-trials 40
.venv\Scripts\python.exe experiment-stage-2\E2_3_regional_single_head.py --train --source gcs --n-trials 40
.venv\Scripts\python.exe experiment-stage-2\E2_4_regional_multi_head.py --train --source gcs --n-trials 40
```

Tuning uses validation `primary_macro_rmse` only. The selected parameter set
is written to `best_params.json`, then refit on train+validation and evaluated
once on the locked test period. Use `--skip-test` only for development runs.

Artifacts are written to `experiment-stage-2/artifacts/<run-id>__<timestamp>/`:

- `tuning_trials.csv` - every sampled configuration, score, elapsed time, and failure
- `best_params.json` - the validation winner
- `config.json` and `dataset_manifest.json` - reproducibility/provenance
- validation/test metrics, predictions, feature importance, and model manifests
