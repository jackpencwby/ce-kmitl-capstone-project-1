# Experiment Stage 2 - LightGBM tuning

XGBoost global + local tree experiments **13/14** use
[`run_xgb_residual.py`](run_xgb_residual.py), with XGBoost at both levels.
See [the XGBoost residual guide](XGB_RESIDUAL_README.md) for architecture,
Tawan-based tuning, CUDA requirements, and commands. The runner documented
below uses Jack's LightGBM runner for 1/2/3/4/15/16, with CPU/OpenCL GPU selection.

Stage 2 implements these experiment rows:

| ID | Scope | Forecast strategy |
|---|---|---|
| 1 | Global | Single-output / direct: one LightGBM estimator for each t+1...t+7 target |
| 2 | Global | Multi-output: one forecaster accepts seven targets and returns t+1...t+7 together |
| 3 | Regional | Single-output / direct |
| 4 | Regional | Multi-output |
| 15 | Global + per-station tree residual correction | Single-output / direct |
| 16 | Global + per-station tree residual correction | Multi-output |

The shared Stage 1 code remains the source of truth for feature engineering,
target construction, station eligibility, regional mapping, time splits,
metrics, and artifact formats. Stage 2 searches the supplied LightGBM grid with a
deterministic **coarse-to-fine hierarchical grid search** (no random sampling): the
full Cartesian product has 16,200,000 candidates, so it is walked in three phases
(structure -> regularization -> local refinement) with successive-halving pruning.
`n_estimators` is taken from the grid as a boosting budget (300/800/2000) and
validation early stopping chooses the actual tree count for 1–4. Experiments
15/16 select tree counts on the last 15% of each fit's training rows, purging
seven days from the fit/stop boundary. They refit on all training rows and reuse
validation-selected counts for test (including OOF fits). Patience defaults to
100 via `--early-stopping-rounds`; small splits (<50 fit or <10 stopping rows)
use the full budget. Existing running processes must be restarted to use this change. See
`jack_stage2_explained.md` for the full description.

For experiments 2/4/16 ("multiple prediction head") there is **one LightGBM model**
per scope. LightGBM has no native vector-leaf trees, so each sample is stacked into
seven rows with a `forecast_horizon` feature and a single `LGBMRegressor` is fitted
once; predictions are reshaped back to t+1...t+7. `MultiOutputRegressor` is not used.

Experiments 15/16 follow Stage 1 `E1_3.py`: fit global models, generate
walk-forward out-of-fold (OOF) predictions within the training period, then
fit a separate LightGBM residual corrector per station. Final predictions
are `global_prediction + local_residual_prediction`. Here, "finetune"
means residual correction, rather than continued boosting of a global model.
Experiment 15 reuses Stage 1's direct residual runner; experiment 16 fits
one horizon-stacked multi-head model at each level (global and per-station). It requires complete seven-target rows, like experiment 2.

OOF folds use the Stage 1 180-day warmup and embargo. Training target dates
must precede the next evaluation origin; OOF residual labels must precede
the holdout boundary. Stations with fewer than 120 usable OOF rows fall back
to global predictions. Both levels use the same trial's LightGBM parameters;
validation selects the combined predictor. Test refitting rebuilds global,
OOF, and local models using train+validation only. OOF fold boundaries are
saved with the artifacts. These runs cost more than global-only experiments
because every trial also trains OOF and per-station models.

## Device selection

Use `--device gpu` for GPU acceleration through OpenCL on Windows, or
`--device cpu` for CPU. The chosen device applies to global, OOF, and local
residual models. It is included in run IDs and artifacts. Before loading data,
`jack_run_stage2.py` probes scalar and horizon-stacked fits on that device;
failure stops the run instead of silently falling back. GPU requires a
GPU-enabled LightGBM build; see `jack_stage2_explained.md` for details.

```powershell
.venv\Scripts\python.exe experiment-stage-2\jack_run_stage2.py --experiment 1
```

## Run

Use `--source gcs` when the ignored local master CSV is unavailable. Start
with a small station cohort and a low trial count to verify the environment:

```powershell
.venv\Scripts\python.exe experiment-stage-2\jack_run_stage2.py --experiment 1 --train --source gcs --max-stations 10 --development
```

Then run the full experiment one process at a time. CPU LightGBM uses all
available logical CPU cores by default:

```powershell
.venv\Scripts\python.exe experiment-stage-2\jack_E2_1_global_single_head.py --train --source gcs
.venv\Scripts\python.exe experiment-stage-2\jack_E2_2_global_multi_head.py --train --source gcs
.venv\Scripts\python.exe experiment-stage-2\jack_E2_3_regional_single_head.py --train --source gcs
.venv\Scripts\python.exe experiment-stage-2\jack_E2_4_regional_multi_head.py --train --source gcs
.venv\Scripts\python.exe experiment-stage-2\E2_15_global_local_tree_single_head.py --device gpu --gpu-platform-id 1 --gpu-device-id 0 --train --source gcs
.venv\Scripts\python.exe experiment-stage-2\E2_16_global_local_tree_multi_head.py --device gpu --gpu-platform-id 1 --gpu-device-id 0 --train --source gcs
```

Tuning uses validation `primary_macro_rmse` only. The selected parameter set
is written to `best_params.json`, then refit on train+validation and evaluated
once on the locked test period. Use `--skip-test` only for development runs.

Artifacts are written to `experiment-stage-2/artifacts/<run-id>__<timestamp>/`:

- `tuning_trials.csv`, `phase{1,2,3}_results.csv`, `tuning_summary.json` - every evaluated configuration (phase, round, budget, per-horizon RMSE, elapsed time, failure)
- `best_params.json` - the validation winner
- `config.json` and `dataset_manifest.json` - reproducibility/provenance
- validation/test metrics, predictions, feature importance, and model manifests
