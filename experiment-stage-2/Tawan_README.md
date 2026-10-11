# Experiment Stage 2 — XGBoost forecasting

Stage 2 implements Tawan's four experiments and reuses Stage 1 for data loading,
causal features, target creation, eligibility, region assignment, chronological
splits, metric definitions, and artifact writing.

| Experiment | Scope | Forecast architecture |
|---|---|---|
| E2_1 (`--experiment 1`) | Global | Seven independent scalar-output XGBoost models |
| E2_2 (`--experiment 2`) | Global | One native seven-target XGBoost model |
| E2_7 (`--experiment 7`) | Regional | Seven independent scalar-output models per region |
| E2_8 (`--experiment 8`) | Regional | One native seven-target model per region |

## Forecast heads

### Single prediction head

Each scope trains seven separate XGBoost estimators, one for each target
`t+1` through `t+7`. For a tuning candidate, all seven estimators share the
same hyperparameters; each estimator can select its own best iteration from
validation early stopping. Both architectures use complete seven-target rows
for training and scoring, matching the same station/date/horizon cohort.

### Multiple prediction head

Each scope trains **one** native XGBoost vector-leaf model with
`multi_strategy="multi_output_tree"`. Its target matrix has shape
`(n_samples, 7)` and columns ordered `t+1` through `t+7`. The saved model is
reloaded during preflight and checked for a `(n_samples, 7)` prediction shape.

`MultiOutputRegressor` is not used. E2_2 and E2_8 do not fit seven independent
target-specific XGBoost models.

## Hardware and early stopping

Every training estimator uses `tree_method="hist"` and `device="cuda:0"`.
The runner performs scalar CUDA and, where requested, native multi-output CUDA
smoke fits before it loads the project dataset. It exits with XGBoost version,
device, exception, and available GPU diagnostics if preflight fails; there is
no CPU fallback.

The installed XGBoost 3.x API accepts `early_stopping_rounds` in the estimator
constructor. The runner uses validation RMSE with a maximum budget of 3,000
boosting rounds and defaults to 100 early-stopping rounds. Each trial records
its best iteration and best score. Final models are refit on Train + Validation
using the selected iteration counts, then the locked Test split is evaluated
once.

## Coarse-to-Fine Hierarchical Grid Search

All selection uses Stage 1's validation `primary_macro_rmse`; lower is better.
The candidate ordering and tie-breaking are deterministic, and `--seed`
controls XGBoost stochastic sampling.

1. **Phase 1 — structural search:** 108 configurations across learning rate,
   depth, child weight, row sampling, and column sampling. Successive-halving
   rounds use budgets 600, 1,400, and 3,000, keeping the best third each time
   (108 → 36 → 12 when no candidate fails).
2. **Phase 2 — regularization:** with the winning structure fixed, evaluate
   the 180 combinations of gamma, L1, and L2 regularization at budget 800;
   carry the best 36 to budget 3,000.
3. **Phase 3 — fine local search:** generate neighboring structural values
   around the Phase 2 winner from the original supplied parameter domain. Up
   to 243 candidates are evaluated at budget 800; the best 36 continue to
   budget 3,000. Regularization remains fixed to the Phase 2 winner.

Regional experiments select one shared parameter configuration across all
regions using the Stage 1 `primary_macro_rmse`; they do not tune a different
configuration for each region. The winning configuration is then fitted as
seven scalar models per region in E2_7, or one native vector model per region
in E2_8.

At the maximum grid sizes this is 651 candidate evaluations per experiment
(156 Phase 1 + 216 Phase 2 + up to 279 Phase 3). Tuning estimator fits depend
on architecture and number of regions:

| Experiment | Fits per candidate | Tuning fits at maximum grid size |
|---|---:|---:|
| E2_1 | 7 | 4,557 |
| E2_2 | 1 | 651 |
| E2_7 | 7 per region | 18,228 for four regions |
| E2_8 | 1 per region | 2,604 for four regions |

These counts exclude the validation-selected refit and final Train + Validation
refit. Actual counts are recorded in each run's `tuning_summary.json` and
`runtime.json`. Development candidate limits reduce the work and are labeled
as development artifacts.

## Commands

Run a hardware/architecture preflight without loading the data:

```powershell
.venv\Scripts\python.exe experiment-stage-2\Tawan_run_stage2.py --experiment 2
```

Run a small development tune. This uses the first three deterministic
candidates per phase and only one region or a small station subset; its output
is marked `DEV_` and should not be reported as a final experiment:

```powershell
.venv\Scripts\python.exe experiment-stage-2\Tawan_run_stage2.py `
  --experiment 2 --train --source local --max-stations 10 --development --skip-test
```

Run the four full experiments one at a time:

```powershell
.venv\Scripts\python.exe experiment-stage-2\Tawan_E2_1_global_single_head.py --train --source local
.venv\Scripts\python.exe experiment-stage-2\Tawan_E2_2_global_multi_head.py --train --source local
.venv\Scripts\python.exe experiment-stage-2\Tawan_E2_7_regional_single_head.py --train --source local
.venv\Scripts\python.exe experiment-stage-2\Tawan_E2_8_regional_multi_head.py --train --source local
```

Use `--source gcs` if the ignored local master table is unavailable. Optional
search controls include `--max-phase1-candidates`, `--max-phase2-candidates`,
`--max-phase3-candidates`, `--no-phase1-pruning`, and
`--early-stopping-rounds`. Candidate caps and `--development` produce
development runs.

## Artifacts

Each run is written under `experiment-stage-2/artifacts/<run-id>__<timestamp>/`.
It contains Stage 1 style metrics and predictions, `config.json`,
`dataset_manifest.json`, phase result CSVs, `tuning_trials.csv`,
`tuning_summary.json`, `best_params.json`, `runtime.json`, a model manifest,
feature importance/status, and named JSON model files under `models/`.
It also writes `validation_metrics_by_region.csv` and, when Test is enabled,
`test_metrics_by_region.csv`; Stage 1 prediction filenames are preserved and
also copied to `validation_predictions.*` / `test_predictions.*` aliases.

The test split is not used for tuning, pruning, early stopping, or model
selection. `--skip-test` leaves it unevaluated.
