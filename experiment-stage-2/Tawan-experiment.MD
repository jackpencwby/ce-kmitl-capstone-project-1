You are working on an existing PM2.5 forecasting research repository.

Your task is to implement **Experiment Stage 2 for the experiments owned by Tawan**.

Before writing or modifying code, inspect the repository carefully, especially all Stage 1 experiment code, configs, preprocessing utilities, dataset loaders, evaluation functions, regional mappings, artifact formats, and README files.

Stage 1 is the **source of truth**.

Do NOT independently redesign feature engineering, target construction, station filtering, time splitting, metrics, regional mapping, or artifact formats unless required to support Stage 2.

---

# 1. Stage 2 experiments to implement

According to the experiment plan, Tawan is responsible for four XGBoost experiments:

| ID | Scope | Model | Forecast strategy |
|---|---|---|---|
| 1 | Global / train all stations together | XGBoost | Single prediction head |
| 2 | Global / train all stations together | XGBoost | Multiple prediction head |
| 7 | Regional / train separately by region | XGBoost | Single prediction head |
| 8 | Regional / train separately by region | XGBoost | Multiple prediction head |

Implement these four experiments as Stage 2.

Suggested files:

```text
experiment-stage-2/
├── run_stage2.py
├── E2_1_global_single_head.py
├── E2_2_global_multi_head.py
├── E2_7_regional_single_head.py
├── E2_8_regional_multi_head.py
├── config/
├── utils/
└── artifacts/
```

Reuse Stage 1 modules instead of duplicating them wherever practical.

---

# 2. Critical definition: multiple prediction head

This requirement is extremely important.

## Single prediction head

Single-output/direct forecasting means:

```text
X -> XGBoost model 1 -> PM2.5 t+1
X -> XGBoost model 2 -> PM2.5 t+2
...
X -> XGBoost model 7 -> PM2.5 t+7
```

There must be **7 independent XGBoost models**, one per forecast horizon.

Each model receives the same feature matrix X but a different scalar target:

```text
y_t1
y_t2
...
y_t7
```

The seven models do not share trees.

---

## Multiple prediction head

For this project, multiple prediction head explicitly means:

> One XGBoost model must jointly learn all seven forecasting targets t+1 ... t+7.

Do NOT implement this using:

```python
MultiOutputRegressor(XGBRegressor(...))
```

Do NOT train seven independent estimators hidden behind a wrapper.

Do NOT use:

```python
multi_strategy="one_output_per_tree"
```

Use XGBoost native multi-target vector-leaf trees:

```python
XGBRegressor(
    tree_method="hist",
    device="cuda",
    multi_strategy="multi_output_tree",
    ...
)
```

The training target must be:

```python
Y.shape == (n_samples, 7)
```

with columns ordered:

```text
t+1
t+2
t+3
t+4
t+5
t+6
t+7
```

Conceptually:

```text
                     ┌── t+1
                     ├── t+2
                     ├── t+3
Features -> XGBoost ─├── t+4
 multi-output tree   ├── t+5
                     ├── t+6
                     └── t+7
```

This must be one jointly trained model whose vector-valued leaves produce seven outputs.

Before implementation, verify that the installed XGBoost version supports:

```python
multi_strategy="multi_output_tree"
```

Prefer a recent XGBoost version compatible with this feature.

Run a smoke test:

```python
X_small.shape == (small_n, n_features)
Y_small.shape == (small_n, 7)

model.fit(X_small, Y_small)
pred = model.predict(X_small)

assert pred.shape == Y_small.shape
```

If unsupported, stop with a clear error.

Do NOT silently fall back to independent models.

---

# 3. CUDA requirement

All XGBoost training in Stage 2 must use NVIDIA GPU acceleration.

Use:

```python
tree_method="hist"
device="cuda"
```

or:

```python
device="cuda:0"
```

Do NOT use CPU fallback.

Before loading the full dataset, perform a small CUDA XGBoost training test.

Example:

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=10,
    max_depth=2,
    tree_method="hist",
    device="cuda",
)

model.fit(X_small, y_small)
```

If CUDA training fails, terminate and report:

- XGBoost version
- device configuration
- exception message
- CUDA/GPU diagnostic information available safely

Do NOT silently continue on CPU.

Record device information in `config.json`.

---

# 4. Hyperparameter tuning strategy

Do NOT use exhaustive full-grid search.

Do NOT use pure random search as the primary tuning strategy.

Implement a:

# Coarse-to-Fine Hierarchical Grid Search

The purpose is to search the large XGBoost parameter space intelligently by tuning related parameter groups in stages.

The process must use validation data only.

The test set must remain locked until final model evaluation.

The overall pipeline is:

```text
TRAIN
   ↓
Phase 1: Structural / coarse search
   ↓
best structural region
   ↓
Phase 2: Regularization search
   ↓
best regularized configuration
   ↓
Phase 3: Fine local search
   ↓
final best hyperparameters
   ↓
refit on TRAIN + VALIDATION
   ↓
TEST exactly once
```

Use:

```text
primary_macro_rmse
```

as the primary optimization metric.

Lower is better.

Reuse Stage 1's exact implementation of this metric.

---

# 5. Original supplied XGBoost parameter space

The original Stage 2 search space is:

```python
XGB_PARAM_GRID = {
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
```

Do not evaluate the complete Cartesian product.

Use this as the allowed parameter domain from which the hierarchical search is derived.

---

# 6. Phase 1 — Coarse structural search

The first phase should identify a promising model-capacity and sampling region.

Search primarily:

```text
learning_rate
max_depth
min_child_weight
subsample
colsample_bytree
```

Use a coarse subset such as:

```python
PHASE1_GRID = {
    "learning_rate": [0.03, 0.1],
    "max_depth": [4, 6, 8],
    "min_child_weight": [1, 5, 20],
    "subsample": [0.7, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}
```

Use fixed baseline regularization during Phase 1:

```python
gamma = 0
reg_alpha = 0
reg_lambda = 1
```

Do not search every original value in Phase 1.

The objective is broad region discovery.

Record all evaluated configurations.

---

# 7. Phase 1 candidate pruning

To prevent unnecessary GPU training, support candidate pruning.

Prefer a simple deterministic approach.

For example:

```text
Round 1:
all Phase 1 candidates
resource ≈ low boosting budget
keep top 1/3

Round 2:
remaining candidates
larger boosting budget
keep top 1/3

Round 3:
remaining candidates
full or near-full budget
select top candidates
```

This is conceptually similar to successive halving.

However, do NOT force sklearn `HalvingGridSearchCV` if it conflicts with:

- custom time-series validation
- macro RMSE calculation
- multi-output XGBoost
- artifact requirements

A custom implementation is acceptable and may be preferable.

The ranking metric must always be validation `primary_macro_rmse`.

Make pruning deterministic.

---

# 8. Phase 2 — Regularization search

After Phase 1 identifies the best structural configuration, freeze or mostly freeze:

```text
learning_rate
max_depth
min_child_weight
subsample
colsample_bytree
```

Then tune:

```text
gamma
reg_alpha
reg_lambda
```

Use:

```python
PHASE2_GRID = {
    "gamma": [0, 0.01, 0.1, 0.5, 1],
    "reg_alpha": [0, 0.001, 0.01, 0.1, 1, 5],
    "reg_lambda": [0.1, 0.5, 1, 2, 5, 10],
}
```

If this full Phase 2 Cartesian product is too expensive, use a deterministic reduced grid or pruning strategy.

Do not switch to untracked random search.

Record exactly which combinations were tested.

---

# 9. Phase 3 — Fine local search

After Phase 2, perform a finer search around the current best configuration.

The local grid must be generated systematically from neighboring values in the original parameter domain.

Example:

If the current best is:

```text
learning_rate = 0.03
max_depth = 6
min_child_weight = 5
subsample = 0.8
colsample_bytree = 0.8
```

then the fine grid may become:

```python
{
    "learning_rate": [0.01, 0.03, 0.05],
    "max_depth": [5, 6, 8],
    "min_child_weight": [2, 5, 10],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}
```

Use only valid values from the original supplied parameter domain unless there is a strong technical reason otherwise.

The final Phase 3 winner becomes the Stage 2 best hyperparameter configuration.

---

# 10. n_estimators and early stopping

Do NOT treat `n_estimators` as a normal exhaustive grid-search dimension unless early stopping is technically unsupported.

Preferred approach:

```python
n_estimators = 3000
```

with validation-based early stopping.

Use:

```text
early_stopping_rounds = 100
```

or a comparable value.

The goal is to let each parameter configuration discover its own optimal boosting iteration.

Record:

```text
best_iteration
best_score
```

for every trial if available.

For example:

```text
learning_rate = 0.03
max_depth = 6

tree 1
tree 2
...
tree 820   <- best validation result
...
tree 920

no improvement for 100 rounds
=> stop
```

If the installed XGBoost native `multi_output_tree` does not support early stopping in the required configuration:

1. verify this explicitly
2. document the limitation
3. use a controlled boosting-budget strategy
4. do NOT silently change architectures

If early stopping cannot be used, then use a small deterministic `n_estimators` search such as:

```python
[500, 1200, 2000]
```

only after confirming the incompatibility.

---

# 11. Fair tuning between single-output and joint multi-output

The comparison:

```text
E2_1 vs E2_2
E2_7 vs E2_8
```

must be fair.

Use the same:

- source dataset
- station eligibility
- features
- targets
- time split
- metric implementation
- parameter domain
- Phase 1 strategy
- Phase 2 strategy
- Phase 3 strategy
- pruning logic
- seed
- test protocol

Do NOT force the same final hyperparameters.

Each architecture must independently select its own best hyperparameters using the same tuning protocol.

Example:

```text
E2_1 best:
max_depth=8
learning_rate=0.03
...

E2_2 best:
max_depth=6
learning_rate=0.05
...
```

This is valid.

The experiment compares the best tuned form of each forecasting strategy.

---

# 12. Hyperparameter search reproducibility

Add:

```text
--seed 42
```

Although this is grid-based rather than random search, the seed is still required for:

- XGBoost stochastic sampling
- any candidate ordering
- any tie-breaking
- any subsampling behavior

Candidate generation must be deterministic.

For a given:

```text
experiment
dataset
seed
configuration
```

the evaluated candidate sequence should be reproducible.

---

# 13. Tuning artifacts

Create:

```text
tuning_trials.csv
```

with at least:

```text
phase
round
trial_id
experiment_id
parameters
best_iteration
validation_primary_macro_rmse
elapsed_seconds
status
failure_reason
```

If practical include:

```text
rmse_t1
rmse_t2
rmse_t3
rmse_t4
rmse_t5
rmse_t6
rmse_t7
```

Also save:

```text
phase1_results.csv
phase2_results.csv
phase3_results.csv
```

or equivalent structured artifacts.

Save final selection as:

```text
best_params.json
```

Example:

```json
{
  "selected_phase": 3,
  "primary_macro_rmse": 0.0,
  "best_iteration": 0,
  "params": {}
}
```

---

# 14. Hyperparameter search summary

Create an artifact:

```text
tuning_summary.json
```

containing:

```text
phase1_candidate_count
phase1_survivors
phase2_candidate_count
phase3_candidate_count
total_models_trained
best_validation_score
best_parameters
best_iteration
total_tuning_seconds
```

This is important for research reproducibility.

---

# 15. Global experiment E2_1

Use all eligible stations together.

Train seven independent models:

```text
model_t1
model_t2
...
model_t7
```

For each tuning configuration, apply the same hyperparameter configuration across all seven horizon models.

Evaluate the resulting seven-horizon forecast together.

Do not tune each horizon independently because that would change the experimental comparison.

Use the complete seven-horizon macro validation metric for candidate ranking.

After tuning:

```text
best config
   ↓
combine TRAIN + VALIDATION
   ↓
train seven final models
   ↓
TEST
```

---

# 16. Global experiment E2_2

Train ONE native joint multi-output XGBoost model:

```python
XGBRegressor(
    ...,
    tree_method="hist",
    device="cuda",
    multi_strategy="multi_output_tree",
)
```

Use:

```python
Y_train.shape == (N, 7)
```

One `.fit()` must jointly learn all seven targets.

Prediction must return:

```python
pred.shape == (N, 7)
```

Do NOT use:

```text
MultiOutputRegressor
seven-model loops
one_output_per_tree
```

Apply the same hierarchical tuning framework used by E2_1.

---

# 17. Regional experiment E2_7

Use the Stage 1 region mapping.

For every region:

```text
region
  ├── model t+1
  ├── model t+2
  ├── ...
  └── model t+7
```

Use seven independent models per region.

For hyperparameter tuning, inspect Stage 1's regional methodology and choose the scientifically consistent strategy.

Preferred approach:

> one common hyperparameter configuration selected using macro performance across all regions

rather than independently tuning each region, because separate regional tuning may create a very large computational budget and make Global vs Regional comparisons less controlled.

Unless Stage 1 already specifies otherwise, evaluate each candidate across all regional models and rank using the project's regional macro validation metric.

Document the exact decision.

---

# 18. Regional experiment E2_8

For each region train one joint multi-target model:

```text
North   -> one native multi_output_tree -> t+1...t+7
Central -> one native multi_output_tree -> t+1...t+7
...
```

Each regional model receives:

```python
Y_region.shape == (N_region, 7)
```

Use:

```python
multi_strategy="multi_output_tree"
```

No `MultiOutputRegressor`.

Apply the same regional hyperparameter-selection strategy as E2_7.

---

# 19. Validation protocol

Use the exact Stage 1 chronological split.

Never shuffle temporal splits.

Protocol:

```text
TRAIN
  ↓
hyperparameter fitting

VALIDATION
  ↓
Phase 1 / 2 / 3 model selection

best configuration
  ↓
TRAIN + VALIDATION
  ↓
final refit

TEST
  ↓
final evaluation exactly once
```

The locked test set must never affect:

- parameter selection
- candidate pruning
- early stopping during tuning
- model selection
- feature selection

---

# 20. Data leakage audit

Verify:

- t+1...t+7 targets are aligned correctly
- lag features use past values only
- rolling features are causal
- future PM2.5 is not present in X
- preprocessing is fitted on training data only
- validation/test information does not influence feature construction improperly
- test is never used during tuning
- station eligibility does not improperly inspect future data

If a real Stage 1 leakage issue is discovered, report it before changing shared behavior.

---

# 21. Metrics

Reuse Stage 1 metric implementations.

Preserve:

```text
RMSE
MSE
MAE
R²
```

where applicable.

Report:

```text
overall
macro
per horizon
per station
per region
```

according to the existing Stage 1 artifact contract.

Always report:

```text
t+1
t+2
t+3
t+4
t+5
t+6
t+7
```

separately.

The primary tuning criterion remains:

```text
primary_macro_rmse
```

---

# 22. Runtime measurement

Measure:

```text
phase1_seconds
phase2_seconds
phase3_seconds
tuning_total_seconds
final_refit_seconds
test_prediction_seconds
total_run_seconds
```

For GPU operations, synchronize where necessary to avoid misleading asynchronous timings.

Also record total model fits executed.

This is especially important because one single-output configuration may require seven XGBoost fits while one joint-output configuration requires one fit.

---

# 23. Artifacts

Each run should create:

```text
experiment-stage-2/
└── artifacts/
    └── <experiment-id>__<timestamp>/
```

Save:

```text
config.json
dataset_manifest.json

phase1_results.csv
phase2_results.csv
phase3_results.csv
tuning_trials.csv
tuning_summary.json
best_params.json

validation_metrics.csv/json
test_metrics.csv/json

validation_predictions.*
test_predictions.*

runtime.json
model_manifest.json
feature_importance.*
```

Preserve Stage 1 formats where possible.

---

# 24. Model artifacts

Single-output:

```text
models/
├── horizon_1.json
├── horizon_2.json
...
└── horizon_7.json
```

Regional single-output:

```text
models/
├── region_A/
│   ├── horizon_1.json
│   └── ...
└── region_B/
```

Global joint multi-output:

```text
models/
└── multi_output.json
```

Regional joint multi-output:

```text
models/
├── region_A_multi_output.json
├── region_B_multi_output.json
└── ...
```

Verify saved multi-output models can be reloaded and still produce:

```python
pred.shape == (N, 7)
```

---

# 25. Feature importance

For single-output models preserve Stage 1 behavior.

For native vector-leaf `multi_output_tree`, inspect which importance functions are supported by the installed XGBoost version.

Do not invent unsupported per-target feature importance.

If unsupported:

- record that limitation explicitly
- continue the experiment
- save a status artifact

---

# 26. CLI

Provide:

```powershell
.venv\Scripts\python.exe experiment-stage-2\run_stage2.py --experiment 1
```

Map:

```text
1 -> Global single-output
2 -> Global joint multi-output
7 -> Regional single-output
8 -> Regional joint multi-output
```

Support:

```text
--experiment
--train
--source
--seed
--max-stations
--skip-test
```

Add tuning controls such as:

```text
--tuning-mode hierarchical-grid
--early-stopping-rounds 100
--phase1-pruning
```

Optional useful controls:

```text
--max-phase1-candidates
--max-phase2-candidates
--max-phase3-candidates
```

These should mainly support development/testing.

The scientific default should execute the complete defined hierarchical tuning protocol.

---

# 27. Development mode

Provide a lightweight smoke-test mode.

Example:

```powershell
.venv\Scripts\python.exe experiment-stage-2\run_stage2.py `
    --experiment 2 `
    --train `
    --source gcs `
    --max-stations 10 `
    --tuning-mode hierarchical-grid `
    --development
```

Development mode may use a very small candidate subset.

Clearly label outputs as development runs.

Do not mix development results with final experiment artifacts.

---

# 28. Preflight checks

Before expensive training print:

```text
[Stage 2 Preflight]

Python:
XGBoost:
CUDA device:
CUDA training: PASS/FAIL
multi_output_tree: PASS/FAIL

Dataset:
Eligible stations:
Features:
Targets: 7

Train:
Validation:
Test:

Experiment:
Scope:
Forecast strategy:
Tuning:
  hierarchical grid
  early stopping
Seed:
```

For E2_2 and E2_8 additionally verify:

```python
Y.ndim == 2
Y.shape[1] == 7

pred.ndim == 2
pred.shape[1] == 7
```

---

# 29. Verification tests

Add tests for:

## CUDA

```python
device == "cuda"
```

and actual GPU training success.

## Target ordering

```python
targets == [
    "t+1",
    "t+2",
    "t+3",
    "t+4",
    "t+5",
    "t+6",
    "t+7",
]
```

## Single-output

Exactly seven independently trained forecasting estimators.

## Joint multi-output

```python
model.get_params()["multi_strategy"] == "multi_output_tree"
```

and one XGBoost forecasting model per scope.

## Prediction shape

```python
pred.shape == (N, 7)
```

## Search reproducibility

Given the same best Phase 1 result, Phase 2 and Phase 3 candidate grids must be reproduced exactly.

## Test isolation

Ensure the test set is not accessed by the tuning module.

---

# 30. README

Create:

```text
experiment-stage-2/README.md
```

Explain:

## Single prediction head

```text
7 targets
-> 7 independent scalar-output XGBoost models
```

## Multiple prediction head

```text
7 targets
-> one XGBoost native multi-target model
-> multi_strategy="multi_output_tree"
-> vector-valued leaves
```

State explicitly:

```text
MultiOutputRegressor is NOT used.
```

Document the tuning methodology as:

# Coarse-to-Fine Hierarchical Grid Search

Explain:

```text
Phase 1
coarse structural search

Phase 2
regularization search

Phase 3
fine local search

then:
Train + Validation refit
Test once
```

Also document the use of early stopping and CUDA.

---

# 31. Methodology wording

The implementation should support a research-method description equivalent to:

> Hyperparameter optimization was performed using a coarse-to-fine hierarchical grid search. First, a coarse grid was used to identify promising regions of the structural hyperparameter space, including learning rate, tree depth, minimum child weight, row subsampling, and feature subsampling. The best-performing structural configuration was subsequently refined through a separate regularization search over gamma, L1 regularization, and L2 regularization. Finally, a localized fine-grid search was performed around the best configuration obtained from the previous stages. Candidate models were selected exclusively according to validation macro RMSE. Early stopping was used where supported to determine an appropriate number of boosting iterations. The held-out test set was evaluated only once after the final hyperparameter configuration was selected.

Keep implementation consistent with this methodology.

---

# 32. Implementation constraints

Do NOT:

- use exhaustive full Cartesian grid search
- use pure random search as the primary method
- use test data for tuning
- use `MultiOutputRegressor`
- use `one_output_per_tree` for E2_2/E2_8
- fall back to CPU
- silently change model architecture
- independently tune each horizon
- run full experiments automatically after coding

Do:

- use CUDA
- use native `multi_output_tree`
- use hierarchical coarse-to-fine grid tuning
- use early stopping where technically supported
- preserve time-series validation
- preserve Stage 1 preprocessing
- save all tuning history
- keep candidate selection deterministic
- fail clearly on unsupported functionality

---

# 33. Work order

Follow this order:

1. Inspect repository.
2. Identify Stage 1 source-of-truth modules.
3. Summarize Stage 1 pipeline.
4. Check installed XGBoost version.
5. Verify CUDA.
6. Verify native `multi_output_tree`.
7. Verify early-stopping compatibility for both scalar and multi-output configurations.
8. Implement shared hierarchical-grid tuning engine.
9. Implement Phase 1 coarse search.
10. Implement Phase 2 regularization search.
11. Implement Phase 3 fine local search.
12. Implement E2_1.
13. Implement E2_2.
14. Implement E2_7.
15. Implement E2_8.
16. Add artifacts and manifests.
17. Add verification tests.
18. Add README.
19. Run only small smoke/development tests.
20. Report implementation status.

Do NOT start final full runs automatically.

---

# 34. Final report

After implementation report:

1. Files created
2. Files modified
3. Stage 1 modules reused
4. XGBoost version
5. CUDA smoke-test result
6. `multi_output_tree` test result
7. Early-stopping compatibility result
8. Phase 1 candidate count
9. Phase 2 candidate count
10. Phase 3 candidate count
11. Total number of model fits in a full tuning run
12. How Phase 3 local grids are generated
13. How E2_1 differs from E2_2
14. How E2_7 differs from E2_8
15. Commands for development runs
16. Commands for full runs
17. Any XGBoost vector-leaf limitations
18. Whether all tests passed

Explicitly confirm:

```text
E2_2 and E2_8 use native XGBoost multi_strategy="multi_output_tree".

They do NOT use sklearn MultiOutputRegressor.

They do NOT train seven independent target-specific XGBoost models.

All XGBoost training uses CUDA.

Hyperparameter optimization uses Coarse-to-Fine Hierarchical Grid Search,
not pure Random Search and not exhaustive full-grid search.

The test set is never used for hyperparameter selection.
```