# Stage 2 experiments 13/14: XGBoost global + local tree correction

Both the global predictor and each station's residual corrector are XGBoost,
matching Stage 1 `E1_3.py` / `common.models._run_residual` and `_tree_corrector`.
This is additive residual correction, not continuation of global trees.

| ID | Global model | Local model per station |
|---|---|---|
| 13 | Seven independent scalar XGBoost heads | Seven scalar XGBoost residual heads |
| 14 | One native `multi_output_tree` XGBoost model | One native seven-output residual tree model |

Experiment 14 follows Tawan's native vector-leaf architecture; it does not
use `MultiOutputRegressor`. Both experiments use complete seven-target rows
and the same Stage 1 features, metrics, station eligibility and date splits
as Tawan's Stage 2 experiments.

## Training and selection

1. Fit the global predictor on training rows.
2. Generate expanding-window OOF global predictions inside training only,
   using Stage 1's 180-day warmup and embargo. Purge seven-day target windows
   before every fold origin and the outer holdout boundary.
3. Fit station models to `actual - global_OOF_prediction`.
4. Predict `global_prediction + station_residual_prediction`. Stations with
   fewer than 120 OOF rows use the global prediction alone.

The XGBoost factory, parameter domains, seed and three-phase
hierarchical search come directly from `Tawan_run_stage2.py`. Both levels use
the same candidate parameters, and candidates are scored on the **combined**
validation predictions. The full search can evaluate up to 651 candidates;
each additionally fits OOF and station models, so start with a development run.

Unlike Tawan's global-only runner, early stopping uses the chronological final
15% of each model's **training rows**, with a seven-day purge. With at least
50 fitting and 10 stopping rows, select rounds on that tail then refit all
supplied training rows. Smaller samples use the candidate budget. This keeps
OOF and outer holdout labels out of early stopping. Validation still selects
hyperparameters by primary macro RMSE.

Early stopping is enabled by default with patience 100 (`--early-stopping-rounds 100`)
for global, all four OOF folds, and local models in experiments 13 and 14.
Each OOF fold now saves its own selected tree count under `oof:<fold>:<horizon>`;
test fitting reuses that count instead of the global model's count. Older
round-count dictionaries without OOF entries fall back to global counts with
a warning. Changes apply to newly started runs; running processes are unchanged.

Final test fitting rebuilds global, OOF and local models on Train+Validation
with the selected round counts and no test-based early stopping. A station
that newly qualifies for local correction uses its global head's selected
round count if no previous local round count exists. `--skip-test` saves the
validation models only. Experiments 13/14 support opt-in checkpoint/resume as described below.

### Checkpoint / resume (E13 and E14)

Add `--checkpoint --train` to save completed global, OOF, and station fits.
Resume with the **same training arguments** plus `--resume <artifact-directory>`.
The directory is printed as `Output:` / `Created checkpoint:` in the log.
An old run created without checkpoints cannot be resumed.

```bash
# On the KMITL server, submit from the repository root:
mkdir -p experiment-stage-2/run_logs
sbatch experiment-stage-2/Japan_E2_13_train.sh
# After a timeout, interruption, or cancellation, use its original artifact directory:
sbatch experiment-stage-2/Japan_E2_13_train.sh --resume experiment-stage-2/artifacts/<run-directory>
```

The script uses `.venv-japan/bin/python`, the same interpreter configured for
Japan-kernel, and requests one 23 GB MIG device, 4 CPUs, 32 GB RAM, and 24 hours.
Install `experiment-stage-2/requirements.txt` in that environment first.
`sbatch` runs independently of any notebook kernel. It does not automatically
resubmit; rerun the resume command when the previous job has stopped.

Checkpoints commit native XGBoost models with checksums and atomic JSON markers.
A restarted search replays its deterministic candidate ordering, loads completed
scores, and skips already committed fits in the interrupted candidate. Once a
tuning candidate finishes, its intermediate models are removed to limit disk use;
final validation/test models remain available until export finishes. Validation
and test refits use separate checkpoint scopes. Training resumes at **completed
model boundaries**, not at an individual boosting iteration. The in-progress model
(including its early-stopping probe) is repeated after an interruption.

A manifest checks data contents, station selection, arguments, Python/package
versions, and source hashes. Changed runs are rejected instead of mixing results.
Only one writer can hold a run directory. Logs contain `Checkpoint fit`, `Resume
fit`, and `Resume: completed candidate`. Runtime and fit counters describe the
current invocation; they are not cumulative across restarts. Do not edit source
files or upgrade the environment until the run finishes.

Slurm sends USR1 two minutes before timeout. The runner exits with status 75 and
keeps all committed checkpoints; even a hard kill can only discard an uncommitted
fit. Reuse the original artifact directory rather than submitting a fresh run.

## Run

Both experiments use Tawan's `tree_method="hist"`. Experiment 13 defaults to
`device="cuda:0"`; experiment 14 defaults to `device="cpu"` because the locally
installed XGBoost 3.0.5 rejects GPU vector-leaf training. This retains native
multi-output trees at both levels. Device is included in run IDs and config.
Use `--device cpu` or `--device cuda:0` to explicitly override it for 13/14.
The chosen device is probed before loading data; a failing GPU probe stops
with an error, without silently falling back. Compare timing by device.

```powershell
# Preflight only
.venv\Scripts\python.exe experiment-stage-2\E2_13_global_local_tree_single_head.py
.venv\Scripts\python.exe experiment-stage-2\E2_14_global_local_tree_multi_head.py

# Small development run (validation only)
.venv\Scripts\python.exe experiment-stage-2\run_xgb_residual.py --experiment 13 --train --development --max-stations 2

# Full runs, one at a time
.venv\Scripts\python.exe experiment-stage-2\E2_13_global_local_tree_single_head.py --train
.venv\Scripts\python.exe experiment-stage-2\E2_14_global_local_tree_multi_head.py --train
```

Use `--source gcs` if needed. Candidate limits and `--seed` use Tawan's CLI.
Artifacts are saved under `experiment-stage-2/artifacts/S2_13__...` or
`S2_14__...` (prefixed `DEV_` for `--development`), including phase/trial logs,
best parameters and round counts, OOF fold manifest, validation/test metrics,
predictions, configuration, and both global/local models with distinct roles.
Saved XGBoost models use JSON. Native vector-tree feature importance is
omitted when the installed XGBoost implementation does not expose it.
