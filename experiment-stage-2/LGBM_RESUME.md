# E15/E16: resumable LightGBM training

Use the existing Japan environment. `--device gpu` selects LightGBM's OpenCL backend;
`--device cpu` is also supported. This does not enable the separate CUDA backend.

Start both experiments, in order, from the repository root:

```bash
stamp=$(date -u +%Y%m%d_%H%M%S)
sbatch --job-name=Japan_E2_15 experiment-stage-2/Japan_LGBM_train.sh \
  15 "experiment-stage-2/artifacts/S2_15_LGBM_GPU_$stamp" \
  "experiment-stage-2/artifacts/S2_16_LGBM_GPU_$stamp"
```

The launcher requests one 23 GB MIG GPU, four CPUs and 32 GB RAM for up to 24 hours.
Two minutes before expiry it queues a continuation with an `afterany` dependency
and forwards USR1 to Python. Python finishes and commits its current native fit
before exiting with code 75. If that fit outlasts the warning period, the queued
job repeats the incomplete fit. At most ten allocations are submitted per experiment.
Other errors stop the chain. Successful E15 completion queues E16 using an
`afterok` dependency on the finishing E15 allocation. E16 is not submitted until
E15's final export marker exists. Closing Jupyter does not cancel these jobs.

If submission fails or the continuation limit is reached, inspect the log and
submit the same command with the same directories. The launcher automatically
uses `--resume` when a manifest exists. Do not submit duplicate chains. Cancellation
of the chain must include both running and pending jobs, e.g. `scancel --user="$USER"
--name=Japan_E2_15,Japan_E2_16` (not USR1, which explicitly requests continuation).

For Python directly:

```bash
.venv-japan/bin/python experiment-stage-2/Japan_E2_15_global_local_tree_single_head.py \
  --train --source local --device gpu --checkpoint
# Supply all original scientific options when resuming:
.venv-japan/bin/python experiment-stage-2/Japan_E2_15_global_local_tree_single_head.py \
  --train --source local --device gpu --resume ARTIFACT_DIRECTORY
```

Each completed global, OOF, and station fit is stored as native LightGBM text
with a SHA256 completion marker written atomically. A partially completed fit is
repeated after interruption; this is model-level resume, not boosting-iteration
resume. Completed tuning candidates retain checked scores and round counts;
their redundant intermediate models are removed. Final validation and test fits
use separate checkpoints and can be exported again without training again.

The manifest checks data, station selection, arguments, source hashes and library
versions before reuse. Keep the code and environment unchanged during a run.
A file lock rejects concurrent writers. Corrupt committed checkpoints abort the
search rather than silently changing candidate rankings. Checkpoint loading uses
native LightGBM models rather than pickle; final artifacts retain the repository's
existing joblib export format, including a compatible prediction wrapper for
models restored from checkpoints.

`checkpoint/completed.json` is written only after exports finish. With
`--skip-test`, it explicitly records that the test split was skipped. Timing and
model-count summaries describe the current invocation and cached replay, not
cumulative compute across interrupted jobs.
