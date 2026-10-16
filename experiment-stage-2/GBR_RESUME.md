# Gradient Boosting: six parallel experiments with resume

`GBR_parallel_train.sh` runs E5, E6, E11, E12, E17 and E18 as six separate
Python processes in one Slurm allocation. Each process is restricted to one
BLAS/OpenMP thread. The request is 6 CPUs, 192 GiB RAM, no GPU, five days,
account/QoS advanced, partition defq. No Jupyter kernel is needed.

From the repository root, after installing the requirements below:

```bash
mkdir -p experiment-stage-2/run_logs
sbatch experiment-stage-2/GBR_parallel_train.sh experiment-stage-2/artifacts/GBR_all6_RUN_NAME
```

Use the same command and run root to resume. Do not submit duplicate copies.
Each E*/ directory has its own identity manifest and single-writer lock. E*.log
contains progress. The launcher waits for all six workers and reports failure
if any fails. Each process loads only its own trusted local joblib checkpoints;
never resume an artifact supplied by another user. SHA256 detects corruption,
not malicious modification of both a file and its checksum.

The manifest validates data, source, library versions and scientific arguments.
Each completed model commits atomically with a checksum marker. The unfinished
model is repeated after interruption; this is model-level, not tree-level resume.
Completed candidates retain their scores, and their redundant models are removed.
Final validation/test models are kept. Completed runs verify exported checksums
before skipping. Runtime reports retain existing semantics and are not a measure
of total compute consumed across interrupted allocations.

Five minutes before expiry the launcher queues an afterany successor, then asks
workers to stop at the next safe fit boundary. A fit that cannot finish before
the hard timeout will be repeated. At most ten allocations are used. The successor
checks the predecessor's state and trains only after COMPLETED or TIMEOUT; it
refuses cancelled/failed predecessors. Failure to inspect that state stops safely.
Cancel all active/pending jobs with `scancel --user="$USER" --name=GBR_all6`.
Ordinary training/checkpoint failures stop automatic continuation and need review.

Environment:

```bash
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
python3 -m venv .venv-gbr
.venv-gbr/bin/python -m pip install -r experiment-stage-2/requirements-gbr.txt
```

The launcher also clears inherited Jupyter Python paths. Requirements pin the
core versions used in verification. Keep this environment and the source fixed
while training.
