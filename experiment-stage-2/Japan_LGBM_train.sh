#!/usr/bin/env bash
# Submit from repo root: sbatch -J Japan_E2_15 this.sh 15 RUN15 RUN16
# RUN16 is optional; it is submitted only after E15 exports successfully.
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1g.23gb:1
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=experiment-stage-2/run_logs/Japan_LGBM-%j.log
#SBATCH --error=experiment-stage-2/run_logs/Japan_LGBM-%j.log
set -euo pipefail
cd "${JAPAN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
SCRIPT="$PWD/experiment-stage-2/Japan_LGBM_train.sh"
EXPERIMENT=${1:?Expected experiment 15 or 16}
RUN_DIR=$(realpath -m "${2:?Expected persistent run directory}")
NEXT_DIR=${3:-}
ATTEMPT=${4:-0}
[[ "$EXPERIMENT" == 15 || "$EXPERIMENT" == 16 ]] || exit 2
[[ "$ATTEMPT" =~ ^[0-9]+$ ]] || exit 2
if [[ -n "$NEXT_DIR" ]]; then NEXT_DIR=$(realpath -m "$NEXT_DIR"); fi
export VIRTUAL_ENV="$PWD/.venv-japan"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
PYTHON=${JAPAN_PYTHON:-$VIRTUAL_ENV/bin/python}
ARGS=(--experiment "$EXPERIMENT" --train --source local --device gpu --checkpoint --run-dir "$RUN_DIR")
if [[ -f "$RUN_DIR/checkpoint/manifest.json" ]]; then ARGS+=(--resume "$RUN_DIR"); fi
echo "Experiment=$EXPERIMENT Job=${SLURM_JOB_ID:-manual} Attempt=$ATTEMPT Run=$RUN_DIR"
child=''
interrupted=0
cancelled=0
continuation=''
queue_resume() {
    [[ -n "$continuation" ]] && return 0
    if [[ "$ATTEMPT" -ge 9 || ! -f "$RUN_DIR/checkpoint/manifest.json" ]]; then return 1; fi
    continuation=$(sbatch --parsable --dependency="afterany:${SLURM_JOB_ID:?}" --job-name="Japan_E2_$EXPERIMENT" \
        "$SCRIPT" "$EXPERIMENT" "$RUN_DIR" "$NEXT_DIR" "$((ATTEMPT + 1))") || return 1
    continuation=${continuation%%;*}
    [[ "$continuation" =~ ^[0-9]+$ ]] || return 1
    echo "Queued continuation job $continuation from the same checkpoint."
}
request_resume() {
    interrupted=1
    # Queue before waiting for a native fit: it may outlast the warning period.
    queue_resume || echo 'Could not queue continuation; manual resume may be required.' >&2
    if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi
}
trap request_resume USR1
trap 'cancelled=1; interrupted=1; if [[ -n "$child" ]]; then kill -TERM "$child" 2>/dev/null || true; fi' TERM
"$PYTHON" -u experiment-stage-2/jack_run_stage2.py "${ARGS[@]}" &
child=$!
set +e
while true; do
    wait "$child"
    status=$?
    # A trapped signal interrupts wait even when the child has already exited.
    # Reap it again to obtain Python's exit status, not Bash's 128+signal status.
    if [[ "$interrupted" == 1 && "$status" -gt 128 ]]; then interrupted=0; continue; fi
    break
done
set -e
trap - USR1 TERM
if [[ "$cancelled" == 1 ]]; then exit 143; fi
if [[ "$status" == 75 ]]; then
    if ! queue_resume; then
        echo 'Continuation stopped: retry limit reached or no checkpoint manifest.' >&2
        exit 75
    fi
    exit 0
fi
if [[ "$status" != 0 ]]; then
    if [[ -n "$continuation" && "$status" != 137 && "$status" != 143 ]]; then scancel "$continuation"; fi
    exit "$status"
fi
# If the warning arrived during final exports, the queued invocation validates
# the completed run and schedules E16. Do not submit E16 twice.
if [[ -n "$continuation" ]]; then exit 0; fi
if [[ ! -f "$RUN_DIR/checkpoint/completed.json" ]]; then
    echo 'Python exited without a completed export marker; not scheduling the next experiment.' >&2
    exit 1
fi
if [[ "$EXPERIMENT" == 15 && -n "$NEXT_DIR" ]]; then
    sbatch --dependency="afterok:${SLURM_JOB_ID:?}" --job-name=Japan_E2_16 \
        "$SCRIPT" 16 "$NEXT_DIR" '' 0
    echo 'E15 complete; queued E16.'
fi
