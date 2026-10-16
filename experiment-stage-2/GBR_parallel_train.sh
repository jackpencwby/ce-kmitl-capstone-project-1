#!/usr/bin/env bash
#SBATCH --partition=defq
#SBATCH --account=advanced
#SBATCH --qos=advanced
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=192G
#SBATCH --time=5-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=GBR_all6
#SBATCH --output=experiment-stage-2/run_logs/GBR_all6-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
SCRIPT="$PWD/experiment-stage-2/GBR_parallel_train.sh"
RUN_ROOT=$(realpath -m "${1:?Supply persistent run root}")
ATTEMPT=${2:-0}
PREVIOUS_JOB=${3:-}
if [[ -n "$PREVIOUS_JOB" ]]; then
    [[ "$PREVIOUS_JOB" =~ ^[0-9]+$ ]] || exit 2
    previous=$(scontrol show job "$PREVIOUS_JOB" -o 2>/dev/null || true)
    state=$(printf '%s' "$previous" | sed -n 's/.*JobState=\([^ ]*\).*/\1/p')
    if [[ -z "$state" ]]; then
        state=$(sacct -X -nP -j "$PREVIOUS_JOB" --format=JobIDRaw,State | awk -F'|' -v id="$PREVIOUS_JOB" '$1==id {print $2}')
    fi
    case "$state" in
        COMPLETED|TIMEOUT) ;;
        *) echo "Predecessor was cancelled, failed or unavailable; not restarting training: $state"; exit 1 ;;
    esac
fi
[[ "$ATTEMPT" =~ ^[0-9]+$ ]] || exit 2
mkdir -p "$RUN_ROOT"
PYTHON=${GBR_PYTHON:-$PWD/.venv-gbr/bin/python}
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
pids=()
continuation=''
interrupted=0
cancelled=0
request_resume() {
    interrupted=1
    if [[ -z "$continuation" && "$ATTEMPT" -lt 9 ]]; then
        if continuation=$(sbatch --parsable --dependency="afterany:${SLURM_JOB_ID:?}" "$SCRIPT" "$RUN_ROOT" "$((ATTEMPT+1))" "$SLURM_JOB_ID"); then
            continuation=${continuation%%;*}
            echo "Queued continuation $continuation"
        else
            continuation=''
            echo 'Continuation submission failed; manual resume required.' >&2
        fi
    fi
    for pid in "${pids[@]}"; do kill -USR1 "$pid" 2>/dev/null || true; done
}
request_cancel() {
    cancelled=1; interrupted=1
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
}
trap request_resume USR1
trap request_cancel TERM
for exp in 5 6 11 12 17 18; do
    run="$RUN_ROOT/E$exp"
    args=(--experiment "$exp" --train --source local --checkpoint --run-dir "$run")
    [[ ! -f "$run/checkpoint/manifest.json" ]] || args+=(--resume "$run")
    echo "Launching E$exp Job=${SLURM_JOB_ID:-manual} Run=$run"
    "$PYTHON" -u experiment-stage-2/run_stage2_gradient_boosting.py "${args[@]}" >> "$RUN_ROOT/E$exp.log" 2>&1 &
    pids+=("$!")
done
failed=0
needs_resume=0
set +e
for pid in "${pids[@]}"; do
    while true; do
        wait "$pid"
        status=$?
        if [[ "$interrupted" == 1 && "$status" -gt 128 ]]; then interrupted=0; continue; fi
        break
    done
    echo "Process $pid exit=$status"
    if [[ "$status" == 75 ]]; then needs_resume=1
    elif [[ "$status" != 0 ]]; then failed=1; fi
done
set -e
trap - USR1 TERM
if [[ "$cancelled" == 1 ]]; then exit 143; fi
if [[ "$failed" == 1 ]]; then
    [[ -z "$continuation" ]] || scancel "$continuation"
    exit 1
fi
if [[ "$needs_resume" == 1 && -z "$continuation" ]]; then
    echo 'Stopped with checkpoints; manual resume required.' >&2
    exit 75
fi
echo 'Allocation finished. Completed experiments will be skipped on resume.'
