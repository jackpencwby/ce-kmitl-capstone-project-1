#!/usr/bin/env bash
# Submit from the repository root. Resume with: sbatch .../Japan_E2_13_train.sh --resume RUN_DIR
#SBATCH --job-name=Japan_E2_13
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1g.23gb:1
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=experiment-stage-2/run_logs/Japan_E2_13-%j.log
#SBATCH --error=experiment-stage-2/run_logs/Japan_E2_13-%j.log
set -euo pipefail
cd "${JAPAN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-.}}"
export VIRTUAL_ENV="$PWD/.venv-japan"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
PYTHON="$VIRTUAL_ENV/bin/python"
ENTRY="experiment-stage-2/Japan_E2_13_global_local_tree_single_head.py"
test -f "$ENTRY"
echo "Job=${SLURM_JOB_ID:-manual} Host=$(hostname) Python=$PYTHON CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON" -c 'import sys,xgboost,torch; print("Python:",sys.executable,"XGBoost:",xgboost.__version__); assert torch.cuda.is_available(), "No CUDA device allocated"; print("GPU:",torch.cuda.get_device_name(0))'
exec "$PYTHON" -u "$ENTRY" --device cuda:0 --source local --train --checkpoint "$@"
