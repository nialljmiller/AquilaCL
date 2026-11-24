#!/bin/bash
# Stage 4: SimCLR training on one H200, deterministic embedding of all windows, evaluation.
# Usage: RUN=<name> sbatch slurm/04_train.sh [extra train.py options]
# Resubmitting with the same RUN resumes from its latest checkpoint.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_train
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
ROOT=/project/ai4wy-eap/nmille39/aquilacl
: "${RUN:?set RUN to a run name}"
DATA=${DATA:-$ROOT/data}
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
echo "== train $RUN $(date)"
python -m aquilacl.train --windows "$DATA/windows_10s" --run "$ROOT/runs/$RUN" "$@"
echo "== evaluate $RUN $(date)"
python -m aquilacl.evaluate --run "$ROOT/runs/$RUN" --windows "$DATA/windows_10s" --events "$DATA/events"
echo "== done $(date)"
