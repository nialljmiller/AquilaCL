#!/bin/bash
# Level-2 sequence model over a level-1 run's clip embeddings (aquilacl.seq) on one H200.
# Usage: PARENT=<level-1 run dir> RUN=<new run name> WINDOWS=<windows dir> SPLIT=<split.json> \
#        sbatch slurm/13_seq.sh [extra seq.py options, e.g. --time-encoding relative]
# Output: $ROOT/runs/$RUN. Resubmitting with the same RUN resumes from its latest checkpoint; seq.py refuses a
# non-empty RUN directory that it did not create, a RUN equal to PARENT, and a parent whose held-out set is not
# the split's search set.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1,gp-2
#SBATCH --job-name=aquilacl_seq
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
ROOT=/project/ai4wy-eap/nmille39/aquilacl
: "${PARENT:?set PARENT to a level-1 run directory}"; : "${RUN:?set RUN to a run name}"
: "${WINDOWS:?set WINDOWS}"; : "${SPLIT:?set SPLIT to the scorecard split.json}"
test -d "$PARENT/embeddings" || { echo "no $PARENT/embeddings"; exit 1; }
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
echo "== seq $RUN (parent $PARENT) $(date)"
python -m aquilacl.seq --parent "$PARENT" --windows "$WINDOWS" --split "$SPLIT" --run "$ROOT/runs/$RUN" "$@"
echo "== done $(date)"
