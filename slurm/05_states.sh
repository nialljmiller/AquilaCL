#!/bin/bash
# Stage 5a (CPU only): behaviour states on a frozen embedding: PCA + diagonal GMM over-clustering, Ward merge,
# K chosen by bird-bootstrap stability, held-out-bird agreement and presence; HMM smoothing; state profiles.
# Reads $RUN and $WINDOWS; writes only into the new directory $OUT.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_states
#SBATCH --cpus-per-task=36
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
# Usage: RUN=<run dir> WINDOWS=<windows dir> OUT=<new dir> [K=<states>] [K_MIN=<smallest K searched, default 6>] sbatch 05_states.sh
: "${RUN:?set RUN to the run directory}"; : "${WINDOWS:?set WINDOWS}"; : "${OUT:?set OUT to a new directory}"
echo "== states $(date)"
python -m aquilacl.states --run "$RUN" --windows "$WINDOWS" --out "$OUT" --workers $(( SLURM_CPUS_PER_TASK - 2 )) \
    ${K:+--k "$K"} ${K_MIN:+--k-min "$K_MIN"}
echo "== done $(date)"
