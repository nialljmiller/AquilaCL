#!/bin/bash
# Stage 5a-2 (CPU only): sub-states within each coarse state of a stage 5a run (PCA refitted within the state,
# diagonal GMM over-clustering, Ward merge, k_sub chosen by a fixed bird-bootstrap Jaccard + presence rule) and
# coarse-state confusion across the stage 5a bootstraps.
# Reads $STATES, $RUN and $WINDOWS; writes only into the new directory $OUT.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_substates
#SBATCH --cpus-per-task=36
#SBATCH --mem=300G
#SBATCH --time=08:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
# Usage: STATES=<stage 5a dir> RUN=<run dir> WINDOWS=<windows dir> OUT=<new dir> [MAX_SUB=6] sbatch 05b_substates.sh
: "${STATES:?set STATES to the stage 5a output directory}"; : "${RUN:?set RUN to the run directory}"
: "${WINDOWS:?set WINDOWS}"; : "${OUT:?set OUT to a new directory}"
echo "== substates $(date)"
python -m aquilacl.substates --states "$STATES" --run "$RUN" --windows "$WINDOWS" --out "$OUT" \
    --workers $(( SLURM_CPUS_PER_TASK - 2 )) ${MAX_SUB:+--max-sub "$MAX_SUB"}
echo "== done $(date)"
