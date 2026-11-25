#!/bin/bash
# Stage 5b (CPU only): analysis suite and figures for the stage 5a behaviour states (modules M1-M8).
# Reads STATES, WINDOWS, RUN and REFERENCE; writes only into OUT, which must be new or empty.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_analysis
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
# Usage: STATES=<5a dir> WINDOWS=<windows dir> RUN=<run dir> REFERENCE=<Movebank reference CSV> OUT=<new dir> sbatch 06_analysis.sh
: "${STATES:?set STATES to the stage 5a output directory}"; : "${WINDOWS:?set WINDOWS}"; : "${RUN:?set RUN}"
: "${REFERENCE:?set REFERENCE to the Movebank reference-data CSV}"; : "${OUT:?set OUT to a new analysis directory}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "== analysis $(date)"
python -m aquilacl.analysis --states "$STATES" --windows "$WINDOWS" --run "$RUN" --reference "$REFERENCE" --out "$OUT"
echo "== done $(date)"
