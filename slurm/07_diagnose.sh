#!/bin/bash
# Diagnosis of a trained run (CPU only): counterfactual sensitivity test, matched sensitivity probes, and where
# held-out bird identity comes from (behaviour subsets, input perturbations, residual yaw/tilt, feature posture).
# Reads $RUN and $WINDOWS; writes only the new JSON file $OUT (refused if it exists).
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_diagnose
#SBATCH --cpus-per-task=36
#SBATCH --mem=300G
#SBATCH --time=4:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
# Usage: RUN=<run dir> WINDOWS=<windows dir> OUT=<new .json> sbatch 07_diagnose.sh [extra diagnose_r4.py options]
: "${RUN:?set RUN to the run directory}"; : "${WINDOWS:?set WINDOWS}"; : "${OUT:?set OUT to a new JSON file}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
echo "== diagnose $RUN $(date)"
python scripts/diagnose_r4.py --run "$RUN" --windows "$WINDOWS" --out "$OUT" --device cpu "$@"
echo "== done $(date)"
