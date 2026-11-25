#!/bin/bash
# CPU: what distinguishes the two sensitivity settings after the common format (noise floor, burst joins, season).
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_senscues
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
: "${WINDOWS:?set WINDOWS}"; : "${OUT:?set OUT to a new JSON file}"
cd "$SLURM_SUBMIT_DIR"
python scripts/sensitivity_cues.py --windows "$WINDOWS" --format "$WINDOWS/format.json" --out "$OUT"
