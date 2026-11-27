#!/bin/bash
# Scorecard pre-freeze counts and the candidate injection list (model-free). Freeze is a separate manual step:
#   python scripts/scorecard_precount.py --freeze $OUT/injections_candidate.parquet --split $SPLIT
# Usage: OUT=<new dir> sbatch slurm/12a_precount.sh
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1,gp-2
#SBATCH --job-name=aquilacl_sc_precount
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
R=/project/ai4wy-eap/nmille39/aquilacl
V=$R/results_2026-11/v4
: "${OUT:?set OUT}"
cd "$HOME/AquilaCL"
python scripts/scorecard_precount.py --windows "$R/data_2026-11/windows_v4" --split "$V/scorecard_split/split.json" \
    --reference "$R/source/movebank_2026-11-24/LifeTrack Golden Eagle Bighorn Basin WY-reference-data(1).csv" \
    --states2 "$V/states_r6_k2" --residency "$V/residency_r6_v3b" --out "$OUT" --workers 15
