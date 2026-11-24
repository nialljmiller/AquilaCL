#!/bin/bash
# Stage 3 (CPU only): 10 s windows + features + reference labels, handcrafted baseline (GMM + HMM smoothing),
# and probes on the first-draft step-30k embeddings (read only). Writes only under $ROOT.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_stage3
#SBATCH --cpus-per-task=36
#SBATCH --mem=300G
#SBATCH --time=08:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
ROOT=/project/ai4wy-eap/nmille39/aquilacl
DATA=$ROOT/data
OLD=/project/ai4wy-eap/nmille39/GoldenEAI/logs/main/00030000_embeddings.csv
echo "== windows $(date)"
python -m aquilacl.windows "$DATA" --profile "$DATA/profile_v2" --out "$DATA/windows_10s" --workers $(( SLURM_CPUS_PER_TASK - 4 ))
echo "== baseline $(date)"
python -m aquilacl.baseline "$DATA/windows_10s" --out "$ROOT/results/baseline_v1"
echo "== probes on old embeddings $(date)"
python -m aquilacl.probes old --embeddings "$OLD" --windows "$DATA/windows_10s" --events "$DATA/events" --out "$ROOT/results/probes_old"
echo "== done $(date)"
