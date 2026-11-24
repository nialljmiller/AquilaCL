#!/bin/bash
# Step 1 (CPU only): CSV -> sorted per-bird, per-sensor Parquet. Reads GoldenEAI.csv, writes only to $OUT.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=ge_ingest
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate goldeneai
SRC=/project/ai4wy-eap/nmille39/GoldenEAI/GoldenEAI.csv
OUT=/project/ai4wy-eap/nmille39/goldeneai_v2/data
test ! -e "$OUT/raw" || { echo "$OUT/raw exists; refusing to overwrite"; exit 1; }
python -m goldeneai.ingest "$SRC" "$OUT" --workers 32
