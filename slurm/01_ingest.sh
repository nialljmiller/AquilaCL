#!/bin/bash
# Stage 1 (CPU only): CSV -> sorted per-bird, per-sensor Parquet. Reads GoldenEAI.csv; writes only under $OUT.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=ge_ingest
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/goldeneai_v2/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate goldeneai
SRC=/project/ai4wy-eap/nmille39/GoldenEAI/GoldenEAI.csv
OUT=/project/ai4wy-eap/nmille39/goldeneai_v2/data
# Known values from the full raw audit of 2026-11-23 (vega room golden-eagle, raw-event-dictionary.md).
python -m goldeneai.ingest "$SRC" "$OUT" --workers 32 --expect \
    total=40092394 acceleration=13203555 gps=14876799 magnetometer=6006020 orientation=6006020 birds=62
