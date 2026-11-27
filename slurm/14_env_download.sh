#!/bin/bash
# Environment data download (resumable): ERA5 hourly surface fields from ARCO-ERA5 (Google Cloud), cropped to the
# fix bounding box, one HDF5 file per variable-month; Copernicus GLO-30 DEM tiles of the 1-degree cells touched by
# fixes. Writes only under $ENV_DIR/{era5,dem}. Unlike the other stages this does NOT refuse an existing output
# directory: reruns skip every complete file and fetch only what is missing (nothing is ever overwritten).
# Needs internet. Two ways to run it:
#   compute node (only if compute nodes reach the internet):  ENV_DIR=... RAW_GPS=... sbatch 14_env_download.sh
#   login node (no SLURM_JOB_ID): ENV_DIR=... RAW_GPS=... bash 14_env_download.sh
#     -> runs under nice -n 19 / ionice -c3 with a shared rate cap MAX_MBPS (default 40 MB/s) and 8 streams.
# Optional: WHAT=era5,dem  VARS=u10,v10,u100,v100,blh,ishf  START=2024-06-01  END=2026-11-20  STREAMS=8
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_env_download
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
: "${ENV_DIR:?set ENV_DIR (e.g. /project/ai4wy-eap/nmille39/aquilacl/env)}"; : "${RAW_GPS:?set RAW_GPS}"
ARGS=(--raw-gps "$RAW_GPS" --env "$ENV_DIR" --what "${WHAT:-era5,dem}" --vars "${VARS:-u10,v10,u100,v100,blh,ishf}"
      --start "${START:-2024-06-01}" --end "${END:-2026-11-20}" --streams "${STREAMS:-8}")
echo "== env download $(date) host $(hostname)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    python -m aquilacl.env download "${ARGS[@]}" --max-mbps "${MAX_MBPS:-0}"
else
    nice -n 19 ionice -c3 python -m aquilacl.env download "${ARGS[@]}" --max-mbps "${MAX_MBPS:-40}"
fi
echo "== du"; du -sh "$ENV_DIR/era5" "$ENV_DIR/dem" 2>/dev/null || true
echo "== leftover .partial files (from interrupted runs; never deleted automatically)"
find "$ENV_DIR" -name '*.partial' | wc -l
echo "== done $(date)"
