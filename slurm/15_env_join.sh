#!/bin/bash
# Environment features per window (CPU only): position from raw GPS (own fixes, else nearest within +-30 min),
# ERA5 winds / boundary-layer height / sensible heat flux / w*, GLO-30 elevation, slope, aspect, relief, height above
# ground, orographic uplift. Output $OUT/<bird>.parquet (rows 1:1 with windows_v4/<bird>/meta.parquet, no
# coordinates or timestamps) and $OUT/README.json. Reads $WINDOWS, $RAW_GPS, $ENV_DIR; writes only into the new
# directory $OUT.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_env_join
#SBATCH --cpus-per-task=36
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
# Usage: WINDOWS=<windows_v4> RAW_GPS=<raw/gps> ENV_DIR=<env root> OUT=<new dir> [BIRDS=a,b] sbatch 15_env_join.sh
: "${WINDOWS:?set WINDOWS}"; : "${RAW_GPS:?set RAW_GPS}"; : "${ENV_DIR:?set ENV_DIR}"; : "${OUT:?set OUT to a new directory}"
if [[ -e "$OUT" && -n "$(ls -A "$OUT")" ]]; then echo "refusing: $OUT exists and is not empty"; exit 1; fi
echo "== env join $(date)"
python -m aquilacl.env join --windows "$WINDOWS" --raw-gps "$RAW_GPS" --env "$ENV_DIR" --out "$OUT" \
    --workers $(( SLURM_CPUS_PER_TASK - 2 )) ${BIRDS:+--birds "$BIRDS"}
echo "== done $(date)"
