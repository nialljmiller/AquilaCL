#!/bin/bash
# Stage 1 (CPU only): CSV -> sorted per-bird, per-sensor Parquet. Reads GoldenEAI.csv; writes only under $OUT.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_ingest
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
: "${SRC:?set SRC to the Movebank export (.csv or .csv.zst)}"
: "${OUT:?set OUT to a new data directory}"
# Optional integrity check: SHA256 of the uncompressed export as computed at its origin.
if [ -n "${SHA256:-}" ]; then
    if [[ "$SRC" == *.zst ]]; then reader="zstd -dc"; else reader="cat"; fi
    got=$($reader "$SRC" | sha256sum | cut -d" " -f1)
    echo "sha256 expected $SHA256 got $got"; [ "$got" = "$SHA256" ] || { echo "checksum mismatch"; exit 1; }
fi
# Optional row counts to verify, e.g. EXPECT="total=73759697 acceleration=26728549 birds=94" (2026-11-24 export).
EXPECT=${EXPECT:-}
python -m aquilacl.ingest "$SRC" "$OUT" --workers 32 ${EXPECT:+--expect $EXPECT}
