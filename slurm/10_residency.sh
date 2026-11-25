#!/bin/bash
# Movement strategy per bird from the raw GPS tracks (rule classes, primary, and NSD models; tracks cut at deploy-off
# and before a stationary end; gap-aware migration dates; threshold sensitivity at 750/1000/1250 km from both
# origins; per-bird distances rounded to 50 km, no northing, no coordinates written), then coarse migrants vs
# residents (resident_full primary, present_partial_year as sensitivity) in four variants with cohort = deployment
# year. The nuisance floor of every grouping is fitted first; groupings with floor AUC >= 0.9 are "not testable in
# this design" (no p-values). Tested where testable: K=2 flight share of scheduled first-of-burst daylight windows,
# flight kinematics, grouped-CV classifier on mean h with bird-label permutation nulls (CPU only).
# Reads GPS, WINDOWS, REFERENCE, STATES2, STATES6, RUN; writes only into OUT, which must be new or empty.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_residency
#SBATCH --cpus-per-task=36
#SBATCH --mem=300G
#SBATCH --time=02:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
# Usage: GPS=<raw gps dir> WINDOWS=<windows dir> REFERENCE="<Movebank reference CSV>" STATES2=<K=2 states dir> \
#        STATES6=<K=6 states dir> RUN=<run dir> OUT=<new dir> [MIN_DAYS=120] [N_PERM=2000] sbatch 10_residency.sh
: "${GPS:?set GPS to the raw GPS directory}"; : "${WINDOWS:?set WINDOWS}"
: "${REFERENCE:?set REFERENCE to the Movebank reference-data CSV}"
: "${STATES2:?set STATES2 to the K=2 states directory}"; : "${STATES6:?set STATES6 to the K=6 states directory}"
: "${RUN:?set RUN to the run directory}"; : "${OUT:?set OUT to a new directory}"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "$OUT is not empty; give a new directory" >&2; exit 1
fi
# parallelism is through joblib workers; keep BLAS single-threaded per worker
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "== residency $(date)"
python -m aquilacl.residency --gps "$GPS" --windows "$WINDOWS" --reference "$REFERENCE" --states2 "$STATES2" \
    --states6 "$STATES6" --run "$RUN" --out "$OUT" --min-days "${MIN_DAYS:-120}" --n-perm "${N_PERM:-2000}" \
    --workers $(( SLURM_CPUS_PER_TASK - 2 ))
echo "== done $(date)"
