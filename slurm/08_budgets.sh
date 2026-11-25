#!/bin/bash
# Time budgets (CPU only): share of TIME per behaviour from the scheduled (k=0) recording stream, with each scheduled
# sample weighted by the time to the next one (capped at 1.5 x the schedule interval); measures whether the schedule
# continues inside continuous bouts and imputes uncovered bout time as flight when it does not; daily hours stratified
# by local solar hour (uncovered hour bins filled from the bird's own data, then the population); daily, monthly, diel
# and group budgets with bootstrap-over-birds CIs; bird-level permutation tests of daily flight hours.
# Reads WINDOWS, STATES2 (K=2), STATES6 (K=6), SUBSTATES and REFERENCE; writes only into OUT, which must be new or empty.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_budgets
#SBATCH --cpus-per-task=36
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
# Usage: WINDOWS=<windows dir> STATES2=<states_r6_k2> STATES6=<states_r6_yaw60_bpb4> SUBSTATES=<substates_r6_k2> \
#        REFERENCE="<Movebank reference CSV>" OUT=<new dir> [MIN_DIRECT_BINS=12] [FILL=population|own] \
#        [N_BOOT=1000] [N_PERM=5000] \
#        [BOUT_IMPUTATION=auto|on|off] sbatch 08_budgets.sh
: "${WINDOWS:?set WINDOWS}"; : "${STATES2:?set STATES2 to the K=2 states directory}"
: "${STATES6:?set STATES6 to the K=6 states directory}"; : "${SUBSTATES:?set SUBSTATES to the substates directory}"
: "${REFERENCE:?set REFERENCE to the Movebank reference-data CSV}"; : "${OUT:?set OUT to a new directory}"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then
    echo "$OUT exists and is not empty; give a new directory" >&2; exit 1
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "== budgets $(date)"
python -m aquilacl.budgets --windows "$WINDOWS" --states2 "$STATES2" --states6 "$STATES6" --substates "$SUBSTATES" \
    --reference "$REFERENCE" --out "$OUT" --min-direct-bins "${MIN_DIRECT_BINS:-12}" --fill "${FILL:-population}" --n-boot "${N_BOOT:-1000}" \
    --n-perm "${N_PERM:-5000}" --bout-imputation "${BOUT_IMPUTATION:-auto}" --workers $(( SLURM_CPUS_PER_TASK - 2 ))
echo "== done $(date)"
