#!/bin/bash
# Flight continuum (PCA of h in K=2 flight windows, flight maps, bird-level sex tests per recording mode) and
# continuous-mode flight bouts (descriptors, GMM + Ward bout types under the fixed stability rule). CPU only.
# Reads RUNS, STATES2, STATES6 (same order), WINDOWS and REFERENCE; writes only into OUT, which must be new or empty.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_flight
#SBATCH --cpus-per-task=36
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
# Usage: RUNS="<run dir> [<run dir>]" STATES2="<K=2 states dir> [...]" STATES6="<K=6 states dir> [...]" \
#        WINDOWS=<windows dir> REFERENCE="<Movebank reference CSV>" OUT=<new dir> [N_BOOT=100] [N_PERM=2000] [UMAP=1] \
#        sbatch 09_flight.sh
# RUNS, STATES2 and STATES6 are split on spaces (their paths must not contain spaces); REFERENCE may contain spaces.
: "${RUNS:?set RUNS to the run directories (space-separated)}"; : "${STATES2:?set STATES2 (same order as RUNS)}"
: "${STATES6:?set STATES6 (same order as RUNS)}"; : "${WINDOWS:?set WINDOWS}"
: "${REFERENCE:?set REFERENCE to the Movebank reference-data CSV}"; : "${OUT:?set OUT to a new directory}"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then echo "$OUT is not empty; give a new directory" >&2; exit 1; fi
read -ra run_arr <<< "$RUNS"
read -ra s2_arr <<< "$STATES2"
read -ra s6_arr <<< "$STATES6"
if [ "${#run_arr[@]}" -ne "${#s2_arr[@]}" ] || [ "${#run_arr[@]}" -ne "${#s6_arr[@]}" ]; then
    echo "RUNS has ${#run_arr[@]} entries, STATES2 ${#s2_arr[@]}, STATES6 ${#s6_arr[@]}" >&2; exit 1
fi
args=()
for i in "${!run_arr[@]}"; do args+=(--run "${run_arr[$i]}" --states2 "${s2_arr[$i]}" --states6 "${s6_arr[$i]}"); done
# parallelism is through joblib workers; keep BLAS single-threaded per worker
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "== flight $(date)"
python -m aquilacl.flight "${args[@]}" --windows "$WINDOWS" --reference "$REFERENCE" --out "$OUT" \
    --n-boot "${N_BOOT:-100}" --n-perm "${N_PERM:-2000}" --workers $(( SLURM_CPUS_PER_TASK - 2 )) ${UMAP:+--umap}
echo "== done $(date)"
