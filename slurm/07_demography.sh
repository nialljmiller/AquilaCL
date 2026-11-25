#!/bin/bash
# Bird-level demography from the latent space (CPU only): confounds, sex and tagged-young classifiers with grouped CV
# and bird-label permutation nulls, wingbeat positive control, age development of tagged-young birds.
# Reads RUNS, STATES (same order), WINDOWS and REFERENCE; writes only into OUT, which must be new or empty.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_demography
#SBATCH --cpus-per-task=36
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
# Usage: RUNS="<run dir> [<run dir> ...]" STATES="<states dir> [...]" WINDOWS=<windows dir> \
#        REFERENCE="<Movebank reference CSV>" OUT=<new dir> [N_PERM=2000] [N_PERM_MONTH=200] sbatch 07_demography.sh
# RUNS and STATES are split on spaces (their paths must not contain spaces); REFERENCE may contain spaces.
: "${RUNS:?set RUNS to the run directories (space-separated)}"; : "${STATES:?set STATES (same order as RUNS)}"
: "${WINDOWS:?set WINDOWS}"; : "${REFERENCE:?set REFERENCE to the Movebank reference-data CSV}"
: "${OUT:?set OUT to a new directory}"
read -ra run_arr <<< "$RUNS"
read -ra st_arr <<< "$STATES"
if [ "${#run_arr[@]}" -ne "${#st_arr[@]}" ]; then
    echo "RUNS has ${#run_arr[@]} entries but STATES has ${#st_arr[@]}" >&2; exit 1
fi
args=()
for i in "${!run_arr[@]}"; do args+=(--run "${run_arr[$i]}" --states "${st_arr[$i]}"); done
# parallelism is through joblib workers; keep BLAS single-threaded per worker
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "== demography $(date)"
python -m aquilacl.demography "${args[@]}" --windows "$WINDOWS" --reference "$REFERENCE" --out "$OUT" \
    --n-perm "${N_PERM:-2000}" --n-perm-month "${N_PERM_MONTH:-200}" --workers $(( SLURM_CPUS_PER_TASK - 2 ))
echo "== done $(date)"
