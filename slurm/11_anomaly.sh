#!/bin/bash
# Stage 7 (CPU only): anomalies at two levels (window within its bird; bird among birds), validated with injected
# faults re-encoded on the CPU (n_inject per fault type is small: about 14,000 windows per run).
# Reads the runs, their K=2 states and $WINDOWS; writes only into the new directory $OUT.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_anomaly
#SBATCH --cpus-per-task=36
#SBATCH --mem=300G
#SBATCH --time=06:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
# Usage: RUNS="<run dir> [<run dir> ...]" STATES2="<K=2 states dir per run, same order>" WINDOWS=<windows dir> \
#        REFERENCE=<reference CSV> OUT=<new dir> [K=20] [N_INJECT=2000] [SEED=0] sbatch 11_anomaly.sh
: "${RUNS:?set RUNS to one or more run directories}"; : "${STATES2:?set STATES2, one per run}"
: "${WINDOWS:?set WINDOWS}"; : "${REFERENCE:?set REFERENCE to the reference-data CSV}"; : "${OUT:?set OUT to a new directory}"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then echo "$OUT is not empty; give a new directory" >&2; exit 1; fi
read -r -a run_arr <<< "$RUNS"; read -r -a st_arr <<< "$STATES2"
if [ "${#run_arr[@]}" -ne "${#st_arr[@]}" ]; then echo "RUNS and STATES2 differ in length" >&2; exit 1; fi
args=()
for i in "${!run_arr[@]}"; do args+=(--run "${run_arr[$i]}" --states2 "${st_arr[$i]}"); done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # parallelism comes from joblib workers
echo "== anomaly $(date)"
python -m aquilacl.anomaly "${args[@]}" --windows "$WINDOWS" --reference "$REFERENCE" --out "$OUT" \
    --workers $(( SLURM_CPUS_PER_TASK - 2 )) --device cpu ${K:+--k "$K"} ${N_INJECT:+--n-inject "$N_INJECT"} \
    ${SEED:+--seed "$SEED"}
echo "== done $(date)"
