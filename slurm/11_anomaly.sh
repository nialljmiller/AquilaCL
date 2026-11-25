#!/bin/bash
# Stage 7 v2 (CPU only): sensor/recording QC on the raw windows (no model) and behavioural anomalies on h
# (full windows; window, bout and day levels; bird among birds), validated with injected faults and h-level
# injections. No re-encoding, no GPU.
# Reads $WINDOWS, the runs and their K=2 / K=6 states; writes only into the new directory $OUT.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_anomaly
#SBATCH --cpus-per-task=36
#SBATCH --mem=300G
#SBATCH --time=03:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
# Usage: RUNS="<run dir> [<run dir> ...]" STATES2="<K=2 states dir per run>" STATES6="<K=6 states dir per run>" \
#        WINDOWS=<windows dir> REFERENCE=<reference CSV> OUT=<new dir> [K=20] [N_INJECT=2000] [SEED=0] sbatch 11_anomaly.sh
: "${RUNS:?set RUNS to one or more run directories}"; : "${STATES2:?set STATES2, one per run}"
: "${STATES6:?set STATES6, one per run}"; : "${WINDOWS:?set WINDOWS}"
: "${REFERENCE:?set REFERENCE to the reference-data CSV}"; : "${OUT:?set OUT to a new directory}"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then echo "$OUT is not empty; give a new directory" >&2; exit 1; fi
read -r -a run_arr <<< "$RUNS"; read -r -a s2_arr <<< "$STATES2"; read -r -a s6_arr <<< "$STATES6"
if [ "${#run_arr[@]}" -ne "${#s2_arr[@]}" ] || [ "${#run_arr[@]}" -ne "${#s6_arr[@]}" ]; then
    echo "RUNS, STATES2 and STATES6 differ in length" >&2; exit 1
fi
args=()
for i in "${!run_arr[@]}"; do args+=(--run "${run_arr[$i]}" --states2 "${s2_arr[$i]}" --states6 "${s6_arr[$i]}"); done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # parallelism comes from joblib workers
echo "== anomaly $(date)"
python -m aquilacl.anomaly "${args[@]}" --windows "$WINDOWS" --reference "$REFERENCE" --out "$OUT" \
    --workers $(( SLURM_CPUS_PER_TASK - 2 )) ${K:+--k "$K"} ${N_INJECT:+--n-inject "$N_INJECT"} ${SEED:+--seed "$SEED"}
echo "== done $(date)"
