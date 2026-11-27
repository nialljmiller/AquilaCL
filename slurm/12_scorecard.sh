#!/bin/bash
# Goal scorecard (docs/SCORECARD_spec.md v2) of one or more runs on one H200 (GPU only for the GPS-zeroed
# re-embedding of G6/G9b). References and nuisance floors are cached in $OUT/../scorecard_reference/ and shared.
# Usage: RUNS="<run dir> [<run dir> ...]" OUT=<new or empty dir> sbatch slurm/12_scorecard.sh [--final] [--rows G1 G2 ...]
# Optional overrides: WINDOWS SPLIT REFERENCE STATES2 RESIDENCY. Target runtime < 2 h per run.
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1,gp-2
#SBATCH --job-name=aquilacl_scorecard
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
ROOT=/project/ai4wy-eap/nmille39/aquilacl
: "${RUNS:?set RUNS to one or more run directories (space separated)}"
: "${OUT:?set OUT to a new or empty directory}"
WINDOWS=${WINDOWS:-$ROOT/data_2026-11/windows_v4}
SPLIT=${SPLIT:-$ROOT/results_2026-11/v4/scorecard_split/split.json}
REFERENCE=${REFERENCE:-"$ROOT/source/movebank_2026-11-24/LifeTrack Golden Eagle Bighorn Basin WY-reference-data(1).csv"}
STATES2=${STATES2:-$ROOT/results_2026-11/v4/states_r6_k2}
RESIDENCY=${RESIDENCY:-$ROOT/results_2026-11/v4/residency_r6_v3b}
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then echo "$OUT is not empty; give a new directory" >&2; exit 1; fi
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
run_args=(); for r in $RUNS; do run_args+=(--run "$r"); done
echo "== scorecard $(date)"
python -m aquilacl.scorecard "${run_args[@]}" --windows "$WINDOWS" --split "$SPLIT" --reference "$REFERENCE" \
    --states2 "$STATES2" --residency "$RESIDENCY" --out "$OUT" --workers $(( SLURM_CPUS_PER_TASK - 1 )) \
    --device cuda "$@"
echo "== done $(date)"
