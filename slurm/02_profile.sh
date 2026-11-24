#!/bin/bash
# Stage 2 (CPU only): sampling schedule, per-tag accelerometer calibration, burst features, GPS kinematics.
# Reads $DATA/events; writes only $DATA/profile (refuses if it is not empty).
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=aquilacl_profile
#SBATCH --cpus-per-task=36
#SBATCH --mem=300G
#SBATCH --time=08:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/aquilacl/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate aquilacl
DATA=/project/ai4wy-eap/nmille39/aquilacl/data
python -m aquilacl.profile "$DATA" --out "${OUT:-$DATA/profile}" --workers $(( SLURM_CPUS_PER_TASK - 2 ))
