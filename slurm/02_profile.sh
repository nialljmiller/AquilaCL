#!/bin/bash
# Stage 2 (CPU only): sampling schedule, per-tag accelerometer calibration, burst features, GPS kinematics.
# Reads $DATA/events; writes only $DATA/profile (refuses if it is not empty).
#SBATCH --account=ai4wy-eap
#SBATCH --partition=gp-1
#SBATCH --job-name=ge_profile
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/project/ai4wy-eap/nmille39/goldeneai_v2/logs/%x_%j.log
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"; conda activate goldeneai
DATA=/project/ai4wy-eap/nmille39/goldeneai_v2/data
python -m goldeneai.profile "$DATA" --workers 62
