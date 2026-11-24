#!/bin/bash
# One-time setup on the AI4WY login node: a NEW env "aquilacl" cloned from the working "ai4wy" env
# (torch 2.10+cu128 for aarch64 already works there). The "ai4wy" env itself is not changed.
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"
conda create -y -n aquilacl --clone ai4wy
conda activate aquilacl
pip install pyarrow pytest
pip install --no-deps -e "${REPO:?set REPO to the repo checkout}"
python -c "import torch, pyarrow, aquilacl; print(torch.__version__, pyarrow.__version__)"
cd "$REPO" && python -m pytest -q
du -sh "$HOME/opt/miniforge3/envs/aquilacl"
