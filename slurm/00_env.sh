#!/bin/bash
# Creates a NEW conda env "goldeneai" next to the existing "ai4wy" env (which is left untouched).
# Run once on the AI4WY login node; it only downloads and installs packages.
set -euo pipefail
source "$HOME/opt/miniforge3/etc/profile.d/conda.sh"
conda create -y -n goldeneai python=3.12
conda activate goldeneai
pip install "torch==2.10.*" --index-url https://download.pytorch.org/whl/cu128
pip install -e "${REPO:-$PWD}[train,analysis,dev]"
python -c "import torch, pyarrow; print(torch.__version__, torch.cuda.is_available(), pyarrow.__version__)"
