#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing PyTorch..."
if [[ "$(uname)" == "Darwin" ]]; then
    echo "  Detected macOS — installing CPU/MPS build"
    pip install torch torchvision
else
    echo "  Detected Linux — installing CUDA 12.1 build"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

echo "Installing robomimic (modified fork)..."
pip install -e robomimic/

echo "Installing CALVIN environment..."
(cd calvin && bash install.sh)

echo "Installing DynaGuide..."
pip install -e .
pip install -r requirements.txt

echo "Installing additional dependencies for analysis..."
pip install diffusers tensorboard

echo "Done. Activate with: source .venv/bin/activate"
