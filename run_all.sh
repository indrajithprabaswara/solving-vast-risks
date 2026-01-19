#!/bin/bash
set -euo pipefail

echo "=========================================================="
echo "Reproducing 'Solving Vast Risks' Experiments & Plots"
echo "=========================================================="

# Check for python
if ! command -v python &> /dev/null; then
    echo "Error: python could not be found."
    exit 1
fi

echo "[1/3] Installing/Verifying Dependencies..."
pip install -r requirements.txt

echo "[2/3] Running Experiments (E1-E7)..."
# This runs the full suite and saves to experiments/results/
python run_all.py

echo "[3/3] Generating Paper Plots..."
python generate_all_plots.py

echo "=========================================================="
echo "Done! Results in experiments/results/, Figures in root/."
echo "=========================================================="
