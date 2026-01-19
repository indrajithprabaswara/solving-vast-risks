# Supplementary Materials

## Overview
This package contains the complete source code and configuration files required to reproduce the results presented in the paper.

## Directory Structure
*   `experiments/`: Source code for simulation logic and experiment runners.
*   `analysis/`: Scripts for generating figures from result data.
*   `metrics/`: Connectivity and safety metric calculations.
*   `simulation/`: Core engine logic.
*   `requirements.txt`: Python package dependencies.
*   `run_all.py`: Master entry point to run the full E1-E7 suite.
*   `run_smoke_test.py`: Fast verification script.

## System Requirements
*   **OS**: Cross-platform (Windows, Linux, macOS).
*   **Python**: Version 3.9+.
*   **Hardware**: Standard CPU.
*   **Dependencies**: `networkx`, `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `joblib`.

## Installation
1.  Unzip the package.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Verification (Smoke Test)
Run the smoke test to verify the environment. This runs a minimal simulation (N=100) in under 10 seconds.

```bash
python run_smoke_test.py
```
**Success Criteria**: Prints "SMOKE TEST PASSED".

## Reproduction Instructions

### 1. Running Simulations
To generate the raw data for all experiments (E1-E7), run:

```bash
python run_all.py
```
*   **Output**: Results stored in `experiments/results/*.parquet`.
*   **Runtime**: Approx 30-45 minutes on a modern CPU.

### 2. Generating Plots
After simulations complete, generate the figures:

```bash
python analysis/paper_plots.py
```
*(Or `analysis/plotting.py` depending on specific figure needs. `run_all.py` calls the main plotting routine automatically.)*

## Notes
*   **Seeds**: Experiments use seeds 0-9 for statistical robustness.
*   **Parallelism**: Scripts default to using available CPU cores via `joblib`.
