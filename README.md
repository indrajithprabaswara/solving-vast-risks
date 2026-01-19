# Solving Vast Risks: When Safe Agents Form Broken Networks

This repository contains the official simulation code and data for the
research paper:

> **"Solving Vast Risks: When Safe Agents Form Broken Networks"**

------------------------------------------------------------------------

## Abstract

As AI systems proliferate, their collective safety depends not only on
individual alignment but on the structural properties of their
interaction networks. We investigate the network-level consequences of
local safety filtering in decentralized agent populations.

Utilizing a statistical physics framework, we model safety constraints
as semantic filters that selectively prune communication edges.

We demonstrate that strictly aligned agents in high-dimensional policy
spaces induce a sharp phase transition, causing a sudden collapse in
functional connectivity ($S_{func}$) even when the underlying
infrastructure remains intact ($S_{struct}$). Furthermore, we find that
message complexity acts as a fragility multiplier, exponentially
shifting the critical threshold.

These findings provide a theoretical basis for designing resilient
multi-agent architectures that balance safety with functional consensus.

------------------------------------------------------------------------

## Setup and Installation

To reproduce the experiments, please follow these steps:

### 1. Clone the Repository

``` bash
git clone https://github.com/YOUR_USERNAME/solving-vast-risks.git
cd solving-vast-risks
```

------------------------------------------------------------------------

### 2. Install Dependencies

The code requires **Python 3.9+** and standard scientific libraries.

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Usage

You can run the full simulation suite to generate the data and plots
used in the paper.

### Run All Experiments (Linux / macOS)

``` bash
./run_all.sh
```

------------------------------------------------------------------------

### Run Visualization Script (Windows / All Platforms)

``` bash
python generate_all_plots.py
```

------------------------------------------------------------------------

### Notes

-   Default simulation parameters:
    -   **Number of agents:** `N = 1000`
    -   **Policy space dimensions:** `d = 20`
-   These parameters can be modified directly in the script headers.

------------------------------------------------------------------------

## Citation

If you use this code or findings in your research, please cite our
paper:

``` bibtex
@article{solvingvastrisks2026,
  title={Solving Vast Risks: When Safe Agents Form Broken Networks},
  author={Karunanayaka, Indrajith P.},
  journal={arXiv preprint},
  year={2026}
}
```

------------------------------------------------------------------------

## License

This project is licensed under the **CC BY-NC 4.0**.

See the `LICENSE` file for more details.
