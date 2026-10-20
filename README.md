# RECAP partial experimental artifact package

This repository is a **partial experimental artifact package** associated with the manuscript:

> **RECAP: Communication-Aware Cooperative Planning for Connected Autonomous Multi-UAV Systems Under Delayed and Lossy Links**

It currently contains selected plotting code, synthetic experiments, exported tables and figures, and partial closed-loop simulation data. It does **not** contain the complete implementation of every algorithm or experiment described in the manuscript. The complete research code will be made publicly available after publication of the paper.

## Current release scope

The present package includes:

- selected synthetic stress-envelope and diagnostic experiments;
- partial multi-UAV closed-loop simulation code and exported traces;
- CSV/JSON data used for selected experimental summaries;
- generated figures and LaTeX table fragments.

The present package does not yet include the complete end-to-end implementation of all semantic-reasoning, recovery, validation, distributed predictive-control, baseline, and statistical-analysis components described in the manuscript.

## Repository layout

- `src/`: Python scripts for the simulations and artifact generation.
- `data/`: CSV traces, summaries, parameter sweeps, and JSON manifests.
- `results/`: Generated figures.
- `results/tables/`: Generated LaTeX tables and summary text.

The supplied data are simulation outputs rather than measurements from human participants or physical flight trials.

## Requirements

- Python 3.10 or later
- NumPy
- pandas
- Matplotlib

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Reproducing the artifacts

Run the scripts from the repository root. Output paths are resolved relative to the repository, so the same commands also work from another working directory.

```bash
python src/simulate_results.py
python src/generate_artifact_bundle.py
python src/generate_real_closed_loop_artifacts.py
python src/generate_minimal_rollout_figure.py
```

The scripts use fixed random seeds where stochastic simulation is involved. They regenerate the CSV/JSON files under `data/`, the figures under `results/`, and the LaTeX fragments under `results/tables/`.

## Data notes

See [`data/README.md`](data/README.md) for a concise description of the exported files.

## Complete code release

The complete implementation will be added to this repository after the associated paper is published.

## Citation

If you use this artifact package, please cite the associated paper. Full bibliographic details will be added after publication.

