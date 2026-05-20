# dVRK Internal Model

Lightweight research code for experimenting with human/internal dynamics models used with the dVRK teleoperation data. This repository contains dataset loaders, modeling code, and example notebooks that explore recovering internal-dynamics parameters and training sequence models.

## Key Contents
- `data/` — place to keep CSV trajectories from either synthetic LQR data or recorded human teleoperation data.
- `dataset/` — dataset utilities and loaders.
- `model/` — model implementations.
- `notebooks/` — example notebooks demonstrating optimization and training workflows.

## Quickstart

1. Create and activate a Python environment.

2. Install the main dependencies (adjust versions to match your system / CUDA):

	```bash
    pip install torch torchvision torchcodec lightning
    ```

Open:
- [notebooks/test_optimization.ipynb](notebooks/test_optimization.ipynb) — experiments to recover LQR/internal-model parameters from synthetic or human trajectories.
- [notebooks/test_train.ipynb](notebooks/test_train.ipynb) — training experiments for sequence-prediction models.

## Notes & Known Issues
- Some optimization runs can be unstable: for example, optimization becomes unstable when the LQR policy `B` parameter is below ~5e-3. Further investigation is required.
- Human data are non-stationary (dynamics vary over time), which makes optimizing a single, fixed-parameter internal model useless. 
- Model architectures and training hyperparameters are experimental and may need tuning for successful training.
