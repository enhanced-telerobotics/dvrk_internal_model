# Human Internal Model Learning

This project implements a small simulation and learning pipeline for estimating a human operator's time-varying internal dynamics model during a 2D reaching task. It attempts to recreate and modify the approach used in Tian et al. 2023, using an LQR-style human controller, a differentiable discrete Riccati solver, synthetic rollout data, and a causal transformer that predicts the human dynamics parameter sequence from observed transitions.

## What Is In This Repository

- `train_human_model.py` - main script for simulation, rollout export, parameter recovery, and transformer training.
- `train_agent.py` - trains and evaluates a PPO robot policy that actively teaches the human by influencing their internal model.
- `riccati.py` - differentiable discrete algebraic Riccati equation solver used inside the human control policy and learning losses.
- `gradient_learner_rollouts/` - saved synthetic rollout CSV files used as training data.
- `human_dynamics_transformer.pth` - saved transformer checkpoint produced by `train_human_model.py`.

## Project Idea

The simulated human moves a 2D point state between targets arranged in a diamond pattern. At each timestep, the human chooses a stochastic control input from an LQR-derived policy:

- state: `[x, y]`
- control: `[u_x, u_y]`
- goal: current target point
- learned human parameter: `B_human_vec = [b_x, b_y]`

The environment evolves according to simple linear dynamics. When `gradient_learner=True`, the simulated human updates its internal dynamics estimate over time. Those changing estimates are saved as `b_x` and `b_y` in each rollout CSV.

The transformer then learns to infer the time-varying human dynamics parameters from rollout prefixes containing:

```text
[x_t, y_t, x_next, y_next, u_x, u_y]
```

The training loss is based on the negative log likelihood of the observed human controls under the LQR policy induced by the predicted dynamics parameters.

## Setup

Create and activate a Python environment, then install the required packages:

```bash
pip install torch numpy scipy matplotlib tensorboard
```

TensorBoard is optional unless you use `tensorboard_log_dir` during training.

## Running `train_human_model.py`

The script is controlled by the `mode` variable near the bottom of `train_human_model.py`:

```python
# mode = "generate_rollouts"
# mode = "recover_b"
mode = "train transformer"
```

After selecting a mode, run:

```bash
python train_human_model.py
```

### Generate Rollouts

Set:

```python
mode = "generate_rollouts"
```

This creates synthetic trajectories and writes CSV files to:

```text
gradient_learner_rollouts/gradient_learner_rollout_XX.csv
```

Each CSV contains:

```text
time, x, y, x_next, y_next, u_x, u_y, goal_x, goal_y, b_x, b_y
```

### Recover A Static Human Dynamics Parameter

Set:

```python
mode = "recover_b"
```

This loads one saved rollout and optimizes a fixed `B_human_vec` by minimizing the negative log likelihood of the observed controls.

### Train The Transformer

Set:

```python
mode = "train transformer"
```

This loads rollout CSVs, trains `HumanDynamicsTransformer`, logs diagnostics if TensorBoard is enabled, and saves the trained model state dict as:

```text
human_dynamics_transformer.pth
```

The current default training configuration uses:

- `context_len=150`
- `num_epochs=50`
- `learning_rate=1e-3`
- `batch_size=4`
- `device="cpu"`

Device selection supports `"cpu"`, `"mps"`, or `"auto"` through `resolve_torch_device()`.

## TensorBoard

When `tensorboard_log_dir` is set, training logs scalar losses, predicted parameter histograms, and rollout-level trajectory diagnostics.

Start TensorBoard with:

```bash
tensorboard --logdir runs
```

Then open the local URL printed by TensorBoard.

## Key Functions

### `train_human_model.py`

- `human_environment_simulation(...)` - simulates the human/environment interaction and optional gradient-based internal model update.
- `save_gradient_learner_trajectory_csv(...)` - exports one rollout to CSV.
- `recover_human_b_from_data(...)` - fits a fixed human dynamics vector from observed data.
- `HumanDynamicsTransformer` - causal transformer that predicts time-varying human dynamics parameters.
- `learn_human_dynamics(...)` - trains the transformer using rollout CSV files.

### `train_agent.py`

- `HumanRobotEnv` - Gymnasium environment wrapping the shared-autonomy simulation. The human holds an incorrect internal model (`theta_H`) and the robot blends its command with the human's LQR command: `u = alpha * u_R + (1 - alpha) * u_H`. The human attributes the blended outcome to their own action, which creates a teaching signal the robot can exploit. Supports two theta update modes: oracle (gradient-learner rule) and dyna (frozen transformer).
- `PassivePolicy` - zero-action baseline that lets the human learn without any robot intervention.
- `train_robot_policy(...)` - trains a PPO `MlpPolicy` (stable-baselines3) inside a `HumanRobotEnv` for a given number of timesteps and saves the checkpoint as a `.zip` file.
- `evaluate_policy(...)` - rolls out a policy for `n_episodes` and returns arrays of theta errors, states, and control signals.
- `plot_evaluation(...)` - plots mean ± std of theta error for active-teach vs passive-learn and saves the figure and raw data as `.npz`.
- `plot_comparison(...)` - loads saved oracle and dyna NPZ files and overlays all three curves (oracle active, dyna active, passive).
- `plot_comparison_from_results(...)` - same three-curve plot from in-memory `evaluate_policy()` result dicts.
- `plot_trajectories(...)` - qualitative figure with 2-D spatial paths and control-norm time series for oracle, dyna, and passive conditions.
- `dare` in `riccati.py` - differentiable wrapper around the discrete algebraic Riccati equation.

## Running `train_agent.py`

Set `mode` near the bottom of `train_agent.py`:

```python
# mode = "train_oracle"    # Train with ground-truth gradient-learner dynamics
# mode = "train_dyna"      # Train with frozen transformer dynamics (Tian et al.)
# mode = "evaluate"        # Load a saved policy and compare against passive learn
mode = "evaluate_all"      # Run oracle, dyna, passive; plot trajectories + theta error
# mode = "compare"         # Load saved NPZ files and overlay oracle/dyna/passive
```

Then run:

```bash
python train_agent.py
```

### Prerequisites

`train_dyna` and `evaluate_all` require `human_dynamics_transformer.pth`. Generate it first by running `train_human_model.py` with `mode = "train transformer"`.

Both training modes require stable-baselines3:

```bash
pip install stable-baselines3
```

### Modes

| Mode | Description | Output |
|------|-------------|--------|
| `train_oracle` | PPO with ground-truth gradient-learner theta updates | `robot_policy_oracle.zip` |
| `train_dyna` | PPO with frozen transformer theta updates | `robot_policy_dyna.zip` |
| `evaluate` | Loads `robot_policy_oracle.zip`, compares active vs passive | `oracle__evaluation_theta_error.png` + `.npz` |
| `evaluate_all` | Loads both policies, runs all three conditions | `eval_all_theta_error.png`, `eval_all_trajectories.png` |
| `compare` | Overlays oracle and dyna from previously saved NPZ files | `comparison_comparison_theta_error.png` |

### Reward Function

The PPO reward at each step (Eq. 14 in paper):

```
reward = -||theta_H - theta_star||^2 - beta * ||u_blend - u_H||^2
```

The first term penalises how far the human's model is from the true dynamics. The second term penalises large robot interventions (minimal-intervention principle).

## Notes

- The code currently uses double precision tensors: `DTYPE = torch.float64`.
- The differentiable Riccati solver in `riccati.py` converts tensors through NumPy/SciPy in the forward pass, so CPU execution is the safest default.
- Training data is synthetic and generated by the same dynamics assumptions used by the learner.
- All dynamics parameters in `train_agent.py` (`A_env`, `B_env`, `Q_human`, `R_human`, `dt`) must match those used in `train_human_model.py` so the loaded transformer checkpoint is compatible with the environment.
