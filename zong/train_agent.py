"""
train_agent.py

Trains a robot RL agent that teaches a human teleoperator by influencing their
internal dynamics estimate (theta_H = B_human_vec). Implements the robot
planning framework from Section 5 of Tian et al. 2023 (HRI).

Full pipeline:
  1. HumanRobotEnv wraps the shared-autonomy simulation as a gymnasium.Env.
  2. PPO (stable-baselines3) trains the robot policy inside that env.
  3. Evaluation compares "active teach" vs. "passive learn" on theta_H error.

Two training modes:
  - Oracle: theta_H updated by the ground-truth gradient-learner rule.
  - Dyna:   theta_H updated by the frozen HumanDynamicsTransformer from
            train_human_model.py (the Tian et al. method).
"""

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from riccati import dare

# Reuse the transformer architecture and dtype constant so the checkpoint
# loaded here is compatible with the one saved by train_human_model.py.
from train_human_model import HumanDynamicsTransformer, DTYPE

# Module-level DARE solver instance, consistent with train_human_model.py.
DARE_SOLVER = dare()


# ──────────────────────────────────────────────────────────────────────────────
# Shared-autonomy Gymnasium environment
# ──────────────────────────────────────────────────────────────────────────────

class HumanRobotEnv(gym.Env):
    """Shared-autonomy env for teaching the human about true robot dynamics.

    The human has an incorrect internal model theta_H = B_human_vec (they
    over-estimate the robot's responsiveness) and updates it each step via a
    gradient-learner rule or the trained transformer.  The robot blends its
    commanded action with the human's LQR command:

        u = alpha * u_R + (1 - alpha) * u_H       (Eq. 13 in paper)

    The blended action changes x_{t+1}, which the human attributes to u_H
    alone.  This mismatch is what lets the robot steer the human's gradient
    update toward the true dynamics.

    Observation (6-D float32): [x, y, b_x, b_y, goal_x, goal_y]
    Action      (2-D float32): robot velocity command u_R in [-u_max, u_max]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        A_env: torch.Tensor,       # true environment state-transition matrix
        B_env: torch.Tensor,       # true environment control-input matrix
        A_human: torch.Tensor,     # A matrix the human assumes (identity here)
        Q_human: torch.Tensor,     # human LQR state-cost matrix
        R_human: torch.Tensor,     # human LQR action-cost matrix
        dt: float,                 # simulation timestep (seconds)
        alpha: float = 0.5,        # robot blending weight (Eq. 13)
        theta_star: torch.Tensor = None,   # true B_env diagonal; human's learning target
        theta_init: torch.Tensor = None,   # human's initial (wrong) B_env estimate
        eta: float = 0.005,        # gradient-learner step size for oracle mode
        beta: float = 5.0,         # effort penalty weight in the reward
        u_max: float = 2.0,        # symmetric action bound for u_R
        max_steps: int = 150,      # episode length (matches training rollout length)
        goal_threshold: float = 0.1,
        use_learned_dynamics: bool = False,  # True → use transformer for theta update
        transformer_model: HumanDynamicsTransformer = None,
        context_len: int = 150,    # max history tokens fed to the transformer
    ):
        super().__init__()

        # ── Dynamics and cost matrices ────────────────────────────────────────
        self.A_env = A_env
        self.B_env = B_env
        self.A_human = A_human
        self.Q_human = Q_human
        self.R_human = R_human
        self.dt = dt

        # ── Blending and learning hyperparameters ─────────────────────────────
        self.alpha = alpha
        self.eta = eta           # only used in oracle mode
        self.beta = beta

        # ── Episode configuration ─────────────────────────────────────────────
        self.u_max = u_max
        self.max_steps = max_steps
        self.goal_threshold = goal_threshold

        # ── Transformer / dynamics mode ───────────────────────────────────────
        self.use_learned_dynamics = use_learned_dynamics
        self.transformer_model = transformer_model
        self.context_len = context_len

        # The human should converge to theta_star = diag(B_env).
        # In our setup B_env = diag([0.2, 0.2]), so theta_star = [0.2, 0.2].
        if theta_star is None:
            self.theta_star = torch.diag(B_env).to(dtype=DTYPE)
        else:
            self.theta_star = theta_star.to(dtype=DTYPE)

        # Human's starting (wrong) belief matches the rollout generation in
        # train_human_model.py: B_human_vec = [0.7, 0.7].
        if theta_init is None:
            self.theta_init = torch.tensor([0.7, 0.7], dtype=DTYPE)
        else:
            self.theta_init = theta_init.to(dtype=DTYPE)

        # Diamond-pattern target sequence, same as human_environment_simulation().
        # The duplicate North at index 4 lets target_index cycle 0→3 without
        # an out-of-bounds access on the final goal check.
        radius = 2.0
        self.targets = torch.tensor([
            [0.0,   radius],   # North  (index 0)
            [radius, 0.0],     # East   (index 1)
            [0.0,  -radius],   # South  (index 2)
            [-radius, 0.0],    # West   (index 3)
            [0.0,   radius],   # North  (index 4, duplicate for safe cycling)
        ], dtype=DTYPE)

        # ── Gymnasium spaces ──────────────────────────────────────────────────
        # Generous bounds; SB3's PPO normalises advantages internally.
        obs_high = np.array([10.0, 10.0, 5.0, 5.0, 3.0, 3.0], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.full(2, -u_max, dtype=np.float32),
            high=np.full(2,  u_max, dtype=np.float32),
        )

        # Runtime state — populated in reset().
        self.x: torch.Tensor = None
        self.theta_H: torch.Tensor = None
        self.target_index: int = 0
        self.step_count: int = 0
        # Rolling buffer of 6-D tokens [x_t, x_{t+1}, u_H_t] for the transformer.
        self.history: list = []

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Start a new episode near a random target with the wrong initial theta_H."""
        super().reset(seed=seed)

        # Pick a random starting target and nudge off-centre, matching training.
        self.target_index = int(np.random.randint(len(self.targets) - 1))
        self.x = self.targets[self.target_index] + torch.tensor([0.1, -0.1], dtype=DTYPE)

        # Reset the human's internal model to its prior wrong belief.
        self.theta_H = self.theta_init.clone()

        self.step_count = 0
        self.history = []

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        """Advance one timestep given the robot's commanded velocity u_R."""
        u_R = torch.tensor(action, dtype=DTYPE)
        current_goal = self.targets[self.target_index]

        # ── 1. Human computes LQR policy under their current theta_H ──────────
        # B_human is what the human *thinks* the control-input matrix is.
        B_human = torch.diag(self.theta_H) * self.dt
        P = DARE_SOLVER(self.A_human, B_human, self.Q_human, self.R_human)
        # S = R + B^T P B; solving the linear system avoids explicit inversion.
        S = self.R_human + B_human.T @ P @ B_human
        K = torch.linalg.solve(S, B_human.T @ P @ self.A_human)
        # Closed-form Gaussian policy mean and covariance (Eq. 11 in paper).
        mu_H = -K @ (self.x - current_goal)
        Sigma_H = torch.linalg.inv(2.0 * S)
        u_H = torch.distributions.MultivariateNormal(mu_H, Sigma_H).sample()

        # ── 2. Blend robot and human actions (Eq. 13) ─────────────────────────
        u_blend = self.alpha * u_R + (1.0 - self.alpha) * u_H

        # ── 3. Advance physical state with TRUE dynamics ───────────────────────
        # The human only knows A_human and B_human (their model), not A_env/B_env.
        x_next = self.A_env @ self.x + self.B_env @ u_blend * self.dt

        # ── 4. Update the human's internal model ──────────────────────────────
        # Key teaching mechanism: the human observes x_next (which includes
        # the robot's contribution via u_blend) but attributes it entirely to
        # their own action u_H.  This skewed attribution drives their gradient
        # update in a direction the robot can exploit.
        if self.use_learned_dynamics and self.transformer_model is not None:
            self.theta_H = self._transformer_theta_update(self.x, x_next, u_H)
        else:
            self.theta_H = self._oracle_theta_update(self.x, x_next, u_H)

        # ── 5. Advance to the next goal once the current one is reached ────────
        if torch.linalg.norm(x_next - current_goal) < self.goal_threshold:
            # Cycle through indices 0-3; index 4 is a safe duplicate of 0.
            self.target_index = (self.target_index + 1) % (len(self.targets) - 1)

        self.x = x_next
        self.step_count += 1

        # ── 6. Reward (Eq. 14 in paper) ───────────────────────────────────────
        # Term 1: penalise remaining misalignment between human model and truth.
        # Term 2: penalise how much the robot deviated from the human's action.
        #         ||u_blend - u_H||^2 = alpha^2 * ||u_R - u_H||^2, so this
        #         discourages large robot interventions (minimal-intervention).
        theta_error = float(torch.linalg.norm(self.theta_H - self.theta_star) ** 2)
        effort = float(torch.linalg.norm(u_blend - u_H) ** 2)
        reward = -theta_error - self.beta * effort

        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {
            "theta_H":     self.theta_H.detach().numpy().copy(),
            "theta_error": theta_error,
            "u_H":         u_H.detach().numpy().copy(),
            "u_R":         u_R.detach().numpy().copy(),
            "u_blend":     u_blend.detach().numpy().copy(),
            # x_next is the state the human reached this step (after blended action).
            "x":           x_next.detach().numpy().copy(),
        }
        return self._get_obs(), reward, terminated, truncated, info

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Return float32 observation [x, theta_H, goal]."""
        goal = self.targets[self.target_index]
        obs = torch.cat([self.x, self.theta_H, goal])
        return obs.detach().numpy().astype(np.float32)

    def _oracle_theta_update(
        self,
        x_t: torch.Tensor,
        x_next: torch.Tensor,
        u_H: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient-learner update: one gradient step on dynamics prediction MSE.

        Mirrors lines 150-159 of human_environment_simulation() in
        train_human_model.py.  The human minimises:
            MSE = ||x_{t+1} - (A_human @ x_t + B_human @ u_H)||^2
        where x_{t+1} was produced by the BLENDED action, not u_H alone.
        This discrepancy is exactly what the robot exploits to teach.
        """
        # Temporarily enable gradients on a fresh copy of theta_H.
        b_vec = self.theta_H.clone().detach().requires_grad_(True)
        B_human = torch.diag(b_vec) * self.dt
        # x_next came from the blended action; human thinks it came from u_H.
        residual = x_next.detach() - (
            self.A_human @ x_t.detach() + B_human @ u_H.detach()
        )
        mse = residual @ residual
        mse.backward()

        with torch.no_grad():
            # Gradient descent step, then clamp to keep theta_H physically valid.
            new_b = b_vec - self.eta * b_vec.grad
            new_b = new_b.clamp(min=0.0001, max=5.0)
        return new_b.detach()

    def _transformer_theta_update(
        self,
        x_t: torch.Tensor,
        x_next: torch.Tensor,
        u_H: torch.Tensor,
    ) -> torch.Tensor:
        """Dyna update: use the frozen trained transformer to predict theta_{t+1}.

        Each step, the transition token [x_t, x_{t+1}, u_H_t] is appended to a
        rolling buffer.  The transformer reads the full buffer and returns the
        predicted theta at the final position, which is theta_{t+1}.

        This is the Tian et al. method: the robot uses its inferred model of
        human learning dynamics rather than the true (unknown) gradient rule.
        """
        # Build the 6-D token and append to the rolling history buffer.
        token = torch.cat([
            x_t.detach(), x_next.detach(), u_H.detach()
        ]).to(dtype=DTYPE)                        # shape: [6]
        self.history.append(token)

        # Keep only the last context_len transitions so we match the transformer's
        # training sequence length.
        if len(self.history) > self.context_len:
            self.history = self.history[-self.context_len:]

        # Stack into [1, T, 6] and run the forward pass (no grad needed).
        seq = torch.stack(self.history, dim=0).unsqueeze(0)   # [1, T, 6]
        self.transformer_model.eval()
        with torch.no_grad():
            # theta_pred[:, t, :] = theta_{t+1}: updated belief after step t.
            theta_pred = self.transformer_model(seq)           # [1, T, theta_dim]

        # The last position of theta_pred holds the prediction for step t+1.
        return theta_pred[0, -1, :].detach()


# ──────────────────────────────────────────────────────────────────────────────
# No-intervention baseline policy
# ──────────────────────────────────────────────────────────────────────────────

class PassivePolicy:
    """Sends zero robot action: human learns by themselves (passive-learn baseline)."""

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return np.zeros(2, dtype=np.float32), None


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_robot_policy(
    env: HumanRobotEnv,
    total_timesteps: int = 200_000,
    policy_save_path: str = "robot_policy",
) -> object:
    """Train a PPO robot policy inside a HumanRobotEnv.

    The MlpPolicy maps [x, theta_H, goal] -> u_R using a 2-hidden-layer MLP.
    Training uses whatever dynamics mode is set on `env`
    (oracle or dyna/transformer).

    Requires stable-baselines3 (pip install stable-baselines3).
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        raise ImportError(
            "stable-baselines3 is required for training.\n"
            "Install with:  pip install stable-baselines3"
        )

    # Validate the env conforms to the gymnasium API before starting.
    print("Checking environment...")
    check_env(env, warn=True)

    print(f"Training PPO for {total_timesteps:,} timesteps "
          f"(dynamics_mode={'dyna' if env.use_learned_dynamics else 'oracle'})...")

    # n_steps=2048 means ~13 episodes per rollout buffer (150 steps/episode).
    # batch_size=64 matches the SB3 default and gives ~32 minibatches per update.
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95, 
        clip_range=0.2,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(policy_save_path)
    print(f"Policy saved to {policy_save_path}.zip")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_policy(
    policy,
    env_kwargs: dict,
    n_episodes: int = 10,
    label: str = "policy",
) -> dict:
    """Roll out a policy for n_episodes and return a dict of trajectory arrays.

    Returns
    -------
    dict with keys:
      "theta_errors" : float32 [n_episodes, max_steps]
          Per-step ||theta_H - theta_star||^2.
      "states"       : float32 [n_episodes, max_steps, 2]
          Human's position x after each blended step.
      "u_H"          : float32 [n_episodes, max_steps, 2]
          Human's LQR command at each step (before blending).
      "u_R"          : float32 [n_episodes, max_steps, 2]
          Robot's commanded action at each step.

    Creates a fresh env from env_kwargs each call so different policies can be
    compared on identical episode initializations when the RNG seed is fixed.
    """
    env = HumanRobotEnv(**env_kwargs)

    # Accumulate per-episode lists; each list element is one step's value.
    all_theta_errors = []
    all_states       = []
    all_u_H          = []
    all_u_R          = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        ep_errors = []
        ep_states = []
        ep_u_H    = []
        ep_u_R    = []
        done = False

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_errors.append(info["theta_error"])
            ep_states.append(info["x"])      # shape (2,) — position after this step
            ep_u_H.append(info["u_H"])       # shape (2,) — human LQR command
            ep_u_R.append(info["u_R"])       # shape (2,) — robot command

        all_theta_errors.append(ep_errors)
        all_states.append(ep_states)
        all_u_H.append(ep_u_H)
        all_u_R.append(ep_u_R)

        print(f"  [{label}] ep {episode+1:2d}/{n_episodes} | "
              f"final theta_error = {ep_errors[-1]:.4f}")

    # Stack to fixed-shape arrays; all episodes are truncated at max_steps so
    # every inner list has the same length and np.array works without padding.
    return {
        "theta_errors": np.array(all_theta_errors, dtype=np.float32),  # [E, T]
        "states":       np.array(all_states,       dtype=np.float32),  # [E, T, 2]
        "u_H":          np.array(all_u_H,          dtype=np.float32),  # [E, T, 2]
        "u_R":          np.array(all_u_R,          dtype=np.float32),  # [E, T, 2]
    }


def plot_evaluation(active_arr: np.ndarray, passive_arr: np.ndarray, prefix = " ") -> None:
    """Plot mean ± std of theta error for active-teach vs passive-learn.

    Mirrors Figure 3 (right column) and Figure 5 (left) from Tian et al. 2023.
    """
    timesteps = np.arange(active_arr.shape[1])

    fig, ax = plt.subplots(figsize=(8, 4))

    # Active-teach curve
    ax.plot(timesteps, active_arr.mean(0), label="active teach (ours)", color="orange")
    ax.fill_between(
        timesteps,
        active_arr.mean(0) - active_arr.std(0),
        active_arr.mean(0) + active_arr.std(0),
        alpha=0.2, color="orange",
    )

    # Passive-learn baseline
    ax.plot(timesteps, passive_arr.mean(0), label="passive learn", color="gray")
    ax.fill_between(
        timesteps,
        passive_arr.mean(0) - passive_arr.std(0),
        passive_arr.mean(0) + passive_arr.std(0),
        alpha=0.2, color="gray",
    )

    ax.set_xlabel("Timestep")
    ax.set_ylabel(r"$\|\theta_H - \theta^*\|^2$")
    ax.set_title("Human Internal Model Error: Active Teach vs Passive Learn")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{prefix} evaluation_theta_error.png", dpi=150)
    plt.show()

    # save the raw data for potential further analysis or overplotting with other methods
    np.savez(
        f"{prefix} evaluation_theta_error_data.npz",
        active_teach=active_arr,
        passive_learn=passive_arr,
    )

    # Print summary statistics matching the quantitative results section.
    print(f"\nActive teach  — final: {active_arr[:, -1].mean():.4f} "
          f"± {active_arr[:, -1].std():.4f}")
    print(f"Passive learn — final: {passive_arr[:, -1].mean():.4f} "
          f"± {passive_arr[:, -1].std():.4f}")


def plot_comparison(
    oracle_npz_path: str,
    dyna_npz_path: str,
    out_prefix: str = "comparison",
) -> None:
    """Load saved NPZ files from plot_evaluation and overlay all three curves.

    Each NPZ has keys 'active_teach' and 'passive_learn' (shape [n_episodes, T]).
    Plots oracle active, dyna active, and passive learn on the same axes so all
    three can be compared directly.  The passive baseline is taken from the oracle
    NPZ; both NPZs should contain equivalent passive curves since the passive
    policy is policy-independent (alpha=0).

    Args:
        oracle_npz_path: path to the oracle evaluation NPZ, e.g. "oracle_ evaluation_theta_error_data.npz"
        dyna_npz_path:   path to the dyna evaluation NPZ, e.g. "dyna_evaluation_theta_error_data.npz"
        out_prefix:      filename prefix for the saved figure and NPZ summary
    """
    # Load oracle run: contains oracle active-teach and passive-learn arrays.
    oracle_data = np.load(oracle_npz_path)
    oracle_active  = oracle_data["active_teach"]   # [n_eps, T]
    passive_arr    = oracle_data["passive_learn"]   # [n_eps, T]

    # Load dyna run: contains dyna active-teach (passive_learn is equivalent).
    dyna_data = np.load(dyna_npz_path)
    dyna_active = dyna_data["active_teach"]         # [n_eps, T]

    timesteps = np.arange(oracle_active.shape[1])

    fig, ax = plt.subplots(figsize=(9, 4))

    def _plot_curve(arr, label, color):
        # Helper to plot mean line with ±1 std shaded band.
        mean = arr.mean(0)
        std  = arr.std(0)
        ax.plot(timesteps, mean, label=label, color=color)
        ax.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color=color)

    _plot_curve(oracle_active, "oracle active teach", color="steelblue")
    _plot_curve(dyna_active,   "dyna active teach",   color="orange")
    _plot_curve(passive_arr,   "passive learn",        color="gray")

    ax.set_xlabel("Timestep")
    ax.set_ylabel(r"$\|\theta_H - \theta^*\|^2$")
    ax.set_title("Human Internal Model Error: Oracle vs Dyna vs Passive")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    fig.savefig(f"{out_prefix}_comparison_theta_error.png", dpi=150)
    plt.show()

    # Print final-step summary for all three curves.
    print(f"\nOracle active — final: {oracle_active[:, -1].mean():.4f} "
          f"± {oracle_active[:, -1].std():.4f}")
    print(f"Dyna active   — final: {dyna_active[:, -1].mean():.4f} "
          f"± {dyna_active[:, -1].std():.4f}")
    print(f"Passive learn — final: {passive_arr[:, -1].mean():.4f} "
          f"± {passive_arr[:, -1].std():.4f}")


def plot_comparison_from_results(
    oracle_result: dict,
    dyna_result: dict,
    passive_result: dict,
    out_prefix: str = "eval_all",
) -> None:
    """Same three-curve theta-error plot as plot_comparison() but from in-memory
    evaluate_policy() result dicts instead of saved NPZ files."""
    oracle_active = oracle_result["theta_errors"]   # [E, T]
    dyna_active   = dyna_result["theta_errors"]     # [E, T]
    passive_arr   = passive_result["theta_errors"]  # [E, T]

    timesteps = np.arange(oracle_active.shape[1])
    fig, ax = plt.subplots(figsize=(9, 4))

    def _plot_curve(arr, label, color):
        mean = arr.mean(0)
        std  = arr.std(0)
        ax.plot(timesteps, mean, label=label, color=color)
        ax.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color=color)

    _plot_curve(oracle_active, "oracle active teach", color="steelblue")
    _plot_curve(dyna_active,   "dyna active teach",   color="orange")
    _plot_curve(passive_arr,   "passive learn",        color="gray")

    ax.set_xlabel("Timestep")
    ax.set_ylabel(r"$\|\theta_H - \theta^*\|^2$")
    ax.set_title("Human Internal Model Error: Oracle vs Dyna vs Passive")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_theta_error.png", dpi=150)
    plt.show()

    print(f"\nOracle active — final: {oracle_active[:, -1].mean():.4f} "
          f"± {oracle_active[:, -1].std():.4f}")
    print(f"Dyna active   — final: {dyna_active[:, -1].mean():.4f} "
          f"± {dyna_active[:, -1].std():.4f}")
    print(f"Passive learn — final: {passive_arr[:, -1].mean():.4f} "
          f"± {passive_arr[:, -1].std():.4f}")


def plot_trajectories(
    oracle_result: dict,
    dyna_result: dict,
    passive_result: dict,
    out_prefix: str = "trajectories",
) -> None:
    """Qualitative trajectory comparison across oracle, dyna, and passive conditions.

    Each dict must be the output of evaluate_policy() and contain:
      "states" [E, T, 2]  — (x, y) position after each step
      "u_H"   [E, T, 2]  — human LQR command at each step
      "u_R"   [E, T, 2]  — robot command at each step

    Produces a figure with two rows:

      Row 1 — 2-D spatial paths (one column per condition).
              All individual episodes are drawn as faint lines; the
              per-timestep mean across episodes is drawn thick.
              The diamond target pattern is overlaid for reference.

      Row 2 — Time-series of control norms.
              Left:  ||u_H|| for all three conditions (mean ± 1 std).
              Right: ||u_R|| for oracle and dyna (passive ≈ 0 by design).
    """
    # Diamond waypoints that define the task (radius matches HumanRobotEnv).
    radius = 2.0
    diamond_x = np.array([0.0, radius, 0.0, -radius, 0.0])
    diamond_y = np.array([radius, 0.0, -radius, 0.0, radius])

    conditions = [
        ("Oracle active", oracle_result,  "steelblue"),
        ("Dyna active",   dyna_result,    "orange"),
        ("Passive learn", passive_result, "gray"),
    ]
    timesteps = np.arange(oracle_result["states"].shape[1])

    fig = plt.figure(figsize=(14, 9))

    # ── Row 1: 2-D spatial trajectories ─────────────────────────────────────
    for col, (title, result, color) in enumerate(conditions):
        ax = fig.add_subplot(2, 3, col + 1)   # positions 1, 2, 3

        states = result["states"]   # [E, T, 2]
        n_eps  = states.shape[0]

        # Individual episode paths — faint so the mean stands out.
        for ep in range(n_eps):
            ax.plot(
                states[ep, :, 0],
                states[ep, :, 1],
                color=color, alpha=0.15, linewidth=0.8,
            )

        # Per-timestep mean trajectory across all episodes.
        mean_traj = states.mean(axis=0)   # [T, 2]
        ax.plot(
            mean_traj[:, 0], mean_traj[:, 1],
            color=color, linewidth=2.0, label="mean",
        )

        # Mark start of the mean trajectory.
        ax.plot(mean_traj[0, 0], mean_traj[0, 1], "o", color=color, markersize=6)

        # Diamond target outline — shared reference across all three panels.
        ax.plot(diamond_x, diamond_y, "k--", linewidth=1.0, alpha=0.5, label="targets")
        ax.plot(diamond_x[:-1], diamond_y[:-1], "k^", markersize=7, alpha=0.7)

        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # ── Row 2, left: ||u_H|| norms over time ────────────────────────────────
    ax_uh = fig.add_subplot(2, 3, 4)
    for title, result, color in conditions:
        # ||u_H||_2 at each step: shape [E, T].
        u_H_norms = np.linalg.norm(result["u_H"], axis=-1)
        mean = u_H_norms.mean(axis=0)
        std  = u_H_norms.std(axis=0)
        ax_uh.plot(timesteps, mean, color=color, label=title)
        ax_uh.fill_between(timesteps, mean - std, mean + std, color=color, alpha=0.15)

    ax_uh.set_xlabel("Timestep")
    ax_uh.set_ylabel(r"$\|u_H\|_2$")
    ax_uh.set_title("Human control magnitude over time")
    ax_uh.legend(fontsize=8)
    ax_uh.grid(True, alpha=0.3)

    # ── Row 2, right: ||u_R|| norms over time (oracle & dyna only) ──────────
    ax_ur = fig.add_subplot(2, 3, 5)
    for title, result, color in conditions[:2]:   # skip passive (u_R ≈ 0)
        u_R_norms = np.linalg.norm(result["u_R"], axis=-1)
        mean = u_R_norms.mean(axis=0)
        std  = u_R_norms.std(axis=0)
        ax_ur.plot(timesteps, mean, color=color, label=title)
        ax_ur.fill_between(timesteps, mean - std, mean + std, color=color, alpha=0.15)

    ax_ur.set_xlabel("Timestep")
    ax_ur.set_ylabel(r"$\|u_R\|_2$")
    ax_ur.set_title("Robot control magnitude over time")
    ax_ur.legend(fontsize=8)
    ax_ur.grid(True, alpha=0.3)

    # ── Row 2, right-most: blended-control breakdown ─────────────────────────
    # Show mean ||u_H|| and ||u_R|| side-by-side per condition as bar chart so
    # the overall contribution balance is immediately readable.
    ax_bar = fig.add_subplot(2, 3, 6)
    x_pos = np.arange(len(conditions))
    bar_width = 0.35
    mean_u_H = [np.linalg.norm(r["u_H"], axis=-1).mean() for _, r, _ in conditions]
    mean_u_R = [np.linalg.norm(r["u_R"], axis=-1).mean() for _, r, _ in conditions]
    ax_bar.bar(x_pos - bar_width / 2, mean_u_H, bar_width, label=r"$\|u_H\|$", color="mediumpurple")
    ax_bar.bar(x_pos + bar_width / 2, mean_u_R, bar_width, label=r"$\|u_R\|$", color="salmon")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([t for t, _, _ in conditions], fontsize=8)
    ax_bar.set_ylabel("Mean control norm (all steps & episodes)")
    ax_bar.set_title("Average control contribution")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Qualitative trajectory comparison: Oracle vs Dyna vs Passive",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_trajectories.png", dpi=150)
    plt.show()
    print(f"Trajectory figure saved to {out_prefix}_trajectories.png")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Dynamics parameters: must match train_human_model.py exactly ──────────
    A_human = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    A_env   = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=DTYPE)
    # True B: how the environment actually responds to controls.
    B_env   = torch.tensor([[0.2, 0.0], [0.0, 0.2]], dtype=DTYPE)
    Q_human = torch.eye(2, dtype=DTYPE)
    R_human = 0.01 * torch.eye(2, dtype=DTYPE)
    dt = 0.2

    # theta_star: what the human's internal model should converge to = diag(B_env).
    theta_star = torch.tensor([0.2, 0.2], dtype=DTYPE)
    # theta_init: human's starting (wrong) belief, same as training rollouts.
    theta_init = torch.tensor([0.7, 0.7], dtype=DTYPE)

    # Base environment keyword arguments shared across all modes.
    base_env_kwargs = dict(
        A_env=A_env,
        B_env=B_env,
        A_human=A_human,
        Q_human=Q_human,
        R_human=R_human,
        dt=dt,
        alpha=0.5,
        theta_star=theta_star,
        theta_init=theta_init,
        eta=0.005,
        beta=1.0,
        u_max=2.0,
        max_steps=150,
        goal_threshold=0.1,
        use_learned_dynamics=False,
    )

    # ── Select mode ───────────────────────────────────────────────────────────
    #mode = "train_oracle"    # Train with ground-truth gradient-learner dynamics
    #mode = "train_dyna"      # Train with frozen transformer dynamics (Tian et al.)
    #mode = "evaluate"        # Load a saved policy and compare against passive learn
    mode = "evaluate_all"    # Run oracle, dyna, passive; plot trajectories + theta error
    #mode = "compare"           # Load saved NPZ files and overlay oracle/dyna/passive

    # ── Oracle training ───────────────────────────────────────────────────────
    if mode == "train_oracle":
        env = HumanRobotEnv(**base_env_kwargs)
        train_robot_policy(
            env,
            total_timesteps=200_000,
            policy_save_path="robot_policy_oracle",
        )

    # ── Dyna training (uses frozen transformer from train_human_model.py) ─────
    if mode == "train_dyna":
        # learn_human_dynamics() saves: torch.save(model.state_dict(), "human_dynamics_transformer.pth")
        # Run train_human_model.py with mode="train transformer" if this file is missing.
        ckpt_path = "human_dynamics_transformer.pth"
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Transformer checkpoint not found at '{ckpt_path}'.\n"
                "Run train_human_model.py with mode='train transformer' first."
            )
        transformer = HumanDynamicsTransformer()
        transformer.load_state_dict(state_dict)
        transformer.eval()

        # Freeze all transformer parameters so they do not change during RL.
        for param in transformer.parameters():
            param.requires_grad_(False)

        dyna_env_kwargs = {
            **base_env_kwargs,
            "use_learned_dynamics": True,
            "transformer_model": transformer,
            "context_len": 150,
        }
        env = HumanRobotEnv(**dyna_env_kwargs)
        train_robot_policy(
            env,
            total_timesteps=200_000,
            policy_save_path="robot_policy_dyna",
        )

    # ── Evaluation ────────────────────────────────────────────────────────────
    if mode == "evaluate":
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError("Install stable-baselines3: pip install stable-baselines3")

        # Load the saved oracle policy (swap filename to test the dyna policy).
        print("Loading trained policy...")
        trained_policy = PPO.load("robot_policy_oracle")
        #trained_policy = PPO.load("robot_policy_dyna")
        print("\nEvaluating active-teach policy (robot intervenes)...")
        active_result = evaluate_policy(
            trained_policy,
            env_kwargs=base_env_kwargs,
            n_episodes=1,
            label="active_teach",
        )

        # Passive baseline: same env but robot sends zero command, so the human
        # only learns from their own (uncorrected) dynamics experience.
        print("\nEvaluating passive-learn baseline (no robot intervention)...")
        passive_result = evaluate_policy(
            PassivePolicy(),
            env_kwargs={**base_env_kwargs, "alpha": 0.0},
            n_episodes=1,
            label="passive_learn",
        )

        # plot_evaluation expects raw [n_episodes, T] theta-error arrays.
        plot_evaluation(
            active_result["theta_errors"],
            passive_result["theta_errors"],
            prefix="oracle_",
        )

    # ── Evaluate all three conditions and plot trajectories ───────────────────
    if mode == "evaluate_all":
        try:
            from stable_baselines3 import PPO
        except ImportError:
            raise ImportError("Install stable-baselines3: pip install stable-baselines3")

        # Load the oracle policy.
        print("Loading oracle policy...")
        oracle_policy = PPO.load("robot_policy_oracle")

        # Load the dyna policy and its transformer checkpoint.
        print("Loading dyna policy and transformer checkpoint...")
        ckpt_path = "human_dynamics_transformer.pth"
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        transformer = HumanDynamicsTransformer()
        transformer.load_state_dict(state_dict)
        transformer.eval()
        for param in transformer.parameters():
            param.requires_grad_(False)
        dyna_policy = PPO.load("robot_policy_dyna")

        dyna_env_kwargs = {
            **base_env_kwargs,
            "use_learned_dynamics": True,
            "transformer_model": transformer,
            "context_len": 150,
        }

        # Run all three rollouts.
        print("\nEvaluating oracle active-teach...")
        oracle_result = evaluate_policy(oracle_policy, base_env_kwargs, n_episodes=1, label="oracle")

        print("\nEvaluating dyna active-teach...")
        dyna_result = evaluate_policy(dyna_policy, dyna_env_kwargs, n_episodes=1, label="dyna")

        print("\nEvaluating passive-learn baseline...")
        passive_result = evaluate_policy(
            PassivePolicy(),
            {**base_env_kwargs, "alpha": 0.0},
            n_episodes=1, label="passive",
        )

        # Theta-error curves (quantitative) and spatial trajectory figure (qualitative).
        plot_comparison_from_results(oracle_result, dyna_result, passive_result)
        plot_trajectories(oracle_result, dyna_result, passive_result, out_prefix="eval_all")

    # ── Comparison plot (loads from previously saved NPZ files) ───────────────
    if mode == "compare":
        plot_comparison(
            oracle_npz_path="oracle_ evaluation_theta_error_data.npz",
            dyna_npz_path="dyna_evaluation_theta_error_data.npz",
            out_prefix="comparison",
        )
