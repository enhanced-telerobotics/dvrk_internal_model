import glob

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from riccati import dare
import numpy as np
from scipy.linalg import solve_discrete_are
import csv

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# Declare the DARE solver as a global variable
DTYPE = torch.float64

DARE_SOLVER = dare()


# Device selection utility for torch
def resolve_torch_device(device="auto"):
    """Resolve a user-requested torch device string.

    Args:
        device: "auto", "cpu", "mps", or a torch.device.

    Returns:
        torch.device selected for computation.
    """
    if isinstance(device, torch.device):
        return device

    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested device='mps', but torch.backends.mps.is_available() is False.")
        return torch.device("mps")

    if device == "cpu":
        return torch.device("cpu")

    raise ValueError("device must be one of: 'auto', 'cpu', 'mps', or torch.device.")


# The first routine we develop is the basic skeleton of the human model interacting with the environment.

def human_environment_simulation(A_human, B_human_vec, Q_human, R_human, A_env, B_env, dt, total_time,gradient_learner=False,plot=True):

    # initialization

    # define the human model parameters:
    #A_human = torch.tensor([[1.0,0.0],[0.0,1.0]], dtype=DTYPE)
    #B_human_vec = torch.tensor([0.5,0.5], dtype=DTYPE)

    if gradient_learner:
        B_human_vec = torch.nn.Parameter(B_human_vec)
        eta = 0.005
        mse_history = []
        neg_log_likelihood_history = []
    
    b_history = [B_human_vec.clone().detach().cpu().numpy()]

    B_human = torch.diag(B_human_vec)

    #Q_human = torch.eye(2, dtype=DTYPE)
    #R_human = 0.01 * torch.eye(2, dtype=DTYPE)

    # define the environment parameters:
    #A_env = torch.tensor([[1.0,0.0],[0.0,1.0]], dtype=DTYPE)
    #B_env = torch.tensor([[0.25,0.0],[0.0,1.0]], dtype=DTYPE)
    #dt = 0.05
    #total_time = 60.0
    rollout_length = int(total_time / dt)

    radius = 2.0
    targets = torch.tensor([
    [0.0,  radius],   # North
    [radius, 0.0],    # East
    [0.0, -radius],   # South
    [-radius, 0.0],   # West
    [0.0,  radius],   # North
    ], dtype=DTYPE)

    #current_state = torch.tensor([0.0, 0.0], dtype=DTYPE)
    # initialize current state close to one of the targets, to make the rollout more interesting and less likely to diverge due to random initialization of B_human_vec
    target_index = np.random.choice(len(targets))
    current_state = targets[target_index] + torch.tensor([0.1, -0.1], dtype=DTYPE)

    goal_threshold = 0.1
    target_index = 0
    current_goal = targets[target_index]

    state_history = []
    next_state_history = []
    control_history = []
    goal_history = []
    time_history = [0.0]

    for t in range(rollout_length):
        if gradient_learner and B_human_vec.grad is not None:
            B_human_vec.grad.zero_()

        # compute the human control policy
        B_human = torch.diag(B_human_vec)*dt
        P = DARE_SOLVER(A_human, B_human, Q_human, R_human)
        S = R_human + B_human.T @ P @ B_human
        K = torch.linalg.solve(S, B_human.T @ P @ A_human)
        precision = 2.0 * S

        Mu = -K @ (current_state - current_goal)
        Sigma = torch.linalg.inv(precision)
        policy = torch.distributions.MultivariateNormal(
            loc=Mu,
            covariance_matrix=Sigma)
        u_H = policy.sample()

        # take the action
        next_state = A_env @ current_state + B_env @ u_H * dt

        # human learning dynamics
        if gradient_learner:
            
            # # compute the optimal action under the current human policy
            # u_star_H = -K @ (current_state - current_goal) 
            
            # # first compute the log likelihood of the observed action under the current human policy
            # # compute the observed control outcome. Note that we have not consider a non unity A matrix yet...
            # #observed_control = (next_state.detach() - current_state.detach()) / (dt * torch.tensor([B_env[0,0], B_env[1,1]]))
            # observed_control = (next_state.detach() - current_state.detach()) / (dt * B_human_vec.detach())
            # residual = observed_control - u_star_H
            # quadratic_terms = (residual @ precision) @ residual
            # _, logdet_precision = torch.linalg.slogdet(precision)
            # action_dim = observed_control.shape[0]

            # neg_log_likelihood = (
            #      0.5 * quadratic_terms
            #      - 0.5 * logdet_precision
            #      + 0.5 * action_dim * np.log(2.0 * np.pi)
            #  ) * -1.0 # don't know why this sign is flipped, but it seems to be necessary for the gradient ascent step to work correctly. This is worth double checking.
            # neg_log_likelihood.backward() # backprop
            #neg_log_likelihood_history.append(neg_log_likelihood.item())

            # alternatively, we just optimize on the MSE, i.e., the transition error
            observed_control = u_H.detach()
            residual = next_state.detach() - (A_human @ current_state + B_human @ observed_control)
            mse = residual @ residual
            mse.backward()
            mse_history.append(mse.item())

            # update the human model using gradient ascent
            with torch.no_grad():
                B_human_vec -= eta*B_human_vec.grad
                B_human_vec.clamp_(min=0.0001, max=5.0)
            
        b_history.append(B_human_vec.clone().detach().cpu().numpy())
        state_history.append(current_state.detach().cpu().numpy())
        next_state_history.append(next_state.detach().cpu().numpy())
        control_history.append(u_H.detach().cpu().numpy())
        goal_history.append(current_goal.detach().cpu().numpy())

        # check if we need to update the goal state
        dist_to_goal = torch.linalg.norm(next_state - current_goal)
        if dist_to_goal < goal_threshold:
            target_index = (target_index + 1) % len(targets)
            current_goal = targets[target_index]

        current_state = next_state
        time_history.append((t+1)*dt)
    
    if gradient_learner and plot:
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(mse_history)
        plt.title("MSE")
        plt.subplot(3,1,2)
        plt.plot(neg_log_likelihood_history)
        plt.title("Log Likelihood")
        plt.subplot(3,1,3)
        b_history_np = np.array(b_history)
        plt.plot(b_history_np[:,0], label="b_x")
        plt.plot(b_history_np[:,1], label="b_y")
        plt.title("B_human_vec")  
        plt.show()  

    return time_history, state_history, next_state_history, control_history, goal_history, b_history

def plot_trajectory(state_history_np, targets_np, control_history_np):

    plt.figure(figsize=(7, 7))

    plt.plot(
        state_history_np[:, 0],
        state_history_np[:, 1],
        linewidth=2,
        label="Robot trajectory",
    )

    plt.scatter(
        targets_np[:, 0],
        targets_np[:, 1],
        s=120,
        marker="x",
        color="black",
        label="Targets",
    )

    plt.scatter(
        state_history_np[0, 0],
        state_history_np[0, 1],
        s=120,
        marker="o",
        label="Start",
    )

    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Diamond Path Following Simulation")
    plt.legend()

    plt.figure(figsize=(10, 4))

    plt.plot(control_history_np[:, 0], label="u_x")
    plt.plot(control_history_np[:, 1], label="u_y")

    plt.xlabel("Timestep")
    plt.ylabel("Control")
    plt.title("Human Control Inputs")
    plt.grid(True)
    plt.legend()

    plt.show()

# Save rollout as CSV for gradient learner
def save_gradient_learner_trajectory_csv(
    filename,
    time_history,
    state_history,
    next_state_history,
    control_history,
    goal_history,
    b_vec_history,
    dt=0.05,
):
    """Save one gradient-learner rollout as a timestep-indexed CSV file."""
    num_steps = len(control_history)

    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "time",
            "x",
            "y",
            "x_next",
            "y_next",
            "u_x",
            "u_y",
            "goal_x",
            "goal_y",
            "b_x",
            "b_y",
        ])

        for k in range(num_steps):
            time = time_history[k]
            state = state_history[k]
            next_state = next_state_history[k]
            control = control_history[k]
            goal = goal_history[k]
            b_vec = b_vec_history[k]

            writer.writerow([
                time,
                state[0],
                state[1],
                next_state[0],
                next_state[1],
                control[0],
                control[1],
                goal[0],
                goal[1],
                b_vec[0],
                b_vec[1],
            ])

# class HumanDynamicsTransformer(torch.nn.Module):
#     """Causal transformer that predicts time-varying B_human_vec from rollout prefixes.

#     Each token contains the observed transition tuple at one timestep:
#     [x_t, y_t, x_next, y_next, u_x, u_y].

#     A learned start token is prepended to the sequence, which lets the model encode an
#     implicit theta_0 / b_0 rather than requiring it as an observed input feature.
#     """

#     def __init__(
#         self,
#         input_dim=6,
#         theta_dim=2,
#         d_model=64,
#         nhead=4,
#         num_layers=2,
#         dim_feedforward=128,
#         max_context_len=512,
#         theta_min=0.0001,
#         theta_max=1.0,
#     ):
#         super().__init__()
#         self.theta_min = theta_min
#         self.theta_max = theta_max
#         self.max_context_len = max_context_len

#         self.input_proj = torch.nn.Linear(input_dim, d_model, dtype=DTYPE)
#         self.start_token = torch.nn.Parameter(torch.zeros(1, 1, d_model, dtype=DTYPE))
#         self.pos_embedding = torch.nn.Parameter(torch.zeros(1, max_context_len + 1, d_model, dtype=DTYPE))

#         encoder_layer = torch.nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             batch_first=True,
#             activation="gelu",
#             dtype=DTYPE,
#         )
#         self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.output_head = torch.nn.Sequential(
#             torch.nn.LayerNorm(d_model, dtype=DTYPE),
#             torch.nn.Linear(d_model, 64, dtype=DTYPE),
#             torch.nn.GELU(),
#             torch.nn.Linear(64, theta_dim, dtype=DTYPE),
#         )

#     def forward(self, transition_sequence):
#         """Predict theta_{t+1} for every prefix ending at timestep t.

#         Args:
#             transition_sequence: tensor with shape (batch, seq_len, 6).

#         Returns:
#             theta_pred: tensor with shape (batch, seq_len, 2). The kth output uses
#                 transition observations from timesteps 0:k and predicts theta_{k+1}.
#         """
#         batch_size, seq_len, _ = transition_sequence.shape
#         if seq_len > self.max_context_len:
#             raise ValueError(
#                 f"seq_len={seq_len} exceeds max_context_len={self.max_context_len}. "
#                 "Increase max_context_len or train on shorter windows."
#             )

#         token_embedding = self.input_proj(transition_sequence)
#         start_embedding = self.start_token.expand(batch_size, -1, -1)
#         token_embedding = torch.cat([start_embedding, token_embedding], dim=1)
#         token_embedding = token_embedding + self.pos_embedding[:, : seq_len + 1, :]

#         causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
#             seq_len + 1,
#             device=transition_sequence.device,
#             dtype=DTYPE,
#         )
#         encoded = self.encoder(token_embedding, mask=causal_mask)

#         # Drop the start-token output. Output index k now corresponds to theta_{k+1}.
#         raw_theta = self.output_head(encoded[:, 1:, :])
#         theta_pred = raw_theta #self.theta_min + (self.theta_max - self.theta_min) * torch.sigmoid(raw_theta)
#         return theta_pred

class HumanDynamicsTransformer(nn.Module):
    """Causal transformer predicting time-varying human dynamics parameters.

    At each sequence position t, the model receives the observed transition token
    (x^t, u^t, x^{t+1}) and, attending causally to positions 0..t, predicts
    theta_{t+1} — the human's updated internal model after the t-th observation.

    A learned scalar parameter theta_0 encodes the prior estimate of the human's
    initial internal model before any observations are made. This is the implicit
    theta_H^0 referenced in Tian et al. Appendix A.3 and required for the NLL
    loss at t=0 in Eq. (5).

    The forward pass returns theta_pred where theta_pred[:, t, :] = theta_{t+1}.
    Call get_theta_sequence() to obtain the causally aligned sequence where
    position t holds theta_t, suitable for direct use in the NLL loss.
    """

    def __init__(
        self,
        input_dim: int = 6,
        theta_dim: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        max_context_len: int = 64,
    ):
        super().__init__()
        self.theta_dim = theta_dim

        # Per-token MLP encoder: projects (x^t, u^t, x^{t+1}) to d_model.
        # Matches the 3-layer MLP encoder described in Tian et al. Appendix A.3.
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(d_model, d_model, dtype=DTYPE),
            nn.ReLU(),
            nn.Linear(d_model, d_model, dtype=DTYPE),
        )

        # Transformer encoder with batch_first layout [B, T, D].
        # Causal masking is applied at forward time rather than baking it into
        # the module, which allows the same model to be used for both training
        # (full sequence) and online inference (incremental decoding).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            dtype=DTYPE,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Output projection: d_model -> theta_dim.
        self.output_head = nn.Linear(d_model, theta_dim, dtype=DTYPE)

        # Learned initial human parameter theta_0, shared across all rollouts.
        # Represents the human's prior internal model before any interaction,
        # corresponding to theta_H^0 in Tian et al. Eq. (5) and Appendix A.3.
        # Initialized to zero; optimized jointly with transformer weights.
        self.theta_0 = nn.Parameter(torch.ones(theta_dim, dtype=DTYPE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict theta_{t+1} at each sequence position t.

        Args:
            x: Tensor of shape [B, T, input_dim], where each token at position t
               is the concatenation [state_t, next_state_t, control_t].

        Returns:
            theta_pred: Tensor of shape [B, T, theta_dim], where
                        theta_pred[:, t, :] = theta_{t+1}, i.e., the predicted
                        human parameter after observing the t-th transition.
        """
        B, T, _ = x.shape

        # Encode each transition token independently.
        embeddings = self.input_proj(x)  # [B, T, d_model]

        # Causal attention mask: upper triangular with -inf off-diagonal entries.
        # Entry [i, j] = -inf when j > i, preventing position i from attending
        # to any future position j. This enforces the causal dependency
        # theta_{t+1} = f(x^{0:t+1}, u^{0:t}) from Tian et al. Eq. (3).
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), dtype=embeddings.dtype, device=x.device),
            diagonal=1,
        )

        # Apply causal transformer encoder.
        out = self.transformer_encoder(embeddings, mask=causal_mask)  # [B, T, d_model]

        # Project to parameter space.
        # theta_pred[:, t, :] = theta_{t+1}: the updated human model after step t.
        theta_pred = self.output_head(out)  # [B, T, theta_dim]
        return theta_pred

    def get_theta_sequence(self, theta_pred: torch.Tensor) -> torch.Tensor:
        """Construct the NLL-aligned parameter sequence theta_0..theta_{T-1}.

        theta_pred[:, t, :] = theta_{t+1} (output of forward()).
        For the NLL loss at time t, Tian et al. Eq. (5) requires theta_t, not
        theta_{t+1}. This method builds the correctly aligned sequence by
        prepending the learned theta_0 and discarding the last prediction:

            theta_aligned[:, 0, :] = theta_0        (learned prior, no observations)
            theta_aligned[:, 1, :] = theta_1        (= theta_pred[:, 0, :])
            theta_aligned[:, t, :] = theta_t        (= theta_pred[:, t-1, :])

        Args:
            theta_pred: [B, T, theta_dim], output of forward().

        Returns:
            theta_aligned: [B, T, theta_dim], where position t holds theta_t.
        """
        B = theta_pred.shape[0]
        # Broadcast theta_0 over the batch dimension.
        theta_0 = self.theta_0.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # [B, 1, theta_dim]
        # Prepend theta_0 and drop the last column (theta_T, unused in NLL).
        return torch.cat([theta_0, theta_pred[:, :-1, :]], dim=1)  # [B, T, theta_dim]


def _load_rollout_csv_as_tensors(filename):
    """Load one saved rollout CSV into tensors used by the transformer trainer."""
    states = []
    next_states = []
    controls = []
    goals = []
    b_vecs = []

    with open(filename, mode="r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            states.append([float(row["x"]), float(row["y"])])
            next_states.append([float(row["x_next"]), float(row["y_next"])])
            controls.append([float(row["u_x"]), float(row["u_y"])])
            goals.append([float(row["goal_x"]), float(row["goal_y"])])
            b_vecs.append([float(row["b_x"]), float(row["b_y"])])

    states = torch.tensor(states, dtype=DTYPE)
    next_states = torch.tensor(next_states, dtype=DTYPE)
    controls = torch.tensor(controls, dtype=DTYPE)
    goals = torch.tensor(goals, dtype=DTYPE)
    b_vecs = torch.tensor(b_vecs, dtype=DTYPE)

    return states, next_states, controls, goals, b_vecs


def learn_human_dynamics(
    filenames,
    A_human,
    Q_human,
    R_human,
    delta_t,
    context_len=64,
    num_epochs=500,
    learning_rate=1e-4,
    batch_size=16,
    supervised_theta_weight=0.0,
    theta_magnitude_weight=0.5,
    use_theta_supervision=False,
    print_every=25,
    plot=True,
    device="auto",
    tensorboard_log_dir=None,
    tensorboard_eval_every=25,
):
    """Fit a transformer that predicts time-varying human dynamics from rollout prefixes.

    The model receives observed transition/control history up to timestep t:
        [state_0:t, next_state_1:t+1, control_0:t]
    and predicts theta_{t+1}, represented here as B_human_vec at the next timestep.

    The primary loss is the same policy negative log likelihood used in
    recover_human_b_from_data, except B_human_vec is predicted by a transformer at
    each timestep. If the CSV contains ground-truth b_x/b_y values from simulation,
    the ground-truth trajectory is normally used only for diagnostics and TensorBoard
    visualization. Setting use_theta_supervision=True enables an auxiliary theta
    MSE loss for debugging transformer capacity.
    """
    if isinstance(filenames, str):
        filenames = [filenames]

    device = resolve_torch_device(device)
    print(f"learn_human_dynamics using device: {device}")

    A_human = A_human.to(device)
    Q_human = Q_human.to(device)
    R_human = R_human.to(device)

    rollout_tensors = [
        tuple(tensor.to(device) for tensor in _load_rollout_csv_as_tensors(filename))
        for filename in filenames
    ]

    model = HumanDynamicsTransformer(max_context_len=context_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    writer = None
    if tensorboard_log_dir is not None:
        if SummaryWriter is None:
            raise ImportError(
                "TensorBoard logging was requested, but SummaryWriter could not be imported. "
                "Install TensorBoard with: pip install tensorboard"
            )
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        writer.add_text("config/device", str(device), 0)
        writer.add_scalar("config/context_len", context_len, 0)
        writer.add_scalar("config/batch_size", batch_size, 0)
        writer.add_scalar("config/learning_rate", learning_rate, 0)
        writer.add_scalar("config/use_theta_supervision", 1.0 if use_theta_supervision else 0.0, 0)
        writer.add_scalar("config/supervised_theta_weight", supervised_theta_weight, 0)

    windows = []
    for rollout_index, (states, next_states, controls, goals, b_vecs) in enumerate(rollout_tensors):
        num_steps = controls.shape[0]
        if num_steps < context_len:
            continue

        # Version A: each rollout is treated as an independent demonstration with
        # its own implicit theta_0. Therefore, every training window must begin at
        # timestep 0, so the learned start token consistently represents the latent
        # initial human model for that rollout.
        windows.append((rollout_index, 0))

    if len(windows) == 0:
        raise ValueError("No training windows were created. Reduce context_len or provide longer rollouts.")

    for epoch in range(num_epochs):
        permutation = torch.randperm(len(windows), device=device)
        epoch_loss = 0.0
        epoch_nll = 0.0
        epoch_theta_mse = 0.0
        num_batches = 0

        for batch_start in range(0, len(windows), batch_size):
            batch_indices = permutation[batch_start : batch_start + batch_size]

            batch_features = []
            batch_states = []
            batch_controls = []
            batch_goals = []
            batch_theta_targets = []

            # for window_id in batch_indices:
            #     rollout_index, start_index = windows[int(window_id)]
            #     states, next_states, controls, goals, b_vecs = rollout_tensors[rollout_index]
            #     stop_index = start_index + context_len

            #     features = torch.cat(
            #         [
            #             states[start_index:stop_index],
            #             next_states[start_index:stop_index],
            #             controls[start_index:stop_index],
            #         ],
            #         dim=1,
            #     )

            #     batch_features.append(features)
            #     batch_states.append(states[start_index:stop_index])
            #     batch_controls.append(controls[start_index:stop_index])
            #     batch_goals.append(goals[start_index:stop_index])

            #     # Since b_history is saved before each transition, index k+1 is the
            #     # post-update parameter theta_{k+1}. Clamp the last target if needed.
            #     target_start = min(start_index + 1, b_vecs.shape[0] - 1)
            #     target_stop = min(stop_index + 1, b_vecs.shape[0])
            #     theta_target = b_vecs[target_start:target_stop]
            #     if theta_target.shape[0] < context_len:
            #         theta_target = torch.cat(
            #             [theta_target, theta_target[-1:].repeat(context_len - theta_target.shape[0], 1)],
            #             dim=0,
            #         )
            #     batch_theta_targets.append(theta_target)

            for window_id in batch_indices:
                rollout_index, start_index = windows[int(window_id)]
                states, next_states, controls, goals, b_vecs = rollout_tensors[rollout_index]
                stop_index = start_index + context_len

                features = torch.cat(
                    [
                        states[start_index:stop_index],
                        next_states[start_index:stop_index],
                        controls[start_index:stop_index],
                    ],
                    dim=1,
                )

                batch_features.append(features)
                batch_states.append(states[start_index:stop_index])
                batch_controls.append(controls[start_index:stop_index])
                batch_goals.append(goals[start_index:stop_index])

                # CORRECTED: target at position t is theta_t = b_vecs[t].
                # Since T >= context_len is enforced by window creation, b_vecs[0:context_len]
                # is always fully populated — no clamping or padding required.
                theta_target = b_vecs[start_index:stop_index]  # [context_len, theta_dim]
                batch_theta_targets.append(theta_target)

            batch_features = torch.stack(batch_features, dim=0)
            batch_states = torch.stack(batch_states, dim=0)
            batch_controls = torch.stack(batch_controls, dim=0)
            batch_goals = torch.stack(batch_goals, dim=0)
            batch_theta_targets = torch.stack(batch_theta_targets, dim=0)

            # optimizer.zero_grad()
            # theta_pred = model(batch_features)

            # batch_nll_terms = []
            # for time_index in range(1,context_len):
            #     theta_t = theta_pred[:, time_index-1, :]
            #     nll_rows = []
            #     for sample_index in range(theta_t.shape[0]):
            #         B_human = torch.diag(theta_t[sample_index]) * delta_t
            #         P = DARE_SOLVER(A_human, B_human, Q_human, R_human)
            #         S = R_human + B_human.T @ P @ B_human
            #         K = torch.linalg.solve(S, B_human.T @ P @ A_human)
            #         precision = 2.0 * S

            #         mu = -K @ (batch_states[sample_index, time_index] - batch_goals[sample_index, time_index])
            #         residual = batch_controls[sample_index, time_index] - mu
            #         quadratic_term = (residual @ precision) @ residual
            #         _, logdet_precision = torch.linalg.slogdet(precision)
            #         action_dim = batch_controls.shape[-1]

            #         nll = (
            #             0.5 * quadratic_term
            #             - 0.5 * logdet_precision
            #             + 0.5 * action_dim * np.log(2.0 * np.pi)
            #         )
            #         nll_rows.append(nll)
            #     batch_nll_terms.append(torch.stack(nll_rows))

            # nll_loss = torch.stack(batch_nll_terms, dim=1).mean()
            # theta_mse = torch.mean((theta_pred - batch_theta_targets) ** 2)

            # # some theta-smoothness loss
            # if context_len > 1:
            #     theta_velocity = theta_pred[:, 1:, :] - theta_pred[:, :-1, :]
            #     theta_smoothness_loss = torch.mean(theta_velocity ** 2)
            # else:
            #     theta_smoothness_loss = torch.zeros((),dtype=DTYPE, device=device)
            # if context_len > 2:
            #     theta_acceleration = theta_pred[:, 2:, :] - 2*theta_pred[:, 1:-1, :] + theta_pred[:, :-2, :]
            #     theta_acceleration_loss = torch.mean(theta_acceleration ** 2)
            # else:
            #     theta_acceleration_loss = torch.zeros((),dtype=DTYPE, device=device)

            # # Optional debugging mode. This is not part of the Tian formulation,
            # # but it helps verify that the transformer can represent the synthetic
            # # latent b trajectory before testing likelihood-only identifiability.
            # if use_theta_supervision:
            #     loss = supervised_theta_weight * theta_mse
            # else:
            #     loss = nll_loss #+ theta_smoothness_loss * 10 #+ theta_acceleration_loss * 0.1


            # loss.backward()
            # optimizer.step()

            optimizer.zero_grad()

            # theta_pred[:, t, :] = theta_{t+1}: parameter predicted after observing step t.
            theta_pred = model(batch_features)  # [B, context_len, theta_dim]

            # theta_aligned[:, t, :] = theta_t: the parameter that should govern u^t.
            # get_theta_sequence() prepends the learned theta_0 and drops the last column,
            # shifting the sequence one step earlier without any gradient penalty.
            theta_aligned = model.get_theta_sequence(theta_pred)  # [B, context_len, theta_dim]

            batch_nll_terms = []
            for time_index in range(context_len):
                # CORRECTED: use theta_aligned (= theta_t) rather than theta_pred (= theta_{t+1}).
                # This evaluates NLL(u^t | x^t; theta_t) as required by Tian et al. Eq. (5).
                theta_t = theta_aligned[:, time_index, :]  # [B, theta_dim]
                nll_rows = []
                for sample_index in range(theta_t.shape[0]):
                    B_human = torch.diag(theta_t[sample_index]) * delta_t
                    P = DARE_SOLVER(A_human, B_human, Q_human, R_human)
                    S = R_human + B_human.T @ P @ B_human
                    K = torch.linalg.solve(S, B_human.T @ P @ A_human)
                    precision = 2.0 * S

                    mu = -K @ (
                        batch_states[sample_index, time_index]
                        - batch_goals[sample_index, time_index]
                    )
                    residual = batch_controls[sample_index, time_index] - mu
                    quadratic_term = (residual @ precision) @ residual
                    _, logdet_precision = torch.linalg.slogdet(precision)
                    action_dim = batch_controls.shape[-1]

                    nll = (
                        0.5 * quadratic_term
                        - 0.5 * logdet_precision
                        + 0.5 * action_dim * np.log(2.0 * np.pi)
                    )
                    nll_rows.append(nll)
                batch_nll_terms.append(torch.stack(nll_rows))

            nll_loss = torch.stack(batch_nll_terms, dim=1).mean()

            # CORRECTED: compare theta_aligned (= theta_t) against ground-truth b_vecs[t].
            # The original compared theta_pred (= theta_{t+1}) against b_vecs[t+1], which
            # was consistent internally but misaligned with the NLL indexing.
            theta_mse = torch.mean((theta_aligned - batch_theta_targets) ** 2)

            # Smoothness regularization on the aligned sequence theta_0..theta_{T-1}.
            if context_len > 1:
                theta_velocity = theta_aligned[:, 1:, :] - theta_aligned[:, :-1, :]
                theta_smoothness_loss = torch.mean(theta_velocity ** 2)
            else:
                theta_smoothness_loss = torch.zeros((), dtype=DTYPE, device=device)

            if context_len > 2:
                theta_acceleration = (
                    theta_aligned[:, 2:, :]
                    - 2 * theta_aligned[:, 1:-1, :]
                    + theta_aligned[:, :-2, :]
                )
                theta_acceleration_loss = torch.mean(theta_acceleration ** 2)
            else:
                theta_acceleration_loss = torch.zeros((), dtype=DTYPE, device=device)
            
            theta_magnitude_loss = torch.mean(theta_aligned ** 2)

            if use_theta_supervision:
                loss = supervised_theta_weight * theta_mse
            else:
                loss = nll_loss + theta_magnitude_weight * theta_magnitude_loss

            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach())
            epoch_nll += float(nll_loss.detach())
            epoch_theta_mse += float(theta_mse.detach())
            num_batches += 1

        row = (
            epoch,
            epoch_loss / num_batches,
            epoch_nll / num_batches,
            epoch_theta_mse / num_batches,
        )
        loss_history.append(row)

        if writer is not None:
            writer.add_scalar("loss/total", row[1], epoch)
            writer.add_scalar("loss/policy_nll", row[2], epoch)
            writer.add_scalar("diagnostic/mse", row[3], epoch)
            #writer.add_scalar("diagnostic/theta_mse_on_synthetic_gt", row[3], epoch)
            #writer.add_scalar("diagnostic/bx_pred_batch_mean", theta_pred[:, :, 0].mean().item(), epoch)
            #writer.add_scalar("diagnostic/by_pred_batch_mean", theta_pred[:, :, 1].mean().item(), epoch)
            writer.add_histogram("diagnostic/bx_pred_batch", theta_pred[:, :, 0].detach().cpu(), epoch)
            writer.add_histogram("diagnostic/by_pred_batch", theta_pred[:, :, 1].detach().cpu(), epoch)
            writer.add_scalar("diagnostic/theta_smoothness_loss", theta_smoothness_loss.item(), epoch)
            writer.add_scalar("diagnostic/theta_magnitude_loss", theta_magnitude_loss.item(), epoch)

            if epoch % tensorboard_eval_every == 0 or epoch == num_epochs - 1:
                log_theta_trajectory_diagnostics_to_tensorboard(
                    writer=writer,
                    model=model,
                    rollout_tensors=rollout_tensors,
                    filenames=filenames,
                    context_len=context_len,
                    epoch=epoch,
                )

        if epoch % print_every == 0 or epoch == num_epochs - 1:
            print(
                f"epoch={epoch:04d}  "
                f"loss={row[1]:.6f}  nll={row[2]:.6f}  theta_mse={row[3]:.6f} "
                f"theta_x_mean:{theta_pred[:, :, 0].mean().item():.6f} "
                f"theta_y_mean:{theta_pred[:, :, 1].mean().item():.6f} "
            )

    if plot:
        plt.figure(figsize=(12, 4))
        plt.subplot(2, 1, 1)
        plt.plot([row[0] for row in loss_history], [row[1] for row in loss_history], label="total loss")
        plt.plot([row[0] for row in loss_history], [row[2] for row in loss_history], label="policy NLL")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Transformer Human-Dynamics Training Loss")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot([row[0] for row in loss_history], [row[3] for row in loss_history])
        plt.xlabel("Epoch")
        plt.ylabel("Theta MSE")
        plt.title("Auxiliary Theta Prediction Error")

        plt.tight_layout()
        plt.show()

    if writer is not None:
        writer.flush()
        writer.close()
    return model, loss_history

def recover_human_b_from_data(filename, A_human_init, B_human_init, Q_human, R_human, A_env, B_env, delta_t, learning_rate):

    # load the csv data from filename
    states = []
    next_states = []
    controls = []
    goals = []
    print_every = 10
    num_iters = 5000
    plot = True

    with open(filename, mode="r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            states.append([float(row["x"]), float(row["y"])])
            goals.append([float(row["goal_x"]), float(row["goal_y"])])
            next_states.append([float(row["x_next"]), float(row["y_next"])])
            controls.append([float(row["u_x"]), float(row["u_y"])])

    states = torch.tensor(states, dtype=DTYPE)
    next_states = torch.tensor(next_states, dtype=DTYPE)
    controls = torch.tensor(controls, dtype=DTYPE)
    goals = torch.tensor(goals, dtype=DTYPE)

    A_human = A_human_init

    if B_human_init.ndim == 2:
        initial_b_vec = torch.diag(B_human_init)
    else:
        initial_b_vec = B_human_init

    B_human_vec = torch.nn.Parameter(initial_b_vec)
    optimizer = torch.optim.Adam([B_human_vec], lr=learning_rate)

    loss_history = []
    b_history = []

    for iteration in range(num_iters):
        optimizer.zero_grad()

        # compute the human control policy
        B_human = torch.diag(B_human_vec)*delta_t
        P = DARE_SOLVER(A_human, B_human, Q_human, R_human)
        S = R_human + B_human.T @ P @ B_human
        K = torch.linalg.solve(S, B_human.T @ P @ A_human)
        precision = 2.0 * S
        Mu = -(states - goals) @ K.T
        residuals = controls - Mu
        quadratic_terms = torch.sum((residuals @ precision) * residuals, dim=1)
        _, logdet_precision = torch.linalg.slogdet(precision)
        action_dim = controls.shape[1]

        nll = (
             0.5 * quadratic_terms
             - 0.5 * logdet_precision
             + 0.5 * action_dim * np.log(2.0 * np.pi)
         )
        nll_sum = nll.sum()
        nll_sum.backward() # backprop
        optimizer.step()

        row = (
            iteration,
            float(B_human_vec[0].detach()),
            float(B_human_vec[1].detach()),
            float(nll_sum.detach())
        )
        loss_history.append(row)

        if iteration % print_every == 0 or iteration == num_iters - 1:
            print(
                f"step={iteration:04d}  "
                f"bx_hat={row[1]:.6f}  by_hat={row[2]:.6f}  "
                f"nll={row[3]:.3f}"
            )

    if plot:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot([row[0] for row in loss_history], [row[1] for row in loss_history], label="b_x")
        plt.plot([row[0] for row in loss_history], [row[2] for row in loss_history], label="b_y")
        plt.xlabel("Iteration")
        plt.ylabel("Estimated b values")
        plt.title("Estimated B_human_vec Components Over Iterations")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot([row[0] for row in loss_history], [row[3] for row in loss_history])
        plt.xlabel("Iteration")
        plt.ylabel("Negative Log Likelihood")
        plt.title("Negative Log Likelihood Over Iterations")

        plt.tight_layout()
        plt.show()

    return loss_history

def log_theta_trajectory_diagnostics_to_tensorboard(
    writer, model, rollout_tensors, filenames, context_len, epoch
):
    if writer is None:
        return

    model.eval()

    with torch.no_grad():
        true_bx_segments = []
        true_by_segments = []
        pred_bx_segments = []
        pred_by_segments = []
        rollout_boundaries = []
        rollout_midpoints = []
        rollout_labels = []

        global_start = 0

        for rollout_index, (states, next_states, controls, goals, b_vecs) in enumerate(rollout_tensors):
            seq_len = min(context_len, controls.shape[0])

            features = torch.cat(
                [states[:seq_len], next_states[:seq_len], controls[:seq_len]],
                dim=1,
            ).unsqueeze(0)

            theta_pred = model(features).squeeze(0)

            # CSV row k stores b_k, the belief associated with control u_k.
            # Therefore the plotted target should align directly with theta_pred[k].
            true_b = b_vecs[:seq_len]

            true_bx_segments.append(true_b[:, 0].detach().cpu().numpy())
            true_by_segments.append(true_b[:, 1].detach().cpu().numpy())
            pred_bx_segments.append(theta_pred[:, 0].detach().cpu().numpy())
            pred_by_segments.append(theta_pred[:, 1].detach().cpu().numpy())

            global_stop = global_start + seq_len
            rollout_boundaries.append(global_stop)
            rollout_midpoints.append(global_start + 0.5 * seq_len)
            rollout_labels.append(str(rollout_index))
            global_start = global_stop

        if len(true_bx_segments) == 0:
            model.train()
            return

        true_bx_concat = np.concatenate(true_bx_segments)
        true_by_concat = np.concatenate(true_by_segments)
        pred_bx_concat = np.concatenate(pred_bx_segments)
        pred_by_concat = np.concatenate(pred_by_segments)
        global_time = np.arange(true_bx_concat.shape[0])

        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(global_time, true_bx_concat, label="true b_x", linewidth=1.5)
        ax1.plot(global_time, pred_bx_concat, "--", label="predicted b_x", linewidth=1.5)
        ax1.set_ylabel("b_x")
        ax1.legend(loc="upper right")
        ax1.grid(True)

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(global_time, true_by_concat, label="true b_y", linewidth=1.5)
        ax2.plot(global_time, pred_by_concat, "--", label="predicted b_y", linewidth=1.5)
        ax2.set_ylabel("b_y")
        ax2.set_xlabel("Concatenated timestep")
        ax2.legend(loc="upper right")
        ax2.grid(True)

        # Delineate rollout boundaries and lightly shade alternating rollouts.
        segment_start = 0
        for segment_index, boundary in enumerate(rollout_boundaries):
            if segment_index % 2 == 1:
                ax1.axvspan(segment_start, boundary, alpha=0.08)
                ax2.axvspan(segment_start, boundary, alpha=0.08)

            if boundary < global_time[-1] + 1:
                ax1.axvline(boundary - 0.5, linestyle=":", linewidth=1.0)
                ax2.axvline(boundary - 0.5, linestyle=":", linewidth=1.0)

            segment_start = boundary

        if len(rollout_midpoints) <= 30:
            ax2.set_xticks(rollout_midpoints)
            ax2.set_xticklabels(rollout_labels, rotation=0)
            ax2.set_xlabel("Rollout index")

        fig.suptitle("Concatenated theta trajectory diagnostics")
        fig.tight_layout()

        writer.add_figure("theta_trajectory/concatenated_rollouts", fig, epoch)
        writer.add_scalar(
            "theta_eval/concatenated_rollouts/mse",
            np.mean((pred_bx_concat - true_bx_concat) ** 2 + (pred_by_concat - true_by_concat) ** 2),
            epoch,
        )
        writer.add_scalar(
            "theta_eval/concatenated_rollouts/bx_mse",
            np.mean((pred_bx_concat - true_bx_concat) ** 2),
            epoch,
        )
        writer.add_scalar(
            "theta_eval/concatenated_rollouts/by_mse",
            np.mean((pred_by_concat - true_by_concat) ** 2),
            epoch,
        )

        plt.close(fig)

    model.train()

if __name__ == "__main__":

    A_human = torch.tensor([[1.0,0.0],[0.0,1.0]], dtype=DTYPE)
    B_human_vec = torch.tensor([0.7,0.7], dtype=DTYPE)
    Q_human = torch.eye(2, dtype=DTYPE)
    R_human = 0.01 * torch.eye(2, dtype=DTYPE)
    A_env = torch.tensor([[1.0,0.0],[0.0,1.0]], dtype=DTYPE)
    B_env = torch.tensor([[0.2,0.0],[0.0,0.2]], dtype=DTYPE)
    dt = 0.2
    total_time = 30.0

    # Options: "auto", "mps", or "cpu".
    training_device = "cpu"

    tensorboard_log_dir = "runs/learn_human_dynamics_version_a"
    use_theta_supervision = False
    theta_supervision_weight = 1.0

    rollout_filenames = []
    num_rollouts = 50

    #mode = "generate_rollouts"
    #mode = "recover_b"
    mode = "train transformer"
    
    if mode == "generate_rollouts":
        for rollout_index in range(num_rollouts):
            # Generate an independent rollout. Clone the initial B vector so each
            # rollout starts from the same nominal initial human model rather than
            # accidentally sharing a mutable tensor across simulations.

            if rollout_index == 1:
                plot_flag = True
            else:
                plot_flag = False

            time_history, state_history, next_state_history, control_history, goal_history, b_history = human_environment_simulation(
                A_human=A_human,
                B_human_vec=B_human_vec.clone(),
                Q_human=Q_human,
                R_human=R_human,
                A_env=A_env,
                B_env=B_env,
                dt=dt,
                total_time=total_time,
                gradient_learner=True,
                plot=plot_flag
            )

            rollout_filename = f"gradient_learner_rollouts/gradient_learner_rollout_{rollout_index:02d}.csv"
            rollout_filenames.append(rollout_filename)

            save_gradient_learner_trajectory_csv(
                filename=rollout_filename,
                time_history=time_history,
                state_history=state_history,
                next_state_history=next_state_history,
                control_history=control_history,
                goal_history=goal_history,
                b_vec_history=b_history,
                dt=dt,
            )

            if rollout_index == 0:
                plot_trajectory(
                    state_history_np=np.array(state_history),
                    targets_np=np.array(goal_history),
                    control_history_np=np.array(control_history),
                )

    if mode == "recover_b":
        rollout_filenames = glob.glob("gradient_learner_rollouts/gradient_learner_rollout_*.csv")
        print(f"Found rollout files: {rollout_filenames}")
        recover_human_b_from_data(
            filename=rollout_filenames[1],
            A_human_init=A_human,
            B_human_init=torch.tensor([0.5, 0.5], dtype=DTYPE),
            Q_human=Q_human,
            R_human=R_human,
            A_env=A_env,
            B_env=B_env,
            delta_t=dt,
            learning_rate=0.001,
        )

    if mode == "train transformer":
        rollout_filenames = glob.glob("gradient_learner_rollouts/gradient_learner_rollout_*.csv")
        print(f"Found rollout files: {rollout_filenames}")
        transformer_model, transformer_loss_history = learn_human_dynamics(
            filenames=rollout_filenames[:10],
            A_human=A_human,
            Q_human=Q_human,
            R_human=R_human,
            delta_t=dt,
            context_len=150,
            num_epochs=50,
            learning_rate=1e-3,
            batch_size=4,
            supervised_theta_weight=theta_supervision_weight,
            use_theta_supervision=use_theta_supervision,
            device=training_device,
            tensorboard_log_dir=tensorboard_log_dir,
            tensorboard_eval_every=10,
        )

        #save the model state dict
        torch.save(transformer_model.state_dict(), "human_dynamics_transformer.pth")
