import math
import torch
from torch import nn, Tensor
import lightning as L
from typing import Tuple

from .transformer import HIMTransfomerNet
from .riccati import dare as DARE


class TeleopHIM(L.LightningModule):
    def __init__(
            self,
            d_model: int,
            d_out: int,
            d_hid: int = 128,
            n_heads: int = 2,
            n_layers: int = 2,
            dropout: float = 0.5,):
        super().__init__()

        # Initialize models
        self.model = HIMTransfomerNet(
            d_model=d_model,
            d_out=d_out,
            d_hid=d_hid,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        self.dare = DARE()

        # Register inital buffers
        N = d_out // 2
        self.register_buffer('B_0', torch.eye(
            N) * 0.001)         # (N, N)
        self.register_buffer('A', torch.eye(N).unsqueeze(
            0).unsqueeze(1))    # (1, 1, N, N)
        self.register_buffer('Q', torch.eye(N).unsqueeze(
            0).unsqueeze(1) * 5)    # (1, 1, N, N)
        self.register_buffer('R', torch.eye(N).unsqueeze(
            0).unsqueeze(1) * 0.1)    # (1, 1, N, N)

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        states, actions, states_next = x
        batch_size, seq_len, _ = states.shape
        data = torch.cat([states, actions, states_next],
                         dim=-1)    # (B, T, D)
        
        # Predict theta and unpack
        theta = self.model(data)
        B, W = torch.chunk(theta, 2, dim=-1)

        # Constrains
        B = torch.tanh(torch.abs(B))    # positive and small
        W = torch.tanh(W)               # small

        # Stack B_0
        B = torch.diag_embed(B)
        B_0 = self.B_0.expand(batch_size, 1, *B.shape[-2:])
        # (B, T+1, N, N)
        B = torch.cat([B_0, B], dim=1)

        return B, W

    def training_step(self, batch, batch_idx):
        # Process batch data
        states, actions, states_next = batch
        batch_size, seq_len, _ = states.shape
        pos, goal = torch.chunk(states, 2, dim=-1)
        pos_next, goal_next = torch.chunk(states_next, 2, dim=-1)

        # Compute error state
        x = pos - goal
        x_next = pos_next - goal_next

        # Predict B and W
        B, W = self.forward(batch)

        # Get P_H via DARE
        # (B, T+1, N, N)
        P = self.dare(self.A, B, self.Q, self.R)

        # Compute optimal control
        # (B, T+1, N, N)
        K = self.lqr_K(P, self.A, B, self.Q, self.R)

        # Compute optimal control and predicted control
        u_H_star = - torch.matmul(K[:, :-1], x.unsqueeze(-1)).squeeze(-1)
        # Use predicted bias to compensate user actions
        u_H_hat = actions - torch.sign(x) * W

        # Clip u_H to be within action limits
        u_H_star = torch.clamp(u_H_star, -1.0, 1.0)
        u_H_hat = torch.clamp(u_H_hat, -1.0, 1.0)

        # Compute next state via predicted control
        x_next_star = self.dynamics(self.A, B[:, :-1], x, u_H_star)
        x_next_hat = self.dynamics(self.A, B[:, :-1], x, u_H_hat)

        # Compute Q-function values
        Q_u_H = self.value_Q(
            P[:, 1:], self.Q, self.R, x, u_H_hat, x_next_hat)   # (B, T)
        Q_u_H_star = self.value_Q(
            P[:, 1:], self.Q, self.R, x, u_H_star, x_next_star)  # (B, T)

        # Compute likelihoods loss (maximum)
        likelihoods = self.likelihood_u_H(
            P[:, 1:], B[:, 1:], self.R, Q_u_H - Q_u_H_star) # (B, T)
        loss = - likelihoods.sum(dim=-1).mean()

        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        # Process batch data
        states, actions, states_next = batch
        batch_size, seq_len, _ = states.shape
        pos, goal = torch.chunk(states, 2, dim=-1)
        pos_next, goal_next = torch.chunk(states_next, 2, dim=-1)

        # Compute error state
        x = pos - goal
        x_next = pos_next - goal_next

        # Predict B and W
        B, W = self.forward(batch)

        # Get P_H via DARE
        # (B, T+1, N, N)
        P = self.dare(self.A, B, self.Q, self.R)

        # Compute optimal control
        # (B, T+1, N, N)
        K = self.lqr_K(P, self.A, B, self.Q, self.R)

        u_H_star = - torch.matmul(K[:, :-1], x.unsqueeze(-1)).squeeze(-1)
        u_H_star += torch.sign(x) * W

        return u_H_star

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    @staticmethod
    def lqr_K(P: Tensor, A: Tensor, B: Tensor, Q: Tensor, R: Tensor) -> Tensor:
        """
        Solve the discrete-time LQR controller for a batch of systems.

        Args:
            P: (..., n, n) solution to the discrete-time Riccati equation
            A: (..., n, n) state transition matrices
            B: (..., n, m) control input matrices
            Q: (..., n, n) state cost matrices
            R: (..., m, m) control cost matrices
        Returns:
            K: (..., m, n) optimal gain matrices
        """

        BT_P = torch.matmul(B.transpose(-1, -2), P)  # (..., m, n)
        BT_P_B = torch.matmul(BT_P, B)  # (..., m, m)
        R_plus = R + BT_P_B  # (..., m, m)
        R_plus_inv = torch.linalg.inv(R_plus)  # (..., m, m)

        BT_P_A = torch.matmul(BT_P, A)  # (..., m, n)

        K = torch.matmul(R_plus_inv, BT_P_A)  # (..., m, n)

        return K

    @staticmethod
    def value_Q(P: Tensor, Q: Tensor, R: Tensor, x: Tensor, u: Tensor, x_next: Tensor) -> Tensor:
        """
        Compute the Q-function for a batch of systems.

        Args:
            P: (..., n, n) solution to the discrete-time Riccati equation
            Q: (..., n, n) state cost matrices
            R: (..., m, m) control cost matrices
            x: (..., n) states
            u: (..., m) controls
            x_next: (..., n) next states
        Returns:
            q: (...) Q-function values
        """

        x_Q = torch.matmul(x.unsqueeze(-2), Q)  # (..., 1, n)
        x_Q_x = torch.matmul(x_Q, x.unsqueeze(-1)).squeeze(-1)  # (...,)

        u_R = torch.matmul(u.unsqueeze(-2), R)  # (..., 1, m)
        u_R_u = torch.matmul(u_R, u.unsqueeze(-1)).squeeze(-1)  # (...,)

        x_next_P = torch.matmul(x_next.unsqueeze(-2), P)  # (..., 1, n)
        x_next_P_x_next = torch.matmul(
            x_next_P, x_next.unsqueeze(-1)).squeeze(-1)  # (...,)

        q = - x_Q_x - u_R_u - x_next_P_x_next  # (...,)

        return q.squeeze(-1)

    @staticmethod
    def likelihood_u_H(P: Tensor, B: Tensor, R: Tensor, delta_Q: Tensor) -> Tensor:
        """
        Compute the probability of human control input under the optimal policy.

        Args:
            P: (..., n, n) solution to the discrete-time Riccati equation
            B: (..., n, m) control input matrices
            R: (..., m, m) control cost matrices
            delta_Q: (...) difference in Q-function values
        Returns:
            prob: (...) probabilities of human control inputs
        """

        BT_P = torch.matmul(B.transpose(-1, -2), P)  # (..., m, n)
        BT_P_B = torch.matmul(BT_P, B)  # (..., m, m)

        H = 2 * R + 2 * BT_P_B  # (..., m, m)

        logdet = torch.logdet(H)

        m_h = B.shape[-1]
        const = m_h * math.log(2 * math.pi)
        
        log_prob = 0.5 * logdet - 0.5 * const + delta_Q

        return log_prob
    
    @staticmethod
    def dynamics(A: Tensor, B: Tensor, x: Tensor, u: Tensor) -> Tensor:
        """
        Compute the next state given current state and control input.

        Args:
            A: (..., n, n) state transition matrices
            B: (..., n, m) control input matrices
            x: (..., n) states
            u: (..., m) controls
        Returns:
            x_next: (..., n) next states
        """
        x_next = torch.matmul(A, x.unsqueeze(-1)).squeeze(-1) + \
            torch.matmul(B, u.unsqueeze(-1)).squeeze(-1)
        return x_next
