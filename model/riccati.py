import torch
import torch.nn as nn
from scipy.linalg import solve_discrete_are


def V_pert(m, n, *, device=None, dtype=None):
    """
    V_{m,n} permutation / perturbation matrix used in the paper.
    Shape: (m*n, m*n)
    """
    V = torch.zeros((m * n, m * n), device=device, dtype=dtype)
    for i in range(m * n):
        block = ((i * m) - ((i * m) % (m * n))) / (m * n)
        col = (i * m) % (m * n)
        V[i, col + round(block)] = 1
    return V


def vec(A: torch.Tensor) -> torch.Tensor:
    """
    Column-stacking vectorization.
    Input: (..., m, n)
    Output: (..., m*n, 1)
    """
    m, n = A.shape[-2], A.shape[-1]
    # columns stacked: reshape after transpose
    return A.transpose(-2, -1).reshape(*A.shape[:-2], m * n, 1)


def inv_vec(v: torch.Tensor, like_A: torch.Tensor) -> torch.Tensor:
    """
    Inverse of vec() using the shape of like_A.
    v: (..., m*n, 1) OR (..., 1, m*n) also acceptable (we'll normalize)
    like_A: (..., m, n)
    Output: (..., m, n)
    """
    m, n = like_A.shape[-2], like_A.shape[-1]
    if v.shape[-1] == 1 and v.shape[-2] == m * n:
        v_flat = v[..., :, 0]  # (..., m*n)
    elif v.shape[-2] == 1 and v.shape[-1] == m * n:
        v_flat = v[..., 0, :]  # (..., m*n)
    else:
        raise ValueError(f"inv_vec expected (...,{m*n},1) or (...,1,{m*n}), got {tuple(v.shape)}")

    # undo: columns stacked -> (n, m) then transpose back to (m, n)
    return v_flat.reshape(*v.shape[:-2], n, m).transpose(-2, -1)


def kronecker(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batched Kronecker product.
    A: (..., a, b)
    B: (..., c, d)
    Output: (..., a*c, b*d)
    """
    a, b = A.shape[-2], A.shape[-1]
    c, d = B.shape[-2], B.shape[-1]
    return torch.einsum("...ab,...cd->...acbd", A, B).reshape(*A.shape[:-2], a * c, b * d)


class Riccati(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                A: torch.Tensor, 
                B: torch.Tensor, 
                Q: torch.Tensor, 
                R: torch.Tensor):
        """
        Supports:
          A: (..., n, n)
          B: (..., n, m)
          Q: (..., n, n)
          R: (..., m, m)
        Returns:
          P: (..., n, n)
        """
        if not (A.dtype == B.dtype == Q.dtype == R.dtype):
            raise TypeError("A, B, Q, R must have the same dtype.")
        if not (A.device == B.device == Q.device == R.device):
            raise TypeError("A, B, Q, R must be on the same device.")

        n = A.shape[-1]
        if A.shape[-2:] != (n, n) or Q.shape[-2:] != (n, n):
            raise ValueError("A and Q must have shape (..., n, n).")
        if B.shape[-2] != n:
            raise ValueError("B must have shape (..., n, m).")
        m = B.shape[-1]
        if R.shape[-2:] != (m, m):
            raise ValueError("R must have shape (..., m, m).")
        # if A.shape[:-2] != B.shape[:-2] or A.shape[:-2] != Q.shape[:-2] or A.shape[:-2] != R.shape[:-2]:
        #     raise ValueError("Leading batch dims of A, B, Q, R must match.")

        # Symmetrize Q and R (batched)
        Qs = 0.5 * (Q + Q.transpose(-1, -2))
        Rs = 0.5 * (R + R.transpose(-1, -2))

        # SciPy is CPU and non-batched: flatten batch and loop
        batch_shape = torch.broadcast_shapes(
            A.shape[:-2], B.shape[:-2], Q.shape[:-2], R.shape[:-2]
        )
        batch = int(torch.tensor(batch_shape).prod()) if len(batch_shape) > 0 else 1

        A = A.expand(*batch_shape, *A.shape[-2:])
        B = B.expand(*batch_shape, *B.shape[-2:])
        Q = Q.expand(*batch_shape, *Q.shape[-2:])
        R = R.expand(*batch_shape, *R.shape[-2:])

        # symmetrize after expand is fine
        Q = 0.5 * (Q + Q.transpose(-1, -2))
        R = 0.5 * (R + R.transpose(-1, -2))

        # SciPy path: must materialize before numpy
        A2 = A.reshape(-1, n, n).contiguous()
        B2 = B.reshape(-1, n, m).contiguous()
        Q2 = Q.reshape(-1, n, n).contiguous()
        R2 = R.reshape(-1, m, m).contiguous()

        # Move to CPU for SciPy
        A_cpu = A2.detach().cpu().numpy()
        B_cpu = B2.detach().cpu().numpy()
        Q_cpu = Q2.detach().cpu().numpy()
        R_cpu = R2.detach().cpu().numpy()

        P_list = []
        for i in range(batch):
            Pi = solve_discrete_are(A_cpu[i], B_cpu[i], Q_cpu[i], R_cpu[i])
            P_list.append(torch.from_numpy(Pi))

        P = torch.stack(P_list, dim=0).to(device=A.device, dtype=A.dtype).reshape(*batch_shape, n, n)

        ctx.save_for_backward(P, A, B, Qs, Rs)
        return P

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: (..., n, n)
        returns: dA, dB, dQ, dR with same shapes as inputs
        """
        P, A, B, Q, R = ctx.saved_tensors
        device, dtype = A.device, A.dtype

        n = A.shape[-1]
        m = B.shape[-1]
        batch_shape = torch.broadcast_shapes(
            A.shape[:-2], B.shape[:-2], Q.shape[:-2], R.shape[:-2], grad_output.shape[:-2], P.shape[:-2]
        )

        A = A.expand(*batch_shape, *A.shape[-2:])
        B = B.expand(*batch_shape, *B.shape[-2:])
        Q = Q.expand(*batch_shape, *Q.shape[-2:])
        R = R.expand(*batch_shape, *R.shape[-2:])
        P = P.expand(*batch_shape, *P.shape[-2:])
        grad_output = grad_output.expand(*batch_shape, *grad_output.shape[-2:])

        # Work in double for stability, then cast back
        A_ = A.double()
        B_ = B.double()
        Q_ = Q.double()
        R_ = R.double()
        P_ = P.double()
        G  = grad_output.double()

        I_n = torch.eye(n, device=device, dtype=torch.double).expand(*batch_shape, n, n)
        I_m = torch.eye(m, device=device, dtype=torch.double).expand(*batch_shape, m, m)
        I_n2 = torch.eye(n * n, device=device, dtype=torch.double).expand(*batch_shape, n * n, n * n)
        I_m2 = torch.eye(m * m, device=device, dtype=torch.double).expand(*batch_shape, m * m, m * m)

        Vnn = V_pert(n, n, device=device, dtype=torch.double)
        Vmm = V_pert(m, m, device=device, dtype=torch.double)
        Vnn = Vnn.expand(*batch_shape, n * n, n * n)
        Vmm = Vmm.expand(*batch_shape, m * m, m * m)

        # vec(grad_output)^T : (..., 1, n^2)
        grad_vec_T = vec(G).transpose(-2, -1)  # (..., 1, n^2)

        # M3, M2, M1 (batched)
        Bt = B_.transpose(-1, -2)
        At = A_.transpose(-1, -2)

        M3 = R_ + Bt @ P_ @ B_                      # (..., m, m)
        M2 = torch.linalg.inv(M3)                   # (..., m, m)
        M1 = P_ - P_ @ B_ @ M2 @ Bt @ P_            # (..., n, n)

        # Build LHS (batched)
        LHS = kronecker(Bt, Bt)                     # (..., m*n, m*n)?? no: Bt is (m,n) so gives (..., m*m, n*n)
        LHS = kronecker(M2, M2) @ LHS               # (..., m^2, n^2)
        LHS = kronecker(P_ @ B_, P_ @ B_) @ LHS     # (..., n^2, n^2)

        LHS = LHS - kronecker(I_n[..., :, :], P_ @ B_ @ M2 @ Bt)
        LHS = LHS - kronecker(P_ @ B_ @ M2 @ Bt, I_n[..., :, :])
        LHS = LHS + I_n2

        LHS = kronecker(At, At) @ LHS
        LHS = I_n2 - LHS

        # invLHS via solve for better numerics than explicit inverse
        # We'll need invLHS @ RHS repeatedly, so solve(LHS, RHS)
        # torch.linalg.solve supports batched solve.
        # (If you prefer explicit inverse: invLHS = torch.linalg.inv(LHS))

        # ---- dA ----
        RHS = (Vnn + I_n2) @ kronecker(I_n[..., :, :], At @ M1)  # (..., n^2, n^2)
        dA_mat = torch.linalg.solve(LHS, RHS)                    # (..., n^2, n^2)
        dA_vec = grad_vec_T @ dA_mat                             # (..., 1, n^2)
        dA = inv_vec(dA_vec, A_)                                 # (..., n, n)

        # ---- dB ----
        RHS = kronecker(I_m[..., :, :], Bt @ P_)                  # (..., m^2, n^2)
        RHS = (I_m2 + Vmm) @ RHS                                  # (..., m^2, n^2)
        RHS = -kronecker(M2, M2) @ RHS                            # (..., m^2, n^2)
        RHS = -kronecker(P_ @ B_, P_ @ B_) @ RHS                   # (..., n^2, n^2)

        RHS = RHS - (I_n2 + Vnn) @ kronecker(P_ @ B_ @ M2, P_)     # (..., n^2, n^2)
        RHS = kronecker(At, At) @ RHS                              # (..., n^2, n^2)

        dB_mat = torch.linalg.solve(LHS, RHS)                      # (..., n^2, n^2)
        dB_vec = grad_vec_T @ dB_mat                               # (..., 1, n^2)
        dB = inv_vec(dB_vec, B_)                                   # (..., n, m)

        # ---- dQ ----
        RHS = I_n2
        dQ_mat = torch.linalg.solve(LHS, RHS)
        dQ_vec = grad_vec_T @ dQ_mat
        dQ = inv_vec(dQ_vec, Q_)
        dQ = 0.5 * (dQ + dQ.transpose(-1, -2))

        # ---- dR ----
        RHS = -kronecker(M2, M2)
        RHS = -kronecker(P_ @ B_, P_ @ B_) @ RHS
        RHS = kronecker(At, At) @ RHS

        dR_mat = torch.linalg.solve(LHS, RHS)
        dR_vec = grad_vec_T @ dR_mat
        dR = inv_vec(dR_vec, R_)
        dR = 0.5 * (dR + dR.transpose(-1, -2))

        return dA.to(dtype), dB.to(dtype), dQ.to(dtype), dR.to(dtype)


class dare(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B, Q, R) -> torch.Tensor:
        return Riccati.apply(A, B, Q, R)
