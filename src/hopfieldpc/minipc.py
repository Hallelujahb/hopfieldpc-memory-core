"""A small standalone predictive-coding classifier, to test Hopfield coupling before FabricPC.

Architecture (one hidden layer, matches the PC chain in the project docs):

    x --mu1 = tanh(W1 x + b1)--> z1 --mu2 = W2 z1 + b2--> z2

Training-time inference clamps x and the one-hot label y (z2 = y) and lets z1 settle:

    E(z1) = 1/2 |z1 - mu1(x)|^2 + 1/2 |y - mu2(z1)|^2

E is quadratic in z1 with Hessian I + W2^T W2, so the PC curvature L_PC = 1 + sigma_max(W2)^2
is exact and the step-size rule from coupling.py applies without estimation.
Weights are then updated with the gradient of E at the settled z1.

Test-time inference has the output free (z2 = mu2(z1)), so E(z1) = 1/2 |z1 - mu1(x)|^2 and
prediction is a feed-forward pass. Adding the Hopfield pull gives

    E_total(z1) = 1/2 |z1 - mu1(x)|^2 + lam * E_H(z1; M, beta),

whose fixed point satisfies z1* = (mu1(x) + lam * retrieved(z1*)) / (1 + lam): a convex
blend of the feed-forward state and the retrieved memory. With class-mean prototypes
as memory this is shrinkage toward class prototypes. It adds no trainable parameters.
"""

from __future__ import annotations

import torch
from torch import nn

from .coupling import safe_step_size, settle
from .readout import class_prototypes


class MiniPC(nn.Module):
    def __init__(self, in_dim: int = 784, hidden: int = 256, out_dim: int = 10) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, out_dim)
        self.hidden_dim = hidden
        self.out_dim = out_dim

    # --- node projections -------------------------------------------------
    def mu1(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.l1(x))

    def mu2(self, z1: torch.Tensor) -> torch.Tensor:
        return self.l2(z1)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward hidden state (the PC initialization z1 = mu1)."""
        return self.mu1(x)

    # --- energy ------------------------------------------------------------
    def energy(self, x: torch.Tensor, y: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((z1 - self.mu1(x)) ** 2).sum(-1) + 0.5 * ((y - self.mu2(z1)) ** 2).sum(-1)

    def curvature(self) -> float:
        """Exact largest Hessian eigenvalue of the clamped energy in z1."""
        w2 = self.l2.weight.detach()
        return 1.0 + float(torch.linalg.matrix_norm(w2, ord=2) ** 2)

    def _memory_or_dummy(self, memory: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
        return memory if memory is not None else torch.zeros(1, self.hidden_dim, dtype=like.dtype)

    # --- inference ---------------------------------------------------------
    def infer_clamped(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        steps: int,
        lam: float = 0.0,
        memory: torch.Tensor | None = None,
        beta: float = 1.0,
    ) -> dict[str, object]:
        """Training-time settling of z1 with x and y clamped (optionally with the pull)."""
        z0 = self.mu1(x).detach()
        mem = self._memory_or_dummy(memory, z0)
        eta = safe_step_size(self.curvature(), lam)
        return settle(
            z0, lambda z: self.energy(x, y, z), mem, beta=beta, lam=lam, eta=eta, steps=steps
        )

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        opt: torch.optim.Optimizer,
        *,
        steps: int,
        lam: float = 0.0,
        memory: torch.Tensor | None = None,
        beta: float = 1.0,
    ) -> float:
        z = self.infer_clamped(x, y, steps=steps, lam=lam, memory=memory, beta=beta)["z"]
        loss = self.energy(x, y, z.detach()).mean()   # weight gradient at the settled state
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.detach())

    # --- prediction --------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu2(self.mu1(x)).argmax(-1)

    def predict_pulled(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        beta: float,
        lam: float,
        steps: int = 30,
    ) -> torch.Tensor:
        """Test-time prediction with the Hopfield pull on z1 (output free)."""
        m1 = self.mu1(x).detach()
        eta = safe_step_size(1.0, lam)     # free-output energy is 1/2|z-m1|^2: curvature exactly 1
        z = settle(
            m1, lambda z: 0.5 * ((z - m1) ** 2).sum(-1), memory,
            beta=beta, lam=lam, eta=eta, steps=steps,
        )["z"]
        with torch.no_grad():
            return self.mu2(z).argmax(-1)


@torch.no_grad()
def class_hidden_prototypes(
    model: MiniPC, x: torch.Tensor, labels: torch.Tensor, num_classes: int, batch: int = 4096
) -> torch.Tensor:
    """Class-mean hidden states [num_classes, hidden] (frozen memory for the pull)."""
    h = torch.cat([model.hidden(x[i : i + batch]) for i in range(0, len(x), batch)])
    return class_prototypes(h, labels, num_classes)
