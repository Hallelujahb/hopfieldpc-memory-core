"""Energy-consistent coupling of Hopfield memory with predictive-coding (PC) inference.

Idea
----
Do not add the Hopfield term as a separate module that overwrites the hidden
state. Add it as one more term of the *same energy* that PC inference descends:

    E_total(z) = E_PC(z) + lam * E_H(z; M, beta)

The gradient of the (full) modern Hopfield energy is

    dE_H/dz = z - softmax(beta * z M^T) M = z - retrieved(z)

so the PC latent update becomes

    z <- z - eta * ( dE_PC/dz + lam * (z - retrieved(z)) )

With lam = 0 this is plain PC. With eta * lam = 1 and no PC term it is exactly
one-step Hopfield retrieval.

Stability
---------
The Hessian of E_H is I - beta * Cov_p(m), where Cov_p is the covariance of the
memory vectors under the softmax weights p and ||m_i|| <= M_max. Since Cov_p is
positive semidefinite and bounded by M_max^2, its eigenvalues lie in

    [1 - beta * M_max^2,  1].

The UPPER curvature of the Hopfield energy is therefore exactly 1, for every
beta and every memory. Monotone descent (the descent lemma) needs only the
upper curvature, so if E_PC has upper curvature L_PC (largest Hessian
eigenvalue), gradient descent on E_PC + lam * E_H decreases the energy at every
step whenever

    eta < 2 / (L_PC + lam)          (eta <= 1 / (L_PC + lam) is conservative).

beta and the memory norm only control the NEGATIVE curvature (how non-convex
the landscape is: how deep and separated the valleys are), not step-size
stability. Verified numerically in benchmarks/pc_hopfield_toy.py: stable up to
~0.95x the limit, divergent at ~1.05x.

Note: everything here assumes retrieval WITHOUT score normalization
(normalize=False), because only then does retrieval match the energy.
"""

from __future__ import annotations

from typing import Callable

import torch

from .api import validate_beta, validate_query_memory
from .energy import hopfield_energy
from .functional import hopfield_retrieve


def hopfield_energy_grad(
    query: torch.Tensor, memory: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    """Analytic gradient of the full Hopfield energy: dE_H/dq = q - retrieved(q)."""

    retrieved, _ = hopfield_retrieve(query, memory, beta=beta, normalize=False)
    return query - retrieved


def hopfield_curvature_bounds(memory: torch.Tensor, beta: float) -> tuple[float, float]:
    """Eigenvalue range [lower, upper] of the Hessian of the full Hopfield energy.

    lower = 1 - beta * M_max^2 (negative curvature: valleys), upper = 1.
    """

    beta = validate_beta(beta)
    if memory.ndim != 2:
        raise ValueError(f"memory must be [num_memories, dim], got {tuple(memory.shape)}.")
    max_sq_norm = float((memory**2).sum(dim=-1).max())
    return 1.0 - beta * max_sq_norm, 1.0


def hopfield_smoothness(memory: torch.Tensor, beta: float) -> float:
    """Lipschitz constant of the Hopfield energy gradient: max(1, beta*M_max^2 - 1).

    Not needed for the monotone-descent step rule (that uses the upper curvature
    1 only); useful when you need an absolute gradient-Lipschitz constant.
    """

    lower, upper = hopfield_curvature_bounds(memory, beta)
    return max(abs(lower), abs(upper))


def safe_step_size(l_pc: float, lam: float, *, conservative: bool = True) -> float:
    """Step size for monotone decrease of E_PC + lam * E_H.

    Args:
        l_pc: upper curvature (largest Hessian eigenvalue) of E_PC.
        lam: Hopfield weight (>= 0); the Hopfield term adds exactly lam.
        conservative: True returns 1/(l_pc + lam); False returns the sharp
            limit 2/(l_pc + lam) (decrease only for eta strictly below it).
    """

    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam!r}.")
    if l_pc <= 0:
        raise ValueError(f"l_pc must be positive, got {l_pc!r}.")
    return (1.0 if conservative else 2.0) / (float(l_pc) + lam)


def estimate_top_curvature(
    pc_energy_fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    *,
    iters: int = 30,
    seed: int = 0,
) -> float:
    """Estimate the largest-magnitude Hessian eigenvalue of E_PC at z (power iteration).

    Uses Hessian-vector products, so it works for any differentiable energy
    (use it to get l_pc for a nonlinear network). It is a LOCAL estimate at z:
    re-estimate if z moves a lot. Batch samples are independent, so the result
    is the maximum over samples.
    """

    g = torch.Generator().manual_seed(seed)
    v = torch.randn(z.shape, generator=g, dtype=z.dtype)
    v = v / v.norm()
    x = z.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(pc_energy_fn(x).sum(), x, create_graph=True)
    eig = 0.0
    for _ in range(iters):
        (hv,) = torch.autograd.grad(grad, x, grad_outputs=v, retain_graph=True)
        eig = float((hv * v).sum())
        n = hv.norm()
        if n == 0:
            return 0.0
        v = hv / n
    return eig


def joint_energy(
    pc_energy: torch.Tensor,
    z: torch.Tensor,
    memory: torch.Tensor,
    beta: float,
    lam: float,
) -> torch.Tensor:
    """E_total = E_PC + lam * E_H (per sample, full energy with constants)."""

    return pc_energy + lam * hopfield_energy(z, memory, beta, include_constants=True)


def pull_step(
    z: torch.Tensor,
    pc_grad: torch.Tensor,
    memory: torch.Tensor,
    beta: float,
    lam: float,
    eta: float,
) -> torch.Tensor:
    """One joint update: z - eta * (pc_grad + lam * (z - retrieved(z)))."""

    validate_query_memory(z, memory)
    return z - eta * (pc_grad + lam * hopfield_energy_grad(z, memory, beta))


def settle(
    z0: torch.Tensor,
    pc_energy_fn: Callable[[torch.Tensor], torch.Tensor],
    memory: torch.Tensor,
    *,
    beta: float,
    lam: float,
    eta: float,
    steps: int,
) -> dict[str, object]:
    """Run joint PC + Hopfield inference on the hidden state z.

    Args:
        z0: initial hidden state [batch, dim] (e.g. from the forward pass).
        pc_energy_fn: maps z [batch, dim] to the PC energy per sample [batch].
            The PC gradient is taken with autograd, so any differentiable
            energy works (including one built from FabricPC nodes).
        memory: Hopfield memory [N, dim].
        beta, lam, eta, steps: coupling and step settings.

    Returns:
        dict with final state "z" and per-step traces (lists of [batch] tensors)
        "pc", "hopfield", "total" of length steps + 1.
    """

    z = z0.detach().clone()
    traces: dict[str, list[torch.Tensor]] = {"pc": [], "hopfield": [], "total": []}

    def record(state: torch.Tensor) -> None:
        with torch.no_grad():
            e_pc = pc_energy_fn(state)
            e_h = hopfield_energy(state, memory, beta, include_constants=True)
        traces["pc"].append(e_pc)
        traces["hopfield"].append(e_h)
        traces["total"].append(e_pc + lam * e_h)

    record(z)
    for _ in range(steps):
        x = z.detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(pc_energy_fn(x).sum(), x)
        z = pull_step(z, grad, memory, beta, lam, eta).detach()
        record(z)

    return {"z": z, **traces}
