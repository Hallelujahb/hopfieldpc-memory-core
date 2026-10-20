"""Energy-consistent coupling of Hopfield memory with predictive-coding (PC) inference.

The Hopfield memory is added as one more term of the energy that PC inference
already descends, instead of a module that overwrites the hidden state:

    E_total(z) = E_PC(z) + lam * E_H(z; M, beta)

The gradient of the full modern Hopfield energy is

    dE_H/dz = z - softmax(beta * z M^T) M = z - retrieved(z)

so the latent update becomes

    z <- z - eta * ( dE_PC/dz + lam * (z - retrieved(z)) )

lam = 0 is plain PC. With no PC term and eta * lam = 1 the update is exactly
one-step Hopfield retrieval.

E_H must be the full energy (hopfield_full_energy). The default hopfield_energy
leaves out the q-dependent term 1/2 |q|^2 and is not the energy that retrieval
descends.

Stability
---------
The Hessian of E_H is I - beta * Cov_p(m), where Cov_p is the covariance of the
memory vectors under the softmax weights p and |m_i| <= M_max. Cov_p is positive
semidefinite and bounded by M_max^2, so the eigenvalues lie in
[1 - beta * M_max^2, 1]. The upper curvature of the Hopfield term is exactly 1
for every beta and every memory. Monotone descent needs only the upper
curvature, so if E_PC has largest Hessian eigenvalue L_PC, gradient descent on
E_PC + lam * E_H decreases the energy at every step when

    eta < 2 / (L_PC + lam)        (eta <= 1 / (L_PC + lam) is the conservative choice)

beta and the memory norm only set the negative curvature (how deep and separated
the valleys are), not the step-size limit.

Scores
------
score="dot" is standard modern Hopfield retrieval, similarity q . m_i.
score="l2" adds the bias -1/2 |m_i|^2 to every similarity, which ranks memories
by Euclidean distance instead. Dot-product scoring favours memories with a large
norm; the l2 score does not. Both give the gradient z - retrieved(z) and the same
Hessian, so the step-size rule is unchanged. The l2 energy is returned without
its additive constants.

Assumes a fixed memory and float32 or float64 inputs.
"""

from __future__ import annotations

from typing import Callable

import torch

from .api import validate_beta, validate_query_memory
from .energy import hopfield_full_energy
from .functional import hopfield_retrieve

SCORES = ("dot", "l2")


def _check_score(score: str) -> None:
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}, got {score!r}.")


def _l2_similarities(query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    return torch.matmul(query, memory.transpose(0, 1)) - 0.5 * (memory**2).sum(dim=-1)


def hopfield_retrieve_scored(
    query: torch.Tensor, memory: torch.Tensor, beta: float = 1.0, *, score: str = "dot"
) -> torch.Tensor:
    """One-step retrieval with the chosen score. Returns retrieved [batch, dim]."""

    _check_score(score)
    if score == "dot":
        return hopfield_retrieve(query, memory, beta=beta, normalize=False)[0]
    beta = validate_beta(beta)
    validate_query_memory(query=query, memory=memory)
    sim = _l2_similarities(query, memory)
    sim = sim - sim.amax(dim=-1, keepdim=True)
    return torch.matmul(torch.softmax(beta * sim, dim=-1), memory)


def hopfield_score_energy(
    query: torch.Tensor, memory: torch.Tensor, beta: float = 1.0, *, score: str = "dot"
) -> torch.Tensor:
    """Per-query energy that hopfield_retrieve_scored descends, shape [batch]."""

    _check_score(score)
    if score == "dot":
        return hopfield_full_energy(query, memory, beta)
    beta = validate_beta(beta)
    validate_query_memory(query=query, memory=memory)
    sim = _l2_similarities(query, memory)
    top = sim.amax(dim=-1, keepdim=True)
    log_sum = torch.logsumexp(beta * (sim - top), dim=-1)
    return -top.squeeze(-1) - log_sum / beta + 0.5 * (query**2).sum(dim=-1)


def hopfield_energy_grad(
    query: torch.Tensor, memory: torch.Tensor, beta: float = 1.0, *, score: str = "dot"
) -> torch.Tensor:
    """Analytic gradient of the Hopfield energy: dE_H/dq = q - retrieved(q)."""

    return query - hopfield_retrieve_scored(query, memory, beta, score=score)


def hopfield_curvature_bounds(memory: torch.Tensor, beta: float) -> tuple[float, float]:
    """Eigenvalue range [lower, upper] of the Hessian of the Hopfield energy.

    lower = 1 - beta * M_max^2 (negative curvature, the valleys), upper = 1.
    """

    beta = validate_beta(beta)
    if memory.ndim != 2:
        raise ValueError(f"memory must be [num_memories, dim], got {tuple(memory.shape)}.")
    max_sq_norm = float((memory**2).sum(dim=-1).max())
    return 1.0 - beta * max_sq_norm, 1.0


def hopfield_smoothness(memory: torch.Tensor, beta: float) -> float:
    """Lipschitz constant of the energy gradient: max(|lower|, upper) of the curvature range.

    The monotone-descent step rule does not need this (it uses the upper curvature
    only). It is useful when an absolute gradient-Lipschitz constant is required.
    """

    lower, upper = hopfield_curvature_bounds(memory, beta)
    return max(abs(lower), abs(upper))


def safe_step_size(l_pc: float, lam: float, *, conservative: bool = True) -> float:
    """Step size for monotone decrease of E_PC + lam * E_H.

    Args:
        l_pc: largest Hessian eigenvalue of E_PC.
        lam: Hopfield weight (>= 0). The Hopfield term adds exactly lam.
        conservative: True returns 1 / (l_pc + lam), False returns the sharp limit
            2 / (l_pc + lam), below which the energy decreases.
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
    """Largest Hessian eigenvalue of E_PC at z, by power iteration on Hessian-vector products.

    Works for any differentiable energy. The estimate is local: repeat it if z moves
    a lot. Batch samples are independent, so the result is a maximum over samples.
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
        norm = hv.norm()
        if norm == 0:
            return 0.0
        v = hv / norm
    return eig


def joint_energy(
    pc_energy: torch.Tensor,
    z: torch.Tensor,
    memory: torch.Tensor,
    beta: float,
    lam: float,
    *,
    score: str = "dot",
) -> torch.Tensor:
    """E_total = E_PC + lam * E_H, per sample."""

    return pc_energy + lam * hopfield_score_energy(z, memory, beta, score=score)


def pull_step(
    z: torch.Tensor,
    pc_grad: torch.Tensor,
    memory: torch.Tensor,
    beta: float,
    lam: float,
    eta: float,
    *,
    score: str = "dot",
) -> torch.Tensor:
    """One joint update: z - eta * (pc_grad + lam * (z - retrieved(z)))."""

    return z - eta * (pc_grad + lam * hopfield_energy_grad(z, memory, beta, score=score))


def settle(
    z0: torch.Tensor,
    pc_energy_fn: Callable[[torch.Tensor], torch.Tensor],
    memory: torch.Tensor,
    *,
    beta: float,
    lam: float,
    eta: float,
    steps: int,
    score: str = "dot",
) -> dict[str, object]:
    """Run joint PC + Hopfield inference on the hidden state z.

    Args:
        z0: initial hidden state [batch, dim], for example the forward-pass value.
        pc_energy_fn: maps z [batch, dim] to the PC energy per sample [batch]. The PC
            gradient comes from autograd, so any differentiable energy works.
        memory: Hopfield memory [N, dim].
        beta, lam, eta, steps, score: coupling and step settings.

    Returns:
        dict with the final state "z" and per-step traces "pc", "hopfield" and
        "total" (lists of [batch] tensors with steps + 1 entries).
    """

    z = z0.detach().clone()
    traces: dict[str, list[torch.Tensor]] = {"pc": [], "hopfield": [], "total": []}

    def record(state: torch.Tensor) -> None:
        with torch.no_grad():
            e_pc = pc_energy_fn(state)
            e_h = hopfield_score_energy(state, memory, beta, score=score)
        traces["pc"].append(e_pc)
        traces["hopfield"].append(e_h)
        traces["total"].append(e_pc + lam * e_h)

    record(z)
    for _ in range(steps):
        x = z.detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(pc_energy_fn(x).sum(), x)
        z = pull_step(z, grad, memory, beta, lam, eta, score=score).detach()
        record(z)

    return {"z": z, **traces}
