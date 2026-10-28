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

E_H is hopfield_energy(..., include_constants=True), the full form of the modern Hopfield
energy. hopfield_energy always carries the q-dependent term 1/2 |q|^2, so its gradient is
q - retrieved(q). The two constants only shift the value, never the gradient, so they do not
change the dynamics.

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

Scope: the rule is exact when L_PC is a true global bound on the PC Hessian (for example
a linear chain). For a nonlinear PC network L_PC is local, estimate_top_curvature gives
an estimate, and the rule is a heuristic. The rule is sufficient, not tight in general.

beta and the memory norm only set the negative curvature (how deep and separated
the valleys are), not the step-size limit.

Assumes a fixed memory and float32 or float64 inputs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from .api import validate_beta
from .diagnostics import attention_entropy, retrieval_distance, topk_mass
from .energy import hopfield_energy
from .functional import hopfield_retrieve


def hopfield_energy_grad(query: torch.Tensor, memory: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Analytic gradient of the Hopfield energy: dE_H/dq = q - retrieved(q)."""

    retrieved, _ = hopfield_retrieve(query, memory, beta=beta, normalize=False)
    return query - retrieved


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
    """Largest signed Hessian eigenvalue of E_PC at z, by power iteration on Hessian-vector products.

    Plain power iteration finds the eigenvalue of largest magnitude, which is negative when
    the energy has a strong negative direction. A first pass estimates that magnitude s, then
    a second pass runs on H + sI (positive semidefinite) and s is subtracted, so the result is
    the top eigenvalue with its sign. Works for any differentiable energy. The estimate is
    local: repeat it if z moves a lot. Batch samples are independent, so the result is a
    maximum over samples.
    """

    g = torch.Generator(device=z.device).manual_seed(seed)
    x = z.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(pc_energy_fn(x).sum(), x, create_graph=True)

    def top(shift: float) -> float:
        v = torch.randn(z.shape, generator=g, dtype=z.dtype, device=z.device)
        v = v / v.norm()
        eig = 0.0
        for _ in range(iters):
            (hv,) = torch.autograd.grad(grad, x, grad_outputs=v, retain_graph=True)
            hv = hv + shift * v
            eig = float((hv * v).sum())
            norm = hv.norm()
            if norm == 0:
                return -shift
            v = hv / norm
        return eig - shift

    return top(abs(top(0.0)))


def hopfield_trace(
    z: torch.Tensor, memory: torch.Tensor, beta: float, *, top_k: int = 3
) -> dict[str, torch.Tensor]:
    """Per-sample retrieval diagnostics at state z, each of shape [batch].

    entropy: entropy of the softmax weights over memories (log N is uniform, 0 is one memory).
    topk_mass: softmax mass on the top_k memories.
    retrieval_distance: |z - retrieved(z)|, which equals the norm of dE_H/dz, so it is zero
        exactly at a fixed point of retrieval.
    """

    retrieved, weights = hopfield_retrieve(z, memory, beta=beta, normalize=False)
    return {
        "entropy": attention_entropy(weights),
        "topk_mass": topk_mass(weights, k=top_k),
        "retrieval_distance": retrieval_distance(retrieved, z),
    }


def joint_energy(
    pc_energy: torch.Tensor,
    z: torch.Tensor,
    memory: torch.Tensor,
    beta: float,
    lam: float,
) -> torch.Tensor:
    """E_total = E_PC + lam * E_H, per sample."""

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

    return z - eta * (pc_grad + lam * hopfield_energy_grad(z, memory, beta))


def _descend(
    z0: torch.Tensor,
    pc_energy_fn: Callable[[torch.Tensor], torch.Tensor],
    hopfield_part: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    hopfield_grad: Callable[[torch.Tensor], torch.Tensor],
    trace_fn: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    *,
    eta: float,
    steps: int,
    log_diagnostics: bool,
) -> dict[str, object]:
    """Shared gradient-descent loop behind settle and settle_banks.

    hopfield_part(z) returns (value to log as "hopfield", value added into "total").
    hopfield_grad(z) is the full Hopfield gradient, already weighted by lam.
    """

    z = z0.detach().clone()
    traces: dict[str, list[torch.Tensor]] = {"pc": [], "hopfield": [], "total": []}
    if log_diagnostics:
        traces.update({"entropy": [], "topk_mass": [], "retrieval_distance": []})

    def record(state: torch.Tensor) -> None:
        with torch.no_grad():
            e_pc = pc_energy_fn(state)
            logged, contribution = hopfield_part(state)
            diag = trace_fn(state) if log_diagnostics else {}
        traces["pc"].append(e_pc)
        traces["hopfield"].append(logged)
        traces["total"].append(e_pc + contribution)
        for key, value in diag.items():
            traces[key].append(value)

    record(z)
    for _ in range(steps):
        x = z.detach().requires_grad_(True)
        (grad,) = torch.autograd.grad(pc_energy_fn(x).sum(), x)
        z = (z - eta * (grad + hopfield_grad(z))).detach()
        record(z)

    return {"z": z, **traces}


def settle(
    z0: torch.Tensor,
    pc_energy_fn: Callable[[torch.Tensor], torch.Tensor],
    memory: torch.Tensor,
    *,
    beta: float,
    lam: float,
    eta: float,
    steps: int,
    log_diagnostics: bool = False,
    top_k: int = 3,
) -> dict[str, object]:
    """Run joint PC + Hopfield inference on the hidden state z.

    Args:
        z0: initial hidden state [batch, dim], for example the forward-pass value.
        pc_energy_fn: maps z [batch, dim] to the PC energy per sample [batch]. The PC
            gradient comes from autograd, so any differentiable energy works.
        memory: Hopfield memory [N, dim].
        beta, lam, eta, steps: coupling and step settings.
        log_diagnostics: also record "entropy", "topk_mass" and "retrieval_distance"
            (see hopfield_trace) at every step.
        top_k: k for the top-k attention mass.

    Returns:
        dict with the final state "z" and per-step traces "pc", "hopfield" and
        "total" (lists of [batch] tensors with steps + 1 entries), plus the
        diagnostic traces when log_diagnostics is True.
    """

    def hopfield_part(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e_h = hopfield_energy(state, memory, beta, include_constants=True)
        return e_h, lam * e_h

    return _descend(
        z0,
        pc_energy_fn,
        hopfield_part,
        lambda s: lam * hopfield_energy_grad(s, memory, beta),
        lambda s: hopfield_trace(s, memory, beta, top_k=top_k),
        eta=eta,
        steps=steps,
        log_diagnostics=log_diagnostics,
    )


@dataclass(frozen=True)
class MemoryBank:
    """A Hopfield memory acting on state[..., start:stop] with weight lam."""

    memory: torch.Tensor
    beta: float
    lam: float
    start: int = 0
    stop: int | None = None

    def __post_init__(self) -> None:
        if self.lam < 0:
            raise ValueError(f"lam must be >= 0, got {self.lam!r}.")
        validate_beta(self.beta)
        if self.memory.ndim != 2:
            raise ValueError(
                f"bank memory must be [num_memories, dim], got {tuple(self.memory.shape)}."
            )
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start!r}.")
        if self.stop is not None:
            if self.stop <= self.start:
                raise ValueError(
                    f"stop must be greater than start, got start={self.start}, stop={self.stop}."
                )
            if self.stop - self.start != self.memory.shape[1]:
                raise ValueError(
                    f"bank slice [{self.start}:{self.stop}] has width {self.stop - self.start}, "
                    f"but the memory dim is {self.memory.shape[1]}."
                )


def effective_lam(banks: Sequence[MemoryBank], dim: int) -> float:
    """Upper curvature the banks add to the joint energy.

    Each bank adds lam_b * H_b with H_b <= I on its slice, so the Hopfield part is bounded by
    the diagonal matrix whose entry i is the sum of lam_b over banks covering coordinate i.
    Its largest eigenvalue is returned: max over coordinates of that sum. Banks on disjoint
    slices (different layers) give max lam_b, banks on the same slice give the sum.
    """

    cover = torch.zeros(dim, dtype=torch.float64)
    for bank in banks:
        cover[bank.start : bank.stop] += bank.lam
    return float(cover.max())


def banks_energy(z: torch.Tensor, banks: Sequence[MemoryBank]) -> torch.Tensor:
    """Sum of lam_b * E_H(z[..., slice_b]; M_b, beta_b), per sample [batch]."""

    total = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
    for bank in banks:
        part = z[..., bank.start : bank.stop]
        total = total + bank.lam * hopfield_energy(part, bank.memory, bank.beta, include_constants=True)
    return total


def banks_grad(z: torch.Tensor, banks: Sequence[MemoryBank]) -> torch.Tensor:
    """Gradient of banks_energy: lam_b * (z_slice - retrieved(z_slice)) on each slice."""

    grad = torch.zeros_like(z)
    for bank in banks:
        part = z[..., bank.start : bank.stop]
        grad[..., bank.start : bank.stop] += bank.lam * hopfield_energy_grad(part, bank.memory, bank.beta)
    return grad


def settle_banks(
    z0: torch.Tensor,
    pc_energy_fn: Callable[[torch.Tensor], torch.Tensor],
    banks: Sequence[MemoryBank],
    *,
    eta: float,
    steps: int,
    log_diagnostics: bool = False,
    top_k: int = 3,
) -> dict[str, object]:
    """Joint PC + several-bank Hopfield inference. Same outputs as settle.

    With log_diagnostics the diagnostic traces have shape [batch, num_banks] per step,
    one column per bank, computed on that bank's slice of the state.
    """

    def hopfield_part(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e_h = banks_energy(state, banks)
        return e_h, e_h

    def trace(state: torch.Tensor) -> dict[str, torch.Tensor]:
        per_bank = [
            hopfield_trace(state[..., b.start : b.stop], b.memory, b.beta, top_k=top_k)
            for b in banks
        ]
        return {
            key: torch.stack([d[key] for d in per_bank], dim=-1)
            for key in ("entropy", "topk_mass", "retrieval_distance")
        }

    return _descend(
        z0,
        pc_energy_fn,
        hopfield_part,
        lambda s: banks_grad(s, banks),
        trace,
        eta=eta,
        steps=steps,
        log_diagnostics=log_diagnostics,
    )
