"""Tests for the PC + Hopfield coupling (src/hopfieldpc/coupling.py)."""

import pytest
import torch

from hopfieldpc import (
    estimate_top_curvature,
    hopfield_curvature_bounds,
    hopfield_energy,
    hopfield_energy_grad,
    hopfield_retrieve,
    hopfield_smoothness,
    pull_step,
    safe_step_size,
    settle,
)

DT = torch.float64
BETAS = [0.5, 2.0, 8.0]


def _memory(n=10, d=16, scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(n, d, generator=g, dtype=DT)
    return scale * m / m.norm(dim=-1, keepdim=True)


def _query(b=4, d=16, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, d, generator=g, dtype=DT)


def _linear_pc(d_h=16, d_o=6, d_in=12, seed=2):
    """E_PC(z) = 1/2 |z - W1 x|^2 + 1/2 |y - W2 z|^2, Hessian I + W2^T W2, exact L_PC."""

    g = torch.Generator().manual_seed(seed)
    w1 = torch.randn(d_h, d_in, generator=g, dtype=DT) * 0.5
    w2 = torch.randn(d_o, d_h, generator=g, dtype=DT) * 0.5
    x = torch.randn(5, d_in, generator=g, dtype=DT)
    y = torch.randn(5, d_o, generator=g, dtype=DT)
    mu1 = x @ w1.T

    def energy(z):
        return 0.5 * ((z - mu1) ** 2).sum(-1) + 0.5 * ((y - z @ w2.T) ** 2).sum(-1)

    l_pc = 1.0 + float(torch.linalg.matrix_norm(w2, ord=2) ** 2)
    return energy, mu1, l_pc


def test_smoothness_is_largest_curvature_magnitude():
    m = _memory(scale=2.0)
    assert hopfield_smoothness(m, 2.0) == pytest.approx(7.0)


@pytest.mark.parametrize("lam", [0.0, 0.1, 1.0, 5.0])
@pytest.mark.parametrize("beta", BETAS)
def test_joint_energy_monotone_at_safe_step(lam, beta):
    energy, mu1, l_pc = _linear_pc()
    out = settle(mu1, energy, _memory(), beta=beta, lam=lam, eta=safe_step_size(l_pc, lam), steps=60)
    total = torch.stack(out["total"])
    assert torch.all(total[1:] <= total[:-1] + 1e-9)
    assert torch.all(total[-1] < total[0])


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_monotone_just_below_sharp_limit(lam):
    energy, mu1, l_pc = _linear_pc()
    eta = 0.95 * safe_step_size(l_pc, lam, conservative=False)
    out = settle(mu1, energy, _memory(), beta=2.0, lam=lam, eta=eta, steps=80)
    total = torch.stack(out["total"])
    assert torch.all(total[1:] <= total[:-1] + 1e-9)


def test_step_far_above_limit_makes_energy_blow_up():
    energy, mu1, l_pc = _linear_pc()
    out = settle(mu1 + 0.1, energy, _memory(), beta=2.0, lam=0.0, eta=2.5 / l_pc, steps=40)
    assert float(out["total"][-1].max()) > float(out["total"][0].max())


def test_estimate_top_curvature_matches_exact_linear_case():
    energy, mu1, l_pc = _linear_pc()
    assert abs(estimate_top_curvature(energy, mu1, iters=200) - l_pc) < 1e-3


def test_lam_zero_is_plain_pc():
    energy, mu1, l_pc = _linear_pc()
    eta = 1.0 / l_pc
    out = settle(mu1, energy, _memory(), beta=2.0, lam=0.0, eta=eta, steps=30)
    z = mu1.clone()
    for _ in range(30):
        x = z.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(energy(x).sum(), x)
        z = (z - eta * g).detach()
    torch.testing.assert_close(out["z"], z)


def test_settle_trace_lengths():
    energy, mu1, _ = _linear_pc()
    out = settle(mu1, energy, _memory(), beta=2.0, lam=0.5, eta=0.05, steps=7)
    assert len(out["total"]) == len(out["pc"]) == len(out["hopfield"]) == 8


@pytest.mark.parametrize("beta", BETAS)
def test_grad_matches_autograd(beta):
    q, m = _query(), _memory()
    x = q.clone().requires_grad_(True)
    hopfield_energy(x, m, beta, include_constants=True).sum().backward()
    torch.testing.assert_close(x.grad, hopfield_energy_grad(q, m, beta))


@pytest.mark.parametrize("beta", BETAS)
def test_unit_gradient_step_equals_one_step_retrieval(beta):
    q, m = _query(), _memory()
    retrieved, _ = hopfield_retrieve(q, m, beta=beta, normalize=False)
    torch.testing.assert_close(q - hopfield_energy_grad(q, m, beta), retrieved)


@pytest.mark.parametrize("beta", BETAS)
def test_repeated_retrieval_never_increases_energy(beta):
    q, m = _query(), _memory()
    previous = hopfield_energy(q, m, beta, include_constants=True)
    for _ in range(6):
        q, _ = hopfield_retrieve(q, m, beta=beta, normalize=False)
        current = hopfield_energy(q, m, beta, include_constants=True)
        assert torch.all(current <= previous + 1e-9)
        previous = current


@pytest.mark.parametrize("beta", BETAS)
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_hessian_eigenvalues_lie_in_predicted_range(beta, scale):
    m = _memory(scale=scale)
    lower, upper = hopfield_curvature_bounds(m, beta)
    assert upper == 1.0
    for seed in range(5):
        q = _query(b=1, seed=seed)

        def f(z):
            return hopfield_energy(z, m, beta, include_constants=True).sum()

        h = torch.autograd.functional.hessian(f, q).reshape(q.shape[1], q.shape[1])
        eig = torch.linalg.eigvalsh(h)
        assert float(eig.max()) <= upper + 1e-9
        assert float(eig.min()) >= lower - 1e-9


def test_no_pc_term_and_unit_eta_lam_is_one_step_retrieval():
    m, q = _memory(), _query()
    out = settle(q, lambda z: (0.0 * z).sum(-1), m, beta=3.0, lam=1.0, eta=1.0, steps=1)
    retrieved, _ = hopfield_retrieve(q, m, beta=3.0, normalize=False)
    torch.testing.assert_close(out["z"], retrieved)


def test_pull_step_shape_and_argument_validation():
    m, q = _memory(), _query()
    assert pull_step(q, torch.zeros_like(q), m, 2.0, 0.5, 0.1).shape == q.shape
    with pytest.raises(ValueError):
        safe_step_size(1.0, -0.1)
    with pytest.raises(ValueError):
        safe_step_size(0.0, 0.1)
