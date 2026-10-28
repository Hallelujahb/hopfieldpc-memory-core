"""Tests for the PC + Hopfield coupling (src/hopfieldpc/coupling.py)."""

import pytest
import torch

from hopfieldpc import (
    MemoryBank,
    banks_energy,
    banks_grad,
    effective_lam,
    estimate_top_curvature,
    hopfield_curvature_bounds,
    hopfield_energy,
    hopfield_energy_grad,
    hopfield_retrieve,
    hopfield_smoothness,
    hopfield_trace,
    pull_step,
    safe_step_size,
    settle,
    settle_banks,
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


SPANS = [(0, 8), (8, 16)]
CONFIGS = [
    ([0.5, 4.0], SPANS),
    ([2.0, 2.0], SPANS),
    ([0.5, 4.0], [(0, 8), (0, 8)]),
    ([1.0, 2.0], [(0, 8), (4, 12)]),
]


def _banks(lams, spans, beta=2.0):
    return [
        MemoryBank(_memory(n=6, d=b - a, seed=10 + i), beta=beta, lam=lam, start=a, stop=b)
        for i, (lam, (a, b)) in enumerate(zip(lams, spans))
    ]


def test_effective_lam_max_for_disjoint_sum_for_same_slice():
    assert effective_lam(_banks([0.5, 4.0], SPANS), 16) == pytest.approx(4.0)
    assert effective_lam(_banks([0.5, 4.0], [(0, 8), (0, 8)]), 16) == pytest.approx(4.5)
    assert effective_lam(_banks([1.0, 2.0], [(0, 8), (4, 12)]), 16) == pytest.approx(3.0)


def test_bank_negative_lam_rejected():
    with pytest.raises(ValueError):
        MemoryBank(_memory(n=6, d=8), beta=2.0, lam=-1.0)


def test_banks_grad_matches_autograd():
    banks = _banks([0.5, 4.0], [(0, 8), (4, 12)])
    z = _query(b=3, d=16)
    x = z.clone().requires_grad_(True)
    banks_energy(x, banks).sum().backward()
    torch.testing.assert_close(x.grad, banks_grad(z, banks))


@pytest.mark.parametrize("lams,spans", CONFIGS)
def test_bank_hessian_upper_bound(lams, spans):
    banks = _banks(lams, spans)
    bound = effective_lam(banks, 16)
    for seed in range(5):
        z = _query(b=1, d=16, seed=seed)
        h = torch.autograd.functional.hessian(lambda x: banks_energy(x, banks).sum(), z).reshape(16, 16)
        assert float(torch.linalg.eigvalsh(h).max()) <= bound + 1e-9


@pytest.mark.parametrize("lams,spans", CONFIGS)
def test_banks_monotone_below_sharp_limit(lams, spans):
    energy, mu1, l_pc = _linear_pc()
    banks = _banks(lams, spans)
    eta = 0.95 * safe_step_size(l_pc, effective_lam(banks, 16), conservative=False)
    out = settle_banks(mu1, energy, banks, eta=eta, steps=80)
    total = torch.stack(out["total"])
    assert torch.all(total[1:] <= total[:-1] + 1e-9)
    assert torch.all(total[-1] < total[0])


def test_single_full_bank_matches_settle():
    energy, mu1, l_pc = _linear_pc()
    m = _memory()
    eta = 1.0 / (l_pc + 0.5)
    a = settle(mu1, energy, m, beta=2.0, lam=0.5, eta=eta, steps=20)
    b = settle_banks(mu1, energy, [MemoryBank(m, beta=2.0, lam=0.5)], eta=eta, steps=20)
    torch.testing.assert_close(a["z"], b["z"])


def test_trace_entropy_and_topk_mass_at_extreme_beta():
    m, q = _memory(n=10), _query()
    flat = hopfield_trace(q, m, 1e-6, top_k=3)
    assert torch.allclose(flat["entropy"], torch.full_like(flat["entropy"], 10.0).log(), atol=1e-3)
    assert torch.allclose(flat["topk_mass"], torch.full_like(flat["topk_mass"], 0.3), atol=1e-3)
    sharp = hopfield_trace(q, m, 1000.0, top_k=1)
    assert torch.all(sharp["topk_mass"] > 0.99)
    assert torch.all(sharp["entropy"] < 0.05)


@pytest.mark.parametrize("beta", BETAS)
def test_trace_retrieval_distance_is_gradient_norm(beta):
    q, m = _query(), _memory()
    torch.testing.assert_close(
        hopfield_trace(q, m, beta)["retrieval_distance"], hopfield_energy_grad(q, m, beta).norm(dim=-1)
    )


def test_settle_diagnostics_are_opt_in_and_do_not_change_the_result():
    energy, mu1, l_pc = _linear_pc()
    kw = {"beta": 2.0, "lam": 0.5, "eta": 1.0 / (l_pc + 0.5), "steps": 12}
    plain = settle(mu1, energy, _memory(), **kw)
    logged = settle(mu1, energy, _memory(), log_diagnostics=True, **kw)
    assert set(plain) == {"z", "pc", "hopfield", "total"}
    for key in ("entropy", "topk_mass", "retrieval_distance"):
        assert len(logged[key]) == 13
        assert logged[key][0].shape == (mu1.shape[0],)
    torch.testing.assert_close(plain["z"], logged["z"])


def test_settle_retrieval_distance_matches_gradient_norm_at_every_step():
    m, q = _memory(), _query()
    out = settle(q, lambda z: (0.0 * z).sum(-1), m, beta=3.0, lam=1.0, eta=0.5, steps=5, log_diagnostics=True)
    assert out["retrieval_distance"][-1].mean() < out["retrieval_distance"][0].mean()


def test_settle_banks_diagnostics_have_one_column_per_bank():
    energy, mu1, _ = _linear_pc()
    banks = _banks([0.5, 4.0], SPANS)
    out = settle_banks(mu1, energy, banks, eta=0.05, steps=5, log_diagnostics=True)
    assert len(out["entropy"]) == 6
    assert out["entropy"][0].shape == (mu1.shape[0], 2)
    z = out["z"]
    for j, b in enumerate(banks):
        expected = hopfield_trace(z[..., b.start : b.stop], b.memory, b.beta)
        torch.testing.assert_close(out["entropy"][-1][:, j], expected["entropy"])
        torch.testing.assert_close(out["retrieval_distance"][-1][:, j], expected["retrieval_distance"])
    assert set(settle_banks(mu1, energy, banks, eta=0.05, steps=2)) == {"z", "pc", "hopfield", "total"}


def test_estimate_top_curvature_returns_signed_top_with_negative_curvature():
    d = torch.tensor([-5.0, 1.0], dtype=DT)

    def energy(z):
        return 0.5 * (d * z**2).sum(-1)

    z = torch.ones(3, 2, dtype=DT)
    assert estimate_top_curvature(energy, z, iters=200) == pytest.approx(1.0, abs=1e-3)


def test_bank_slice_width_must_match_memory_dim():
    with pytest.raises(ValueError, match="width"):
        MemoryBank(_memory(n=6, d=8), beta=2.0, lam=1.0, start=0, stop=12)


def test_bank_rejects_bad_beta_and_empty_slice():
    with pytest.raises(ValueError, match="beta"):
        MemoryBank(_memory(n=6, d=8), beta=0.0, lam=1.0)
    with pytest.raises(ValueError, match="stop"):
        MemoryBank(_memory(n=6, d=8), beta=2.0, lam=1.0, start=8, stop=8)
