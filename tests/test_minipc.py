import torch

from hopfieldpc import hopfield_retrieve, safe_step_size
from hopfieldpc.minipc import MiniPC, class_hidden_prototypes

CENT = torch.randn(4, 20, generator=torch.Generator().manual_seed(123)) * 2


def _blobs(n, seed, spread=1.5):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 4, (n,), generator=g)
    return CENT[y] + spread * torch.randn(n, 20, generator=g), y


def test_pc_training_learns_synthetic_blobs():
    torch.manual_seed(0)
    x, y = _blobs(1200, 1)
    model = MiniPC(20, 32, 4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    onehot = torch.nn.functional.one_hot(y, 4).float()
    for _ in range(12):
        for i in torch.randperm(len(x)).split(100):
            model.train_step(x[i], onehot[i], opt, steps=10)
    assert (model.predict(x) == y).float().mean() > 0.8


def test_clamped_inference_energy_is_monotone_at_safe_step():
    torch.manual_seed(0)
    x, y = _blobs(64, 2)
    onehot = torch.nn.functional.one_hot(y, 4).double()
    model = MiniPC(20, 32, 4).double()
    mem = torch.randn(4, 32, dtype=torch.float64)
    for lam in (0.0, 0.5):
        out = model.infer_clamped(x.double(), onehot, steps=40, lam=lam, memory=mem, beta=1.0)
        tot = torch.stack(out["total"])
        assert torch.all(tot[1:] <= tot[:-1] + 1e-9)
        assert torch.all(tot[-1] < tot[0])


def test_curvature_matches_exact_hessian_top_eigenvalue():
    torch.manual_seed(0)
    model = MiniPC(20, 16, 4).double()
    w2 = model.l2.weight.detach()
    exact = float(torch.linalg.eigvalsh(torch.eye(16, dtype=torch.float64) + w2.T @ w2).max())
    assert abs(model.curvature() - exact) < 1e-9


def test_pull_fixed_point_is_blend_of_forward_state_and_retrieved_memory():
    torch.manual_seed(0)
    model = MiniPC(20, 16, 4).double()
    x = torch.randn(8, 20, dtype=torch.float64)
    mem = torch.nn.functional.normalize(torch.randn(4, 16, dtype=torch.float64), dim=-1)
    lam, beta = 1.0, 0.2
    m1 = model.mu1(x).detach()

    from hopfieldpc import settle
    z = settle(m1, lambda s: 0.5 * ((s - m1) ** 2).sum(-1), mem,
               beta=beta, lam=lam, eta=safe_step_size(1.0, lam), steps=300)["z"]
    retrieved, _ = hopfield_retrieve(z, mem, beta=beta)
    torch.testing.assert_close(z, (m1 + lam * retrieved) / (1 + lam), atol=1e-6, rtol=0)


def test_predict_pulled_with_zero_lambda_equals_feedforward():
    torch.manual_seed(0)
    model = MiniPC(20, 16, 4)
    x = torch.randn(32, 20)
    mem = torch.randn(4, 16)
    assert torch.equal(model.predict_pulled(x, mem, beta=1.0, lam=0.0), model.predict(x))


def test_class_hidden_prototypes_shape():
    torch.manual_seed(0)
    x, y = _blobs(200, 3)
    protos = class_hidden_prototypes(MiniPC(20, 16, 4), x, y, 4)
    assert protos.shape == (4, 16)
