import pytest
import torch

from benchmarks.mini_pc_mnist import (
    MiniPC,
    class_hidden_prototypes,
    random_memory_like,
    run_mini_pc,
    save_reports,
    validate_config,
    DEFAULT_CONFIG,
)
from hopfieldpc import hopfield_retrieve, safe_step_size, settle

CENTERS = torch.randn(4, 20, generator=torch.Generator().manual_seed(123)) * 2


def _blobs(n, seed, spread=1.0):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 4, (n,), generator=g)
    return CENTERS[y] + spread * torch.randn(n, 20, generator=g), y


def test_curvature_matches_exact_hessian_top_eigenvalue():
    torch.manual_seed(0)
    model = MiniPC(20, 16, 4).double()
    w2 = model.l2.weight.detach()
    exact = float(torch.linalg.eigvalsh(torch.eye(16, dtype=torch.float64) + w2.T @ w2).max())
    assert abs(model.curvature() - exact) < 1e-9


@pytest.mark.parametrize("score", ["dot", "l2"])
def test_clamped_inference_energy_is_monotone_at_safe_step(score):
    torch.manual_seed(0)
    x, y = _blobs(64, 2)
    onehot = torch.nn.functional.one_hot(y, 4).double()
    model = MiniPC(20, 32, 4).double()
    memory = torch.randn(4, 32, dtype=torch.float64)
    for lam in (0.0, 0.5):
        out = model.infer_clamped(x.double(), onehot, steps=40, lam=lam, memory=memory, beta=1.0, score=score)
        total = torch.stack(out["total"])
        assert torch.all(total[1:] <= total[:-1] + 1e-9)
        assert torch.all(total[-1] < total[0])


def test_pull_fixed_point_is_blend_of_forward_state_and_retrieved_memory():
    torch.manual_seed(0)
    model = MiniPC(20, 16, 4).double()
    x = torch.randn(8, 20, dtype=torch.float64)
    memory = torch.nn.functional.normalize(torch.randn(4, 16, dtype=torch.float64), dim=-1)
    lam, beta = 1.0, 0.2
    m1 = model.mu1(x).detach()
    z = settle(
        m1, lambda s: 0.5 * ((s - m1) ** 2).sum(-1), memory,
        beta=beta, lam=lam, eta=safe_step_size(1.0, lam), steps=300,
    )["z"]
    retrieved, _ = hopfield_retrieve(z, memory, beta=beta)
    torch.testing.assert_close(z, (m1 + lam * retrieved) / (1 + lam), atol=1e-6, rtol=0)


def test_predict_pulled_with_zero_lambda_equals_feedforward():
    torch.manual_seed(0)
    model = MiniPC(20, 16, 4)
    x = torch.randn(32, 20)
    assert torch.equal(model.predict_pulled(x, torch.randn(4, 16), beta=1.0, lam=0.0), model.predict(x))


def test_memory_helpers():
    torch.manual_seed(0)
    x, y = _blobs(200, 3)
    protos = class_hidden_prototypes(MiniPC(20, 16, 4), x, y, 4)
    assert protos.shape == (4, 16)
    rand = random_memory_like(protos, seed=0)
    torch.testing.assert_close(rand.norm(dim=-1), protos.norm(dim=-1))


def test_config_validation():
    with pytest.raises(ValueError):
        validate_config({**DEFAULT_CONFIG, "train_lams": [0.1]})
    with pytest.raises(ValueError):
        validate_config({**DEFAULT_CONFIG, "pull_scores": ["cosine"]})
    with pytest.raises(ValueError):
        validate_config({**DEFAULT_CONFIG, "val_size": 0})


def test_run_on_synthetic_blobs(tmp_path):
    torch.manual_seed(0)
    train_x, train_y = _blobs(1000, 1)
    test_x, test_y = _blobs(300, 2)
    data = {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}
    config = {
        "seeds": [0], "num_classes": 4, "hidden": 16, "batch_size": 100, "lr": 1e-2,
        "max_epochs": 4, "patience": 2, "infer_steps": 10, "val_size": 200,
        "train_lams": [0.0, 0.1], "noise_levels": [0.0, 1.0], "clamp_noise": False,
        "pull_lams": [0.1, 0.5], "pull_betas": [0.5, 4.0], "pull_steps": 10,
        "readout_betas": [10.0], "num_keys": 200,
    }
    res = run_mini_pc(config, data=data)
    methods = {r["method"] for r in res["rows"]}
    expected = {
        "pc_feedforward[train_lam=0]", "pc_feedforward[train_lam=0.1]",
        "pull[prototypes,dot]", "pull[prototypes,l2]", "pull[random,dot]", "pull[random,l2]",
        "hidden_nearest_mean_l2", "hidden_knn5", "hopfield_readout",
    }
    assert expected <= methods
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in res["rows"])
    clean = {r["method"]: r["accuracy"] for r in res["rows"] if r["noise"] == 0.0}
    assert clean["pc_feedforward[train_lam=0]"] > 0.4
    save_reports(res, tmp_path)
    assert (tmp_path / "summary.md").exists()
