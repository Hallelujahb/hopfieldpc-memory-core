import torch

from benchmarks.readout_controls import predict_methods, run_readout_controls, save_reports


def _blobs(n, centers, seed, spread=0.05):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, centers.shape[0], (n,), generator=g)
    x = centers[y] + spread * torch.randn(n, centers.shape[1], generator=g)
    return x.clamp(0.0, 1.0), y


def _data():
    centers = torch.rand(3, 16, generator=torch.Generator().manual_seed(7))
    train_x, train_y = _blobs(300, centers, 1)
    test_x, test_y = _blobs(120, centers, 2)
    return {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}


def test_predict_methods_returns_a_label_per_sample_for_every_method():
    data = _data()
    protos = torch.stack([data["train_x"][data["train_y"] == c].mean(0) for c in range(3)])
    preds = predict_methods(
        data["test_x"], protos, data["train_x"], data["train_y"],
        betas=[10.0], ks=[1, 3], num_classes=3, chunk=50,
    )
    assert set(preds) == {
        "nearest_mean_l2", "dot_argmax", "cosine_argmax", "knn1", "knn3", "hopfield_knn(beta=10)",
    }
    assert all(p.shape == (120,) for p in preds.values())


def test_run_on_separable_blobs(tmp_path):
    config = {
        "seeds": [0], "num_classes": 3, "num_keys": 100, "noise_levels": [0.0, 0.3],
        "beta_values": [10.0], "knn_values": [1, 5], "chunk": 50,
    }
    res = run_readout_controls(config, data=_data())
    clean = {r["method"]: r["accuracy"] for r in res["rows"] if r["noise"] == 0.0}
    assert clean["knn1"] > 0.9
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in res["rows"])
    assert any("delta_mean" in e for e in res["summary"])
    save_reports(res, tmp_path)
    assert (tmp_path / "summary.md").exists()
