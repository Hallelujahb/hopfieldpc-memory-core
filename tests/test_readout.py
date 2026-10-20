import pytest
import torch

from hopfieldpc import HopfieldShapeError, hopfield_retrieve
from hopfieldpc.readout import class_prototypes, hopfield_readout, one_hot_values


def _data(n=12, d=8, b=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(n, d, generator=g, dtype=torch.float64)
    query = torch.randn(b, d, generator=g, dtype=torch.float64)
    return query, keys


@pytest.mark.parametrize("normalize", [False, True])
def test_values_equal_keys_reduces_to_hopfield_retrieve(normalize):
    q, k = _data()
    out, w = hopfield_readout(q, k, k, beta=2.0, normalize=normalize)
    ref, w_ref = hopfield_retrieve(q, k, beta=2.0, normalize=normalize)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(w, w_ref)


def test_one_hot_values_give_a_class_distribution():
    q, k = _data()
    labels = torch.arange(12) % 3
    out, _ = hopfield_readout(q, k, one_hot_values(labels, 3, dtype=torch.float64), beta=1.5)
    assert out.shape == (5, 3)
    assert torch.all(out >= 0)
    torch.testing.assert_close(out.sum(-1), torch.ones(5, dtype=torch.float64))


def test_high_beta_returns_label_of_best_matching_key():
    q, k = _data()
    labels = torch.arange(12) % 4
    out, _ = hopfield_readout(q, k, one_hot_values(labels, 4, dtype=torch.float64), beta=1000.0)
    expected = labels[(q @ k.T).argmax(-1)]
    assert torch.equal(out.argmax(-1), expected)


def test_gradients_flow_to_query_keys_and_values():
    q, k = _data()
    v = torch.randn(12, 3, dtype=torch.float64)
    for t in (q, k, v):
        t.requires_grad_(True)
    out, _ = hopfield_readout(q, k, v, beta=1.0)
    out.sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))


def test_shape_errors():
    q, k = _data()
    with pytest.raises(HopfieldShapeError):
        hopfield_readout(q, k, torch.zeros(11, 3, dtype=torch.float64))
    with pytest.raises(HopfieldShapeError):
        hopfield_readout(q, k, torch.zeros(12, dtype=torch.float64))
    with pytest.raises(TypeError):
        hopfield_readout(q, k, [[0.0]])


def test_class_prototypes_are_class_means():
    f = torch.tensor([[0.0, 0.0], [2.0, 2.0], [10.0, 0.0]])
    p = class_prototypes(f, torch.tensor([0, 0, 1]), 2)
    torch.testing.assert_close(p, torch.tensor([[1.0, 1.0], [10.0, 0.0]]))
    with pytest.raises(ValueError):
        class_prototypes(f, torch.tensor([0, 0, 1]), 3)


def test_one_hot_validates_range():
    with pytest.raises(ValueError):
        one_hot_values(torch.tensor([0, 3]), 3)
