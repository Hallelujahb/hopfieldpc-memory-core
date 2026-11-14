"""normalize=True retrieval is not a gradient field, so no energy exists for it.

F(q) = q - retrieved(q) is the gradient of a scalar energy only if its
Jacobian is symmetric. Unnormalized retrieval satisfies this (it is the
gradient of hopfield_energy); cosine-normalized scoring does not.
"""

import pytest
import torch
from torch.autograd.functional import jacobian

from hopfieldpc import hopfield_retrieve


def _jacobian(q, memory, beta, normalize):
    def f(x):
        out, _ = hopfield_retrieve(x.unsqueeze(0), memory, beta=beta, normalize=normalize)
        return out.squeeze(0)

    return jacobian(f, q)


def _asymmetry(J):
    return (torch.linalg.matrix_norm(J - J.T) / torch.linalg.matrix_norm(J)).item()


def _case(beta_seed):
    g = torch.Generator().manual_seed(beta_seed)
    memory = torch.randn(5, 6, generator=g, dtype=torch.float64)
    q = torch.randn(6, generator=g, dtype=torch.float64)
    return q, memory


@pytest.mark.parametrize("beta", [0.5, 2.0, 8.0])
def test_unnormalized_retrieval_is_a_gradient_field(beta):
    q, memory = _case(0)
    assert _asymmetry(_jacobian(q, memory, beta, normalize=False)) < 1e-10


@pytest.mark.parametrize("beta", [0.5, 2.0, 8.0])
def test_normalized_retrieval_is_not_a_gradient_field(beta):
    q, memory = _case(0)
    assert _asymmetry(_jacobian(q, memory, beta, normalize=True)) > 1e-3
