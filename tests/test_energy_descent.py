"""The default hopfield_energy omits a q-dependent term; the full energy is what retrieval descends."""

import pytest
import torch

from hopfieldpc import (
    hopfield_energy,
    hopfield_full_energy,
    hopfield_retrieve,
    multi_step_retrieve,
)

BETAS = [0.5, 2.0, 8.0]


def _setup(seed=0):
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(10, 16, generator=g, dtype=torch.float64)
    q = torch.randn(4, 16, generator=g, dtype=torch.float64)
    return q, m / m.norm(dim=-1, keepdim=True)


def test_full_energy_matches_include_constants():
    q, m = _setup()
    torch.testing.assert_close(
        hopfield_full_energy(q, m, 2.0),
        hopfield_energy(q, m, 2.0, include_constants=True),
    )


def test_full_energy_reduce_mean():
    q, m = _setup()
    per_query = hopfield_full_energy(q, m, 2.0)
    torch.testing.assert_close(hopfield_full_energy(q, m, 2.0, reduce="mean"), per_query.mean())


@pytest.mark.parametrize("beta", BETAS)
def test_full_energy_never_increases_under_retrieval(beta):
    q, m = _setup()
    previous = hopfield_full_energy(q, m, beta)
    for _ in range(6):
        q, _ = multi_step_retrieve(q, m, beta=beta, num_updates=1)
        current = hopfield_full_energy(q, m, beta)
        assert torch.all(current <= previous + 1e-9)
        previous = current


def test_default_energy_is_not_a_descent_function():
    memory = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    q0 = torch.tensor([[0.8, 0.5]])
    q1, _ = hopfield_retrieve(q0, memory, beta=5.0)

    assert hopfield_energy(q1, memory, beta=5.0).item() > hopfield_energy(q0, memory, beta=5.0).item()
    assert hopfield_full_energy(q1, memory, beta=5.0).item() < hopfield_full_energy(q0, memory, beta=5.0).item()
