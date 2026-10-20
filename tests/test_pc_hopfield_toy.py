"""The toy study's claims, as tests: monotone below the step-size limit, divergent above it."""

import pytest
import torch

from benchmarks.pc_hopfield_toy import build
from hopfieldpc import safe_step_size, settle


@pytest.mark.parametrize("lam", [0.3, 1.0, 3.0])
def test_step_size_rule_is_sharp_in_the_toy(lam):
    energy, mu1, memory, _labels, l_pc = build()
    limit = safe_step_size(l_pc, lam, conservative=False)

    below = settle(mu1, energy, memory, beta=4.0, lam=lam, eta=0.95 * limit, steps=300)
    tot = torch.stack(below["total"]).mean(-1)
    assert torch.all(tot[1:] <= tot[:-1] + 1e-9)

    above = settle(mu1, energy, memory, beta=4.0, lam=lam, eta=1.05 * limit, steps=300)
    tot = torch.stack(above["total"]).mean(-1)
    assert float(tot[-1]) > float(tot[0])


def test_pull_moves_state_toward_prototype_and_costs_pc_energy():
    energy, mu1, memory, labels, l_pc = build()
    target = memory[labels]
    results = {}
    for lam in (0.0, 1.0):
        eta = safe_step_size(l_pc, lam)
        out = settle(mu1, energy, memory, beta=4.0, lam=lam, eta=eta, steps=300)
        results[lam] = (
            float((out["z"] - target).norm(dim=-1).mean()),
            float(out["pc"][-1].mean()),
        )
    assert results[1.0][0] < results[0.0][0]   # closer to the clean prototype
    assert results[1.0][1] > results[0.0][1]   # but PC explains the input less well
