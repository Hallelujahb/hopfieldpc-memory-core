"""The toy study's claims as tests: monotone below the step-size limit, divergent above it."""

import pytest
import torch

from benchmarks.pc_hopfield_toy import build, run_toy, save_reports
from hopfieldpc import safe_step_size, settle


@pytest.mark.parametrize("lam", [0.3, 1.0, 3.0])
def test_step_size_rule_is_sharp_in_the_toy(lam):
    energy, mu1, memory, _labels, l_pc = build()
    limit = safe_step_size(l_pc, lam, conservative=False)

    below = settle(mu1, energy, memory, beta=4.0, lam=lam, eta=0.95 * limit, steps=300)
    total = torch.stack(below["total"]).mean(-1)
    assert torch.all(total[1:] <= total[:-1] + 1e-9)

    above = settle(mu1, energy, memory, beta=4.0, lam=lam, eta=1.05 * limit, steps=300)
    total = torch.stack(above["total"]).mean(-1)
    assert float(total[-1]) > float(total[0])


def test_pull_moves_state_toward_prototype_and_costs_pc_energy():
    energy, mu1, memory, labels, l_pc = build()
    target = memory[labels]
    results = {}
    for lam in (0.0, 1.0):
        out = settle(mu1, energy, memory, beta=4.0, lam=lam, eta=safe_step_size(l_pc, lam), steps=300)
        results[lam] = (float((out["z"] - target).norm(dim=-1).mean()), float(out["pc"][-1].mean()))
    assert results[1.0][0] < results[0.0][0]
    assert results[1.0][1] > results[0.0][1]


def test_run_toy_tables_and_report_files(tmp_path):
    res = run_toy(
        {
            "inference_steps": 100,
            "lambda_values": [0.0, 1.0],
            "step_rule_lambdas": [1.0],
            "step_fractions": [0.5, 1.5],
        }
    )
    assert len(res["lambda_sweep"]) == 2
    assert [r["monotone"] for r in res["step_rule"]] == [True, False]
    save_reports(res, tmp_path)
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "lambda_sweep.csv").exists()
