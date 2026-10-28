import tomllib
from pathlib import Path

import torch

from benchmarks.synthetic_retrieval import compute_capacity_summary
from hopfieldpc.functional import hopfield_retrieve, multi_step_retrieve


def _row(size, acc):
    return {"beta": 1.0, "noise": 0.1, "num_updates": 1,
            "memory_size": size, "top1_acc": acc}


def test_capacity_stops_at_first_failing_size():
    rows = [_row(10, 1.0), _row(20, 0.5), _row(40, 0.97)]
    out = compute_capacity_summary(rows, threshold=0.95)
    assert out[0]["capacity_at_95"] == 10


def test_capacity_key_has_no_float_error():
    out = compute_capacity_summary([_row(10, 1.0)], threshold=0.57)
    assert "capacity_at_57" in out[0]


def test_multistep_weights_are_those_of_the_last_update():
    # Documented contract: weights come from the last step (q_{T-1} -> q_T).
    # For T=1 they are the weights at the original query.
    g = torch.Generator().manual_seed(0)
    memory = torch.randn(8, 16, generator=g)
    query = torch.randn(4, 16, generator=g)
    _, w1 = multi_step_retrieve(
        query=query, memory=memory, beta=0.5, num_updates=1, normalize=False)
    _, w_q = hopfield_retrieve(
        query=query, memory=memory, beta=0.5, normalize=False)
    assert torch.allclose(w1, w_q, atol=1e-6)
    states = multi_step_retrieve(
        query=query, memory=memory, beta=0.5, num_updates=3,
        normalize=False, return_states=True)[2]
    _, w3 = multi_step_retrieve(
        query=query, memory=memory, beta=0.5, num_updates=3, normalize=False)
    _, w_q2 = hopfield_retrieve(
        query=states[2], memory=memory, beta=0.5, normalize=False)
    assert torch.allclose(w3, w_q2, atol=1e-6)


def test_torchvision_is_declared():
    root = Path(__file__).resolve().parents[1]
    cfg = tomllib.loads((root / "pyproject.toml").read_text())
    declared = str(cfg["project"].get("dependencies", [])) + str(
        cfg["project"].get("optional-dependencies", {}))
    assert "torchvision" in declared
