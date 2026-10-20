"""Toy study of Hopfield + predictive-coding coupling.

Training-time inference with input x and output y clamped and the hidden state z free:

    E_PC(z) = 1/2 |z - W1 x|^2 + 1/2 |y - W2 z|^2

The energy is linear-quadratic, so L_PC = 1 + sigma_max(W2)^2 is exact. The output is
smaller than the hidden layer, so the clamped output cannot pin down every hidden
coordinate. The Hopfield memory holds the clean hidden prototypes.

Two questions: how does the pull change the settled hidden state, and does the step-size
rule eta < 2 / (L_PC + lam) hold. This is a mechanism check, not evidence about MNIST
or FabricPC.

Run:
    python -m benchmarks.pc_hopfield_toy
    python -m benchmarks.pc_hopfield_toy --config configs/pc_hopfield_toy.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.reporting import load_config, write_csv, write_json
from hopfieldpc.coupling import hopfield_curvature_bounds, safe_step_size, settle

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 0,
    "d_in": 32,
    "d_h": 16,
    "d_o": 4,
    "num_prototypes": 6,
    "num_samples": 200,
    "noise": 0.5,
    "hidden_scale": 1.5,
    "beta": 4.0,
    "inference_steps": 300,
    "lambda_values": [0.0, 0.05, 0.1, 0.3, 1.0, 3.0],
    "step_rule_lambdas": [0.3, 1.0, 3.0],
    "step_fractions": [0.5, 0.95, 1.05, 1.5],
    "reports_dir": "reports/pc_hopfield_toy",
}


def build(config: dict[str, Any] | None = None):
    """Return (energy_fn, mu1, memory, labels, l_pc). Data is drawn with numpy for reproducibility."""

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    d_in, d_h, d_o = cfg["d_in"], cfg["d_h"], cfg["d_o"]
    k, n = cfg["num_prototypes"], cfg["num_samples"]
    rng = np.random.default_rng(cfg["seed"])

    w1 = torch.from_numpy(rng.standard_normal((d_h, d_in)) / d_in**0.5)
    w2 = torch.from_numpy(rng.standard_normal((d_o, d_h)) / d_h**0.5)
    protos_in = torch.from_numpy(rng.standard_normal((k, d_in)))
    h = protos_in @ w1.T
    h = cfg["hidden_scale"] * h / h.norm(dim=-1, keepdim=True)
    y_k = h @ w2.T
    labels = torch.from_numpy(rng.integers(0, k, n))
    x = protos_in[labels] + cfg["noise"] * torch.from_numpy(rng.standard_normal((n, d_in)))
    mu1 = x @ w1.T
    y = y_k[labels]

    def energy(z: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((z - mu1) ** 2).sum(-1) + 0.5 * ((y - z @ w2.T) ** 2).sum(-1)

    l_pc = 1.0 + float(torch.linalg.matrix_norm(w2, ord=2) ** 2)
    return energy, mu1, h, labels, l_pc


def run_toy(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    energy, mu1, memory, labels, l_pc = build(cfg)
    target = memory[labels]
    beta, steps = cfg["beta"], cfg["inference_steps"]
    lower, upper = hopfield_curvature_bounds(memory, beta)

    start = {
        "e_pc": float(energy(mu1).mean()),
        "dist_to_prototype": float((mu1 - target).norm(dim=-1).mean()),
    }

    sweep = []
    for lam in cfg["lambda_values"]:
        eta = safe_step_size(l_pc, lam)
        out = settle(mu1, energy, memory, beta=beta, lam=lam, eta=eta, steps=steps)
        z = out["z"]
        sweep.append(
            {
                "lam": lam,
                "eta": eta,
                "final_e_pc": float(out["pc"][-1].mean()),
                "final_e_total": float(out["total"][-1].mean()),
                "dist_to_prototype": float((z - target).norm(dim=-1).mean()),
                "nearest_memory_acc": float(((z @ memory.T).argmax(-1) == labels).double().mean()),
            }
        )

    rule = []
    for lam in cfg["step_rule_lambdas"]:
        limit = safe_step_size(l_pc, lam, conservative=False)
        for fraction in cfg["step_fractions"]:
            out = settle(mu1, energy, memory, beta=beta, lam=lam, eta=fraction * limit, steps=steps)
            total = torch.stack(out["total"]).mean(-1)
            rule.append(
                {
                    "lam": lam,
                    "eta_fraction": fraction,
                    "energy_start": float(total[0]),
                    "energy_end": float(total[-1]),
                    "monotone": bool(torch.all(total[1:] <= total[:-1] + 1e-9)),
                }
            )

    return {
        "config": cfg,
        "l_pc": l_pc,
        "curvature_range": [lower, upper],
        "start": start,
        "lambda_sweep": sweep,
        "step_rule": rule,
    }


def save_reports(results: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    write_json(output_dir / "results.json", results)
    write_csv(output_dir / "lambda_sweep.csv", results["lambda_sweep"])
    write_csv(output_dir / "step_rule.csv", results["step_rule"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy study of Hopfield + PC coupling.")
    parser.add_argument("--config", type=Path, default=Path("configs/pc_hopfield_toy.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG, **load_config(args.config)}
    results = run_toy(cfg)
    lo, up = results["curvature_range"]
    print(f"L_PC={results['l_pc']:.3f}  Hopfield curvature range at beta={cfg['beta']:g}: [{lo:.2f}, {up:.0f}]")
    print(f"start (forward pass): E_PC={results['start']['e_pc']:.4f}  dist={results['start']['dist_to_prototype']:.4f}\n")

    print("| lam | safe eta | final E_PC | final E_total | mean dist to clean prototype | nearest-memory acc |")
    print("|---|---|---|---|---|---|")
    for r in results["lambda_sweep"]:
        print(
            f"| {r['lam']:g} | {r['eta']:.3f} | {r['final_e_pc']:.4f} | {r['final_e_total']:.4f} "
            f"| {r['dist_to_prototype']:.4f} | {r['nearest_memory_acc']:.3f} |"
        )
    print("\n| lam | eta / (2/(L_PC+lam)) | total energy start | total energy end | monotone |")
    print("|---|---|---|---|---|")
    for r in results["step_rule"]:
        print(
            f"| {r['lam']:g} | {r['eta_fraction']:g} | {r['energy_start']:.4f} "
            f"| {r['energy_end']:.4g} | {r['monotone']} |"
        )

    out = args.output_dir or Path(cfg["reports_dir"])
    save_reports(results, out)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
