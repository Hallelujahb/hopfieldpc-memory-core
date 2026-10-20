"""Toy study: does adding a Hopfield energy term change PC inference, and when is it stable?

Setting (training-time inference, input x and output y clamped, hidden z free):

    E_PC(z) = 1/2 |z - W1 x|^2 + 1/2 |y - W2 z|^2       (linear, so L_PC is exact)

The output is lower-dimensional than the hidden layer (d_o < d_h), so the clamped
output cannot pin down every hidden coordinate. The Hopfield memory holds the
clean hidden prototypes. This toy is built to ask one clear question
(does the pull recover what PC cannot infer?); it is NOT evidence about MNIST
or FabricPC.

Run:  python -m benchmarks.pc_hopfield_toy
"""

from __future__ import annotations

import numpy as np
import torch

from hopfieldpc import hopfield_curvature_bounds, safe_step_size, settle


def build(seed=0, d_in=32, d_h=16, d_o=4, k=6, n=200, noise=0.5, hscale=1.5):
    # Data is drawn with numpy so results are reproducible across torch versions.
    r = np.random.default_rng(seed)
    w1 = torch.from_numpy(r.standard_normal((d_h, d_in)) / d_in**0.5)
    w2 = torch.from_numpy(r.standard_normal((d_o, d_h)) / d_h**0.5)
    protos_in = torch.from_numpy(r.standard_normal((k, d_in)))
    h = protos_in @ w1.T
    h = hscale * h / h.norm(dim=-1, keepdim=True)          # clean hidden prototypes = memory
    y_k = h @ w2.T
    labels = torch.from_numpy(r.integers(0, k, n))
    x = protos_in[labels] + noise * torch.from_numpy(r.standard_normal((n, d_in)))
    mu1 = x @ w1.T
    y = y_k[labels]

    def energy(z):
        return 0.5 * ((z - mu1) ** 2).sum(-1) + 0.5 * ((y - z @ w2.T) ** 2).sum(-1)

    l_pc = 1.0 + float(torch.linalg.matrix_norm(w2, ord=2) ** 2)
    return energy, mu1, h, labels, l_pc


def main():
    energy, mu1, memory, labels, l_pc = build()
    target = memory[labels]
    beta = 4.0
    lo, up = hopfield_curvature_bounds(memory, beta)
    print(f"L_PC={l_pc:.3f}  Hopfield curvature range at beta={beta}: [{lo:.2f}, {up:.0f}]\n")

    print("Part 1: effect of lambda (eta = safe step, 300 inference steps)")
    print("| lam | safe eta | final E_PC | final E_total | mean dist(z, clean prototype) | nearest-memory acc |")
    print("|---|---|---|---|---|---|")
    for lam in [0.0, 0.05, 0.1, 0.3, 1.0, 3.0]:
        eta = safe_step_size(l_pc, lam)
        out = settle(mu1, energy, memory, beta=beta, lam=lam, eta=eta, steps=300)
        z = out["z"]
        dist = (z - target).norm(dim=-1).mean()
        acc = ((z @ memory.T).argmax(-1) == labels).double().mean()
        print(f"| {lam:g} | {eta:.3f} | {out['pc'][-1].mean():.4f} | {out['total'][-1].mean():.4f} | {dist:.4f} | {acc:.3f} |")

    print("\nPart 2: step-size rule  eta < 2 / (L_PC + lam)")
    print("| lam | eta / (2/(L_PC+lam)) | total energy start | total energy end | monotone? |")
    print("|---|---|---|---|---|")
    for lam in [0.3, 1.0, 3.0]:
        limit = safe_step_size(l_pc, lam, conservative=False)
        for frac in [0.5, 0.95, 1.05, 1.5]:
            out = settle(mu1, energy, memory, beta=beta, lam=lam, eta=frac * limit, steps=300)
            tot = torch.stack(out["total"]).mean(-1)
            mono = bool(torch.all(tot[1:] <= tot[:-1] + 1e-9))
            print(f"| {lam:g} | {frac:g} | {tot[0]:.4f} | {tot[-1]:.4g} | {mono} |")


if __name__ == "__main__":
    main()
