"""Baselines that any Hopfield readout result must beat (MNIST pixels, no PC network yet).

The existing mnist_prototype_retrieval.py classifies by argmax of the *dot product*
with class-mean prototypes. This script shows how that rule compares with the
standard alternatives, and how a heteroassociative Hopfield readout over stored
training samples (a soft kNN) compares with kNN:

  nearest_mean_l2   argmin ||x - mu_c||            (the usual nearest-class-mean)
  dot_argmax        argmax x . mu_c                (rule used by the existing benchmark)
  cosine_argmax     argmax cos(x, mu_c)
  knn1 / knn5       k nearest training samples (L2)
  hopfield_knn      Hopfield readout over stored samples, cosine scoring, one-hot values

Run:
    python -m benchmarks.mnist_readout_controls
    python -m benchmarks.mnist_readout_controls --num-keys 10000 --seeds 5 --betas 10 30 100

Report mean +- std over seeds. Any claim about a Hopfield readout on PC hidden
features should be made against these rows computed on the SAME features.
"""

from __future__ import annotations

import argparse

import torch

from benchmarks.mnist_prototype_retrieval import add_gaussian_noise, load_mnist
from hopfieldpc.readout import class_prototypes, hopfield_readout, one_hot_values

NUM_CLASSES = 10


def predict_all(x, protos, key_x, key_y, betas, k_values=(1, 5)):
    """Return {method: predicted labels [N]}."""
    preds = {}
    preds["nearest_mean_l2"] = torch.cdist(x, protos).argmin(-1)
    preds["dot_argmax"] = (x @ protos.T).argmax(-1)
    xn = torch.nn.functional.normalize(x, dim=-1)
    pn = torch.nn.functional.normalize(protos, dim=-1)
    preds["cosine_argmax"] = (xn @ pn.T).argmax(-1)
    dist = torch.cdist(x, key_x)
    for k in k_values:
        idx = dist.topk(k, largest=False).indices              # [N, k]
        votes = one_hot_values(key_y[idx].reshape(-1), NUM_CLASSES).reshape(x.shape[0], k, -1).sum(1)
        preds[f"knn{k}"] = votes.argmax(-1)
    values = one_hot_values(key_y, NUM_CLASSES)
    for beta in betas:
        out, _ = hopfield_readout(x, key_x, values, beta=beta, normalize=True)
        preds[f"hopfield_knn(beta={beta:g})"] = out.argmax(-1)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-keys", type=int, default=5000)
    ap.add_argument("--num-test", type=int, default=1000)
    ap.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.3, 0.5, 1.0])
    ap.add_argument("--betas", type=float, nargs="+", default=[10.0, 30.0, 100.0])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--data-dir", default="./data")
    args = ap.parse_args()

    train_x, train_y = load_mnist(args.data_dir, train=True)
    test_x, test_y = load_mnist(args.data_dir, train=False)
    protos = class_prototypes(train_x, train_y, NUM_CLASSES)   # from ALL train samples

    results: dict[tuple[float, str], list[float]] = {}
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        key_idx = torch.randperm(len(train_x))[: args.num_keys]
        test_idx = torch.randperm(len(test_x))[: args.num_test]
        key_x, key_y = train_x[key_idx], train_y[key_idx]
        x_clean, y = test_x[test_idx], test_y[test_idx]
        for noise in args.noise_levels:
            x = x_clean if noise == 0 else add_gaussian_noise(x_clean, noise)
            for name, pred in predict_all(x, protos, key_x, key_y, args.betas).items():
                results.setdefault((noise, name), []).append((pred == y).float().mean().item())

    names = []
    for _, n in results:
        if n not in names:
            names.append(n)
    print(f"keys={args.num_keys} test={args.num_test} seeds={args.seeds}\n")
    print("| method | " + " | ".join(f"noise {n:g}" for n in args.noise_levels) + " |")
    print("|---|" + "---|" * len(args.noise_levels))
    for name in names:
        cells = []
        for n in args.noise_levels:
            t = torch.tensor(results[(n, name)])
            cells.append(f"{t.mean():.3f} +- {t.std(unbiased=False):.3f}")
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
