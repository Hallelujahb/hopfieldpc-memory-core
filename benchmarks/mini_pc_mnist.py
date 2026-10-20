"""Does a Hopfield term help a predictive-coding classifier on (noisy) MNIST? Honest comparison.

Trains a standalone one-hidden-layer PC network (src/hopfieldpc/minipc.py) per seed, then
evaluates, on clean and noisy test inputs, with NO retraining and NO added parameters:

  pc_feedforward           the PC network's own prediction (the baseline)
  pull(lam, beta)          Method A/B at test time: z1 pulled toward frozen class prototypes
  hidden_nearest_mean_l2   control: nearest class mean on the SAME hidden features
  hidden_knn5              control: 5-NN on the SAME hidden features
  hopfield_readout(beta)   Method C: heteroassociative Hopfield readout over stored hidden
                           features (soft kNN, cosine scoring, one-hot labels)

A Hopfield result only counts if it beats pc_feedforward AND the matching control rows,
averaged over seeds. Noise is Gaussian and NOT clamped (clamping to [0,1] biases the input).

Run:
    python -m benchmarks.mini_pc_mnist
    python -m benchmarks.mini_pc_mnist --epochs 8 --seeds 5 --json-out reports/mini_pc.json
"""

from __future__ import annotations

import argparse
import json

import torch

from benchmarks.mnist_prototype_retrieval import load_mnist
from hopfieldpc.minipc import MiniPC, class_hidden_prototypes
from hopfieldpc.readout import hopfield_readout, one_hot_values

C = 10


def evaluate(model, x, y, protos, keys_h, keys_y, args):
    rows = {"pc_feedforward": (model.predict(x) == y).float().mean().item()}
    for lam in args.lams:
        for beta in args.betas:
            pred = model.predict_pulled(x, protos, beta=beta, lam=lam)
            rows[f"pull(lam={lam:g},beta={beta:g})"] = (pred == y).float().mean().item()
    with torch.no_grad():
        h = model.hidden(x)
    rows["hidden_nearest_mean_l2"] = (torch.cdist(h, protos).argmin(-1) == y).float().mean().item()
    idx = torch.cdist(h, keys_h).topk(5, largest=False).indices
    votes = one_hot_values(keys_y[idx].reshape(-1), C).reshape(len(h), 5, C).sum(1)
    rows["hidden_knn5"] = (votes.argmax(-1) == y).float().mean().item()
    vals = one_hot_values(keys_y, C)
    for beta in args.readout_betas:
        out, _ = hopfield_readout(h, keys_h, vals, beta=beta, normalize=True)
        rows[f"hopfield_readout(beta={beta:g})"] = (out.argmax(-1) == y).float().mean().item()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--infer-steps", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--lams", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    ap.add_argument("--betas", type=float, nargs="+", default=[0.5, 2.0])
    ap.add_argument("--readout-betas", type=float, nargs="+", default=[10.0, 30.0, 100.0])
    ap.add_argument("--num-keys", type=int, default=5000)
    ap.add_argument("--num-test", type=int, default=2000)
    ap.add_argument("--train-lam", type=float, default=0.0, help="Hopfield pull during TRAINING inference (0 = off)")
    ap.add_argument("--train-beta", type=float, default=1.0)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    train_x, train_y = load_mnist(args.data_dir, train=True)
    test_x, test_y = load_mnist(args.data_dir, train=False)
    onehot = one_hot_values(train_y, C)

    results: dict[tuple[float, str], list[float]] = {}
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        model = MiniPC(784, args.hidden, C)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            mem = class_hidden_prototypes(model, train_x, train_y, C) if args.train_lam > 0 else None
            loss = 0.0
            perm = torch.randperm(len(train_x))
            for i in perm.split(args.batch):
                loss = model.train_step(train_x[i], onehot[i], opt, steps=args.infer_steps,
                                        lam=args.train_lam, memory=mem, beta=args.train_beta)
            acc = (model.predict(test_x) == test_y).float().mean().item()
            print(f"seed {seed} epoch {epoch + 1}/{args.epochs}  last-batch energy {loss:.4f}  clean test acc {acc:.4f}", flush=True)

        protos = class_hidden_prototypes(model, train_x, train_y, C)
        kidx = torch.randperm(len(train_x))[: args.num_keys]
        with torch.no_grad():
            keys_h, keys_y = model.hidden(train_x[kidx]), train_y[kidx]
        tidx = torch.randperm(len(test_x))[: args.num_test]
        for noise in args.noise_levels:
            x = test_x[tidx] + noise * torch.randn(len(tidx), 784)
            for name, a in evaluate(model, x, test_y[tidx], protos, keys_h, keys_y, args).items():
                results.setdefault((noise, name), []).append(a)

    names = list(dict.fromkeys(n for _, n in results))
    print(f"\nhidden={args.hidden} epochs={args.epochs} seeds={args.seeds} train_lam={args.train_lam}\n")
    print("| method | " + " | ".join(f"noise {n:g}" for n in args.noise_levels) + " |")
    print("|---|" + "---|" * len(args.noise_levels))
    summary = {}
    for name in names:
        cells = []
        for n in args.noise_levels:
            t = torch.tensor(results[(n, name)])
            cells.append(f"{t.mean():.3f} +- {t.std(unbiased=False):.3f}")
            summary[f"{name}|noise={n:g}"] = [float(v) for v in t]
        print(f"| {name} | " + " | ".join(cells) + " |")
    if args.json_out:
        import pathlib
        pathlib.Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json_out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
