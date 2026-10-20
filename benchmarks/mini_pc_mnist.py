"""Does a Hopfield term help a predictive-coding classifier? Controlled comparison on MNIST.

A one-hidden-layer PC network is trained per seed with early stopping on a held-out validation
split. Test-time options are compared on clean and noisy test inputs, with no retraining and
no added parameters:

  pc_feedforward[train_lam=L]  the network's own prediction. L > 0 means the Hopfield pull was
                               also used during training inference (the baseline is L = 0)
  pull[memory,score]           hidden state pulled toward a frozen memory at test time.
                               memory is the class-mean hidden prototypes or a random control
                               with the same norms; score is dot or l2
  hidden_nearest_mean_l2       control: nearest class mean on the same hidden features
  hidden_knn5                  control: 5-NN on the same hidden features
  hopfield_readout             soft-kNN Hopfield readout over stored hidden features

Pull strength, beta and readout beta are chosen on a validation copy of the data corrupted with
the same noise level, never on the test set. The table reports test accuracy as mean +/- sample
std over seeds, and the paired per-seed difference from pc_feedforward[train_lam=0].

The success criteria are fixed before running in docs/mini_pc_protocol.md.

Run:
    python -m benchmarks.mini_pc_mnist
    python -m benchmarks.mini_pc_mnist --quick
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from benchmarks.mnist_prototype_retrieval import add_gaussian_noise, load_mnist
from benchmarks.reporting import (
    format_markdown,
    load_config,
    summarize_rows,
    write_csv,
    write_json,
)
from hopfieldpc.coupling import SCORES, safe_step_size, settle
from hopfieldpc.readout import class_prototypes, hopfield_readout, one_hot_values

DEFAULT_CONFIG: dict[str, Any] = {
    "seeds": [0, 1, 2, 3, 4],
    "num_classes": 10,
    "hidden": 256,
    "batch_size": 256,
    "lr": 1e-3,
    "max_epochs": 30,
    "patience": 3,
    "min_delta": 5e-4,
    "infer_steps": 15,
    "val_size": 5000,
    "split_seed": 0,
    "test_size": None,
    "train_lams": [0.0, 0.1],
    "train_beta": 1.0,
    "train_score": "l2",
    "noise_levels": [0.0, 0.3, 0.5, 1.0],
    "clamp_noise": True,
    "pull_lams": [0.05, 0.1, 0.3, 1.0],
    "pull_betas": [0.05, 0.5, 2.0, 8.0],
    "pull_scores": ["dot", "l2"],
    "pull_steps": 30,
    "readout_betas": [10.0, 30.0, 100.0],
    "num_keys": 10000,
    "data_dir": "./data",
    "reports_dir": "reports/mini_pc_mnist",
}
QUICK_OVERRIDES: dict[str, Any] = {
    "seeds": [0], "max_epochs": 2, "patience": 1, "num_keys": 2000, "test_size": 2000,
}
BASELINE = "pc_feedforward[train_lam=0]"


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class MiniPC(nn.Module):
    """x -> z1 = hidden (free) -> z2 = output, with mu1 = tanh(W1 x + b1) and mu2 = W2 z1 + b2.

    With x and the one-hot label y clamped the energy of the hidden state is

        E(z1) = 1/2 |z1 - mu1(x)|^2 + 1/2 |y - mu2(z1)|^2

    which is quadratic in z1 with Hessian I + W2^T W2, so the largest curvature is exact.
    At test time the output is free, the energy is 1/2 |z1 - mu1(x)|^2 and prediction is a
    feed-forward pass. Adding the Hopfield pull then gives the fixed point
    z1* = (mu1(x) + lam * retrieved(z1*)) / (1 + lam).
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, out_dim)
        self.hidden_dim = hidden

    def mu1(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.l1(x))

    def mu2(self, z1: torch.Tensor) -> torch.Tensor:
        return self.l2(z1)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu1(x)

    def energy(self, x: torch.Tensor, y: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((z1 - self.mu1(x)) ** 2).sum(-1) + 0.5 * ((y - self.mu2(z1)) ** 2).sum(-1)

    def curvature(self) -> float:
        w2 = self.l2.weight.detach()
        return 1.0 + float(torch.linalg.matrix_norm(w2, ord=2) ** 2)

    def infer_clamped(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        steps: int,
        lam: float = 0.0,
        memory: torch.Tensor | None = None,
        beta: float = 1.0,
        score: str = "dot",
    ) -> dict[str, object]:
        z0 = self.mu1(x).detach()
        if memory is None:
            memory = torch.zeros(1, self.hidden_dim, dtype=z0.dtype)
        return settle(
            z0,
            lambda z: self.energy(x, y, z),
            memory,
            beta=beta,
            lam=lam,
            eta=safe_step_size(self.curvature(), lam),
            steps=steps,
            score=score,
        )

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        opt: torch.optim.Optimizer,
        *,
        steps: int,
        lam: float = 0.0,
        memory: torch.Tensor | None = None,
        beta: float = 1.0,
        score: str = "dot",
    ) -> float:
        z = self.infer_clamped(x, y, steps=steps, lam=lam, memory=memory, beta=beta, score=score)["z"]
        loss = self.energy(x, y, z.detach()).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.detach())

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu2(self.mu1(x)).argmax(-1)

    def predict_pulled(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        beta: float,
        lam: float,
        steps: int = 30,
        score: str = "dot",
    ) -> torch.Tensor:
        """Test-time prediction with the Hopfield pull on z1 and the output free."""
        m1 = self.mu1(x).detach()
        z = settle(
            m1,
            lambda s: 0.5 * ((s - m1) ** 2).sum(-1),
            memory,
            beta=beta,
            lam=lam,
            eta=safe_step_size(1.0, lam),
            steps=steps,
            score=score,
        )["z"]
        with torch.no_grad():
            return self.mu2(z).argmax(-1)


@torch.no_grad()
def features(model: MiniPC, x: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    return torch.cat([model.hidden(x[i : i + chunk]) for i in range(0, len(x), chunk)])


def class_hidden_prototypes(
    model: MiniPC, x: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    return class_prototypes(features(model, x), labels, num_classes)


def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred == y).float().mean())


def validate_config(cfg: dict[str, Any]) -> None:
    lams = cfg["train_lams"]
    if not lams or float(lams[0]) != 0.0:
        raise ValueError("train_lams must start with 0.0, the baseline.")
    for score in list(cfg["pull_scores"]) + [cfg["train_score"]]:
        if score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}, got {score!r}.")
    if cfg["val_size"] <= 0:
        raise ValueError("val_size must be positive: hyperparameters are chosen on validation data.")


def prepare_splits(data: dict[str, torch.Tensor], cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(cfg["split_seed"])
    perm = torch.randperm(len(data["train_x"]), generator=gen)
    val_idx, train_idx = perm[: cfg["val_size"]], perm[cfg["val_size"] :]
    test_x, test_y = data["test_x"], data["test_y"]
    if cfg["test_size"] is not None:
        pick = torch.randperm(len(test_x), generator=torch.Generator().manual_seed(0))[: cfg["test_size"]]
        test_x, test_y = test_x[pick], test_y[pick]
    return {
        "train_x": data["train_x"][train_idx], "train_y": data["train_y"][train_idx],
        "val_x": data["train_x"][val_idx], "val_y": data["train_y"][val_idx],
        "test_x": test_x, "test_y": test_y,
    }


def corrupt(x: torch.Tensor, noise: float, clamp: bool) -> torch.Tensor:
    if noise == 0:
        return x
    return add_gaussian_noise(x, noise) if clamp else x + noise * torch.randn_like(x)


def fit(
    model: MiniPC,
    split: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    *,
    lam: float,
) -> list[float]:
    """Train with early stopping on validation accuracy. Restores the best weights."""

    num_classes = cfg["num_classes"]
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    onehot = one_hot_values(split["train_y"], num_classes)
    best, best_state, stale = -1.0, None, 0
    history: list[float] = []
    for _ in range(cfg["max_epochs"]):
        memory = None
        if lam > 0:
            memory = class_hidden_prototypes(model, split["train_x"], split["train_y"], num_classes)
        for idx in torch.randperm(len(split["train_x"])).split(cfg["batch_size"]):
            model.train_step(
                split["train_x"][idx], onehot[idx], opt,
                steps=cfg["infer_steps"], lam=lam, memory=memory,
                beta=cfg["train_beta"], score=cfg["train_score"],
            )
        val_acc = accuracy(model.predict(split["val_x"]), split["val_y"])
        history.append(val_acc)
        _log(f"  epoch {len(history)}: val acc {val_acc:.4f}")
        if val_acc > best + cfg["min_delta"]:
            best, stale = val_acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return history


def _select(candidates: list[dict[str, float]], predict_val, y_val: torch.Tensor) -> dict[str, float]:
    best, best_acc = candidates[0], -1.0
    for cand in candidates:
        acc = accuracy(predict_val(cand), y_val)
        if acc > best_acc:
            best, best_acc = cand, acc
    return best


def _knn_predict(h, keys_h, keys_y, k, num_classes, chunk=2000):
    values = one_hot_values(keys_y, num_classes)
    out = []
    for start in range(0, len(h), chunk):
        idx = torch.cdist(h[start : start + chunk], keys_h).topk(k, largest=False).indices
        out.append(values[idx].sum(1).argmax(-1))
    return torch.cat(out)


def _readout_predict(h, keys_h, values, beta, chunk=2000):
    out = []
    for start in range(0, len(h), chunk):
        scores, _ = hopfield_readout(h[start : start + chunk], keys_h, values, beta=beta, normalize=True)
        out.append(scores.argmax(-1))
    return torch.cat(out)


def random_memory_like(protos: torch.Tensor, seed: int) -> torch.Tensor:
    """Random vectors with the same per-row norms as the prototypes (content control)."""
    gen = torch.Generator().manual_seed(seed)
    rand = torch.randn(protos.shape, generator=gen)
    return rand / rand.norm(dim=-1, keepdim=True) * protos.norm(dim=-1, keepdim=True)


def evaluate_noise_level(
    *,
    models: dict[float, MiniPC],
    memories: dict[str, torch.Tensor],
    keys_h: torch.Tensor,
    keys_y: torch.Tensor,
    split: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    seed: int,
    noise: float,
) -> list[dict[str, Any]]:
    base = models[0.0]
    num_classes = cfg["num_classes"]
    x_val = corrupt(split["val_x"], noise, cfg["clamp_noise"])
    x_test = corrupt(split["test_x"], noise, cfg["clamp_noise"])
    y_val, y_test = split["val_y"], split["test_y"]
    rows: list[dict[str, Any]] = []

    def add(method: str, acc: float, chosen: str = "") -> None:
        rows.append({"seed": seed, "noise": noise, "method": method, "accuracy": acc, "chosen": chosen})

    for lam, model in models.items():
        add(f"pc_feedforward[train_lam={lam:g}]", accuracy(model.predict(x_test), y_test))

    grid = [{"lam": a, "beta": b} for a, b in itertools.product(cfg["pull_lams"], cfg["pull_betas"])]
    for mem_name, memory in memories.items():
        for score in cfg["pull_scores"]:

            def pulled(x, c, memory=memory, score=score):
                return base.predict_pulled(x, memory, beta=c["beta"], lam=c["lam"], steps=cfg["pull_steps"], score=score)

            best = _select(grid, lambda c: pulled(x_val, c), y_val)
            add(f"pull[{mem_name},{score}]", accuracy(pulled(x_test, best), y_test),
                f"lam={best['lam']:g},beta={best['beta']:g}")

    h_val, h_test = features(base, x_val), features(base, x_test)
    add("hidden_nearest_mean_l2", accuracy(torch.cdist(h_test, memories["prototypes"]).argmin(-1), y_test))
    add("hidden_knn5", accuracy(_knn_predict(h_test, keys_h, keys_y, 5, num_classes), y_test))

    values = one_hot_values(keys_y, num_classes)
    candidates = [{"beta": b} for b in cfg["readout_betas"]]
    best = _select(candidates, lambda c: _readout_predict(h_val, keys_h, values, c["beta"]), y_val)
    add("hopfield_readout", accuracy(_readout_predict(h_test, keys_h, values, best["beta"]), y_test),
        f"beta={best['beta']:g}")
    return rows


def run_mini_pc(
    config: dict[str, Any] | None = None,
    data: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    validate_config(cfg)
    if data is None:
        _log("loading MNIST (slow: converts 70k images one by one)")
        train_x, train_y = load_mnist(cfg["data_dir"], train=True)
        test_x, test_y = load_mnist(cfg["data_dir"], train=False)
        data = {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}
    split = prepare_splits(data, cfg)
    in_dim, num_classes = split["train_x"].shape[1], cfg["num_classes"]

    rows: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    for seed in cfg["seeds"]:
        models: dict[float, MiniPC] = {}
        for lam in cfg["train_lams"]:
            torch.manual_seed(seed)
            model = MiniPC(in_dim, cfg["hidden"], num_classes)
            history = fit(model, split, cfg, lam=float(lam))
            models[float(lam)] = model
            _log(f"seed {seed} train_lam {lam:g}: trained {len(history)} epochs")
            training.append({"seed": seed, "train_lam": lam, "epochs": len(history), "best_val_acc": max(history)})

        base = models[0.0]
        protos = class_hidden_prototypes(base, split["train_x"], split["train_y"], num_classes)
        memories = {"prototypes": protos, "random": random_memory_like(protos, seed)}
        gen = torch.Generator().manual_seed(seed)
        key_idx = torch.randperm(len(split["train_x"]), generator=gen)[: cfg["num_keys"]]
        keys_h, keys_y = features(base, split["train_x"][key_idx]), split["train_y"][key_idx]

        for noise in cfg["noise_levels"]:
            _log(f"seed {seed} evaluating noise {noise:g}")
            torch.manual_seed(1000 * seed + int(round(100 * noise)))
            rows.extend(
                evaluate_noise_level(
                    models=models, memories=memories, keys_h=keys_h, keys_y=keys_y,
                    split=split, cfg=cfg, seed=seed, noise=noise,
                )
            )

    return {
        "config": cfg,
        "rows": rows,
        "training": training,
        "summary": summarize_rows(rows, baseline=BASELINE),
    }


def save_reports(results: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    write_json(output_dir / "results.json", results)
    write_csv(output_dir / "results.csv", results["rows"])
    write_csv(output_dir / "training.csv", results["training"])
    write_csv(output_dir / "summary.csv", results["summary"])
    table = format_markdown(results["summary"], results["config"]["noise_levels"])
    (output_dir / "summary.md").write_text(table + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini PC + Hopfield comparison on MNIST.")
    parser.add_argument("--config", type=Path, default=Path("configs/mini_pc_mnist.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="Smoke run: 1 seed, 2 epochs, small test set.")
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG, **load_config(args.config)}
    if args.quick:
        cfg.update(QUICK_OVERRIDES)
        print("quick mode: smoke test only, do not report these numbers")
    results = run_mini_pc(cfg)

    print("training (epochs run, best validation accuracy):")
    for t in results["training"]:
        print(f"  seed {t['seed']} train_lam {t['train_lam']:g}: {t['epochs']} epochs, val acc {t['best_val_acc']:.4f}")
    print(f"\ntest accuracy, mean +/- sample std over {len(cfg['seeds'])} seeds (diff is paired vs {BASELINE})\n")
    print(format_markdown(results["summary"], cfg["noise_levels"]))
    out = args.output_dir or Path(cfg["reports_dir"])
    save_reports(results, out)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
