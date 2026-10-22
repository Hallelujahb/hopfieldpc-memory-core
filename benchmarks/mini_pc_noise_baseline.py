"""v2: does the Hopfield pull still help a baseline trained on noisy inputs?

Protocol: docs/mini_pc_protocol_v2_noise_baseline.md. Reuses the v1 model, split, grid and
selection code from benchmarks/mini_pc_mnist.py.

Run:
    python -m benchmarks.mini_pc_noise_baseline
    python -m benchmarks.mini_pc_noise_baseline --quick
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import torch

from benchmarks.mini_pc_mnist import (
    DEFAULT_CONFIG,
    MiniPC,
    _knn_predict,
    _select,
    accuracy,
    class_hidden_prototypes,
    corrupt,
    features,
    fit,
    prepare_splits,
    random_memory_like,
)
from benchmarks.mnist_prototype_retrieval import load_mnist
from benchmarks.reporting import (
    format_markdown,
    load_config,
    summarize_rows,
    write_csv,
    write_json,
)
from hopfieldpc.readout import one_hot_values

TRAIN_NOISES = [0.0, 0.3, 0.5]
BASELINE = "noisetrained_feedforward"
QUICK_OVERRIDES: dict[str, Any] = {
    "seeds": [0], "max_epochs": 2, "patience": 1, "num_keys": 2000, "test_size": 2000,
}


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def fixed_noisy_val(split: dict[str, torch.Tensor], cfg: dict[str, Any], seed: int) -> torch.Tensor:
    """Validation images, one third each at the training noise levels, drawn once."""
    state = torch.get_rng_state()
    torch.manual_seed(10_000 + seed)
    parts = [
        corrupt(chunk, sigma, cfg["clamp_noise"])
        for chunk, sigma in zip(split["val_x"].chunk(len(TRAIN_NOISES)), TRAIN_NOISES)
    ]
    torch.set_rng_state(state)
    return torch.cat(parts)


def fit_noisy(model: MiniPC, split, cfg, val_x_noisy: torch.Tensor) -> list[float]:
    """Same loop and early stopping as v1 fit(), but batches are noise-augmented."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    onehot = one_hot_values(split["train_y"], cfg["num_classes"])
    best, best_state, stale = -1.0, None, 0
    history: list[float] = []
    for _ in range(cfg["max_epochs"]):
        for idx in torch.randperm(len(split["train_x"])).split(cfg["batch_size"]):
            sigma = TRAIN_NOISES[int(torch.randint(len(TRAIN_NOISES), (1,)))]
            xb = corrupt(split["train_x"][idx], sigma, cfg["clamp_noise"])
            model.train_step(xb, onehot[idx], opt, steps=cfg["infer_steps"], lam=0.0)
        val_acc = accuracy(model.predict(val_x_noisy), split["val_y"])
        history.append(val_acc)
        _log(f"  noisy-train epoch {len(history)}: noisy val acc {val_acc:.4f}")
        if val_acc > best + cfg["min_delta"]:
            best, stale = val_acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return history


def evaluate(clean, noisy, keys_h, keys_y, split, cfg, seed, noise) -> list[dict[str, Any]]:
    num_classes = cfg["num_classes"]
    x_val = corrupt(split["val_x"], noise, cfg["clamp_noise"])
    x_test = corrupt(split["test_x"], noise, cfg["clamp_noise"])
    y_val, y_test = split["val_y"], split["test_y"]
    rows: list[dict[str, Any]] = []

    def add(method: str, acc: float, chosen: str = "") -> None:
        rows.append({"seed": seed, "noise": noise, "method": method, "accuracy": acc, "chosen": chosen})

    add("clean_feedforward", accuracy(clean.predict(x_test), y_test))
    add(BASELINE, accuracy(noisy.predict(x_test), y_test))

    protos = class_hidden_prototypes(noisy, split["train_x"], split["train_y"], num_classes)
    memories = {"prototypes": protos, "random": random_memory_like(protos, seed)}
    grid = [{"lam": a, "beta": b} for a, b in itertools.product(cfg["pull_lams"], cfg["pull_betas"])]
    for mem_name, memory in memories.items():

        def pulled(x, c, memory=memory):
            return noisy.predict_pulled(
                x, memory, beta=c["beta"], lam=c["lam"], steps=cfg["pull_steps"], score="l2"
            )

        best = _select(grid, lambda c: pulled(x_val, c), y_val)
        add(f"noisetrained_pull[{mem_name},l2]", accuracy(pulled(x_test, best), y_test),
            f"lam={best['lam']:g},beta={best['beta']:g}")

    h_test = features(noisy, x_test)
    add("noisetrained_hidden_nearest_mean_l2", accuracy(torch.cdist(h_test, protos).argmin(-1), y_test))
    add("noisetrained_hidden_knn5", accuracy(_knn_predict(h_test, keys_h, keys_y, 5, num_classes), y_test))
    return rows


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    _log("loading MNIST (slow: converts 70k images one by one)")
    train_x, train_y = load_mnist(cfg["data_dir"], train=True)
    test_x, test_y = load_mnist(cfg["data_dir"], train=False)
    split = prepare_splits(
        {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}, cfg
    )
    in_dim, num_classes = split["train_x"].shape[1], cfg["num_classes"]

    rows: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    for seed in cfg["seeds"]:
        torch.manual_seed(seed)
        clean = MiniPC(in_dim, cfg["hidden"], num_classes)
        hist = fit(clean, split, cfg, lam=0.0)
        training.append({"seed": seed, "model": "clean", "epochs": len(hist), "best_val_acc": max(hist)})
        _log(f"seed {seed} clean model: {len(hist)} epochs, best val {max(hist):.4f} (compare with v1)")

        torch.manual_seed(seed)
        noisy = MiniPC(in_dim, cfg["hidden"], num_classes)
        val_noisy = fixed_noisy_val(split, cfg, seed)
        hist = fit_noisy(noisy, split, cfg, val_noisy)
        training.append({"seed": seed, "model": "noisetrained", "epochs": len(hist), "best_val_acc": max(hist)})
        _log(f"seed {seed} noise-trained model: {len(hist)} epochs, best noisy val {max(hist):.4f}")

        gen = torch.Generator().manual_seed(seed)
        key_idx = torch.randperm(len(split["train_x"]), generator=gen)[: cfg["num_keys"]]
        keys_h, keys_y = features(noisy, split["train_x"][key_idx]), split["train_y"][key_idx]

        for noise in cfg["noise_levels"]:
            _log(f"seed {seed} evaluating noise {noise:g}")
            torch.manual_seed(1000 * seed + int(round(100 * noise)))  # same noise draws as v1
            rows.extend(evaluate(clean, noisy, keys_h, keys_y, split, cfg, seed, noise))

    return {"config": cfg, "rows": rows, "training": training,
            "summary": summarize_rows(rows, baseline=BASELINE)}


def main() -> None:
    parser = argparse.ArgumentParser(description="v2: noise-trained baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/mini_pc_mnist.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/mini_pc_noise_baseline"))
    parser.add_argument("--quick", action="store_true", help="Smoke run: 1 seed, 2 epochs, small test set.")
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG, **load_config(args.config)}
    if args.quick:
        cfg.update(QUICK_OVERRIDES)
        print("quick mode: smoke test only, do not report these numbers")
    results = run(cfg)

    print("\ntraining (epochs run, best validation accuracy):")
    for t in results["training"]:
        print(f"  seed {t['seed']} {t['model']}: {t['epochs']} epochs, val acc {t['best_val_acc']:.4f}")
    print(f"\ntest accuracy, mean +/- sample std over {len(cfg['seeds'])} seeds "
          f"(diff is paired vs {BASELINE})\n")
    print(format_markdown(results["summary"], cfg["noise_levels"]))

    out = args.output_dir
    write_json(out / "results.json", results)
    write_csv(out / "results.csv", results["rows"])
    write_csv(out / "training.csv", results["training"])
    write_csv(out / "summary.csv", results["summary"])
    (out / "summary.md").write_text(
        format_markdown(results["summary"], cfg["noise_levels"]) + "\n", encoding="utf-8")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
