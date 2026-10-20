"""Baselines that a Hopfield readout has to be compared with (MNIST pixels).

The existing mnist_prototype_retrieval benchmark classifies by the dot product with class
means. This script puts that rule next to the standard alternatives and next to a
heteroassociative Hopfield readout over stored training samples (a soft kNN):

  nearest_mean_l2   argmin ||x - mu_c||
  dot_argmax        argmax x . mu_c            (rule used by mnist_prototype_retrieval)
  cosine_argmax     argmax cos(x, mu_c)
  knn1, knn5        k nearest stored samples (L2)
  hopfield_knn      Hopfield readout over stored samples, cosine scoring, one-hot values

No hyperparameter is selected here: every beta in the config is reported. Differences in the
table are paired per-seed differences from knn5. Noise is Gaussian and clamped to [0, 1], as
in mnist_prototype_retrieval.

Run:
    python -m benchmarks.readout_controls
    python -m benchmarks.readout_controls --quick
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from benchmarks.mnist_prototype_retrieval import (
    add_gaussian_noise,
    build_prototypes,
    load_mnist,
)
from benchmarks.reporting import (
    format_markdown,
    load_config,
    summarize_rows,
    write_csv,
    write_json,
)
from hopfieldpc.readout import hopfield_readout, one_hot_values

DEFAULT_CONFIG: dict[str, Any] = {
    "seeds": [0, 1, 2, 3, 4],
    "num_classes": 10,
    "num_keys": 10000,
    "num_test": None,
    "noise_levels": [0.0, 0.3, 0.5, 1.0],
    "beta_values": [10.0, 30.0, 100.0],
    "knn_values": [1, 5],
    "chunk": 1000,
    "data_dir": "./data",
    "reports_dir": "reports/readout_controls",
}
QUICK_OVERRIDES: dict[str, Any] = {"seeds": [0], "num_keys": 2000, "num_test": 2000}
BASELINE = "knn5"


def predict_methods(
    x: torch.Tensor,
    protos: torch.Tensor,
    key_x: torch.Tensor,
    key_y: torch.Tensor,
    *,
    betas: list[float],
    ks: list[int],
    num_classes: int = 10,
    chunk: int = 1000,
) -> dict[str, torch.Tensor]:
    """Predicted labels [N] for every method."""

    values = one_hot_values(key_y, num_classes)
    protos_unit = F.normalize(protos, dim=-1)
    parts: dict[str, list[torch.Tensor]] = {}

    def add(name: str, pred: torch.Tensor) -> None:
        parts.setdefault(name, []).append(pred)

    for start in range(0, len(x), chunk):
        xb = x[start : start + chunk]
        add("nearest_mean_l2", torch.cdist(xb, protos).argmin(-1))
        add("dot_argmax", (xb @ protos.T).argmax(-1))
        add("cosine_argmax", (F.normalize(xb, dim=-1) @ protos_unit.T).argmax(-1))
        dist = torch.cdist(xb, key_x)
        for k in ks:
            idx = dist.topk(k, largest=False).indices
            add(f"knn{k}", values[idx].sum(1).argmax(-1))
        for beta in betas:
            out, _ = hopfield_readout(xb, key_x, values, beta=beta, normalize=True)
            add(f"hopfield_knn(beta={beta:g})", out.argmax(-1))

    return {name: torch.cat(p) for name, p in parts.items()}


def run_readout_controls(
    config: dict[str, Any] | None = None,
    data: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if data is None:
        train_x, train_y = load_mnist(cfg["data_dir"], train=True)
        test_x, test_y = load_mnist(cfg["data_dir"], train=False)
    else:
        train_x, train_y = data["train_x"], data["train_y"]
        test_x, test_y = data["test_x"], data["test_y"]

    num_classes = cfg["num_classes"]
    if cfg["num_test"] is not None:
        pick = torch.randperm(len(test_x), generator=torch.Generator().manual_seed(0))
        pick = pick[: cfg["num_test"]]
        test_x, test_y = test_x[pick], test_y[pick]

    protos = build_prototypes(train_x, train_y, num_classes)
    rows: list[dict[str, Any]] = []
    for seed in cfg["seeds"]:
        gen = torch.Generator().manual_seed(seed)
        key_idx = torch.randperm(len(train_x), generator=gen)[: cfg["num_keys"]]
        key_x, key_y = train_x[key_idx], train_y[key_idx]
        for noise in cfg["noise_levels"]:
            torch.manual_seed(1000 * seed + int(round(100 * noise)))
            x = test_x if noise == 0 else add_gaussian_noise(test_x, noise)
            preds = predict_methods(
                x, protos, key_x, key_y,
                betas=cfg["beta_values"], ks=cfg["knn_values"],
                num_classes=num_classes, chunk=cfg["chunk"],
            )
            for method, pred in preds.items():
                rows.append(
                    {
                        "seed": seed,
                        "noise": noise,
                        "method": method,
                        "accuracy": float((pred == test_y).float().mean()),
                    }
                )

    return {"config": cfg, "rows": rows, "summary": summarize_rows(rows, baseline=BASELINE)}


def save_reports(results: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    write_json(output_dir / "results.json", results)
    write_csv(output_dir / "results.csv", results["rows"])
    write_csv(output_dir / "summary.csv", results["summary"])
    table = format_markdown(results["summary"], results["config"]["noise_levels"])
    (output_dir / "summary.md").write_text(table + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST pixel baselines for the Hopfield readout.")
    parser.add_argument("--config", type=Path, default=Path("configs/readout_controls.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="Smoke run: 1 seed, small sets.")
    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG, **load_config(args.config)}
    if args.quick:
        cfg.update(QUICK_OVERRIDES)
        print("quick mode: smoke test only, do not report these numbers")
    results = run_readout_controls(cfg)

    print(f"keys={cfg['num_keys']} seeds={len(cfg['seeds'])} (mean +/- sample std; diff is paired vs {BASELINE})\n")
    print(format_markdown(results["summary"], cfg["noise_levels"]))
    out = args.output_dir or Path(cfg["reports_dir"])
    save_reports(results, out)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
