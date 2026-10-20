"""Small helpers shared by the experiment scripts: config loading, summaries, report files."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _std(values: torch.Tensor) -> float:
    return float(values.std()) if len(values) > 1 else 0.0


def summarize_rows(
    rows: list[dict[str, Any]], baseline: str | None = None
) -> list[dict[str, Any]]:
    """Mean and sample std of accuracy over seeds for every (noise, method).

    rows: dicts with seed, noise, method and accuracy. If baseline names a method,
    each other method also gets the mean and std of the paired per-seed difference
    from the baseline.
    """

    grouped: dict[tuple[float, str], dict[int, float]] = {}
    for row in rows:
        grouped.setdefault((row["noise"], row["method"]), {})[row["seed"]] = row["accuracy"]

    summary = []
    for (noise, method), per_seed in grouped.items():
        seeds = sorted(per_seed)
        accs = torch.tensor([per_seed[s] for s in seeds])
        entry: dict[str, Any] = {
            "noise": noise,
            "method": method,
            "n_seeds": len(seeds),
            "mean": float(accs.mean()),
            "std": _std(accs),
        }
        base = grouped.get((noise, baseline)) if baseline else None
        if base is not None and method != baseline and set(base) == set(per_seed):
            diff = torch.tensor([per_seed[s] - base[s] for s in seeds])
            entry["delta_mean"] = float(diff.mean())
            entry["delta_std"] = _std(diff)
        summary.append(entry)
    return summary


def format_markdown(summary: list[dict[str, Any]], noise_levels: list[float]) -> str:
    methods = list(dict.fromkeys(e["method"] for e in summary))
    index = {(e["noise"], e["method"]): e for e in summary}
    lines = [
        "| method | " + " | ".join(f"noise {n:g}" for n in noise_levels) + " |",
        "|---|" + "---|" * len(noise_levels),
    ]
    for method in methods:
        cells = []
        for noise in noise_levels:
            entry = index.get((noise, method))
            if entry is None:
                cells.append("-")
                continue
            cell = f"{entry['mean']:.3f} +/- {entry['std']:.3f}"
            if "delta_mean" in entry:
                cell += f" (diff {entry['delta_mean']:+.3f} +/- {entry['delta_std']:.3f})"
            cells.append(cell)
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
