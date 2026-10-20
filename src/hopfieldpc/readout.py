"""Heteroassociative Hopfield readout: keys are matched, values are returned.

hopfield_retrieve returns a blend of the same vectors it matched against (keys equal
values). A classifier readout needs the two to differ: match the query against stored
hidden representations (keys) and return their labels or prototypes (values):

    weights = softmax(beta * query @ keys.T)
    output  = weights @ values

With values = one_hot(labels) the output is a class distribution. With keys = values it
reduces to hopfield_retrieve. This is a soft k-nearest-neighbour classifier, so any
result obtained with it has to be compared with kNN and nearest-prototype baselines
(see benchmarks/readout_controls.py).
"""

from __future__ import annotations

import torch

from .api import HopfieldShapeError, validate_query_memory
from .functional import hopfield_retrieve


def hopfield_readout(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    beta: float = 1.0,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (output [B, value_dim], weights [B, N]).

    Args:
        query: [B, dim].
        keys: [N, dim] stored representations the query is matched against.
        values: [N, value_dim] what each key returns (labels, prototypes, ...).
        beta: retrieval sharpness, positive and finite.
        normalize: L2-normalize query and keys for scoring only (cosine similarity).
            Values are never normalized.
    """

    validate_query_memory(query=query, memory=keys)
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"values must be a torch.Tensor, got {type(values).__name__}.")
    if values.ndim != 2:
        raise HopfieldShapeError(
            f"values must be [num_memories, value_dim], got {tuple(values.shape)}."
        )
    if values.shape[0] != keys.shape[0]:
        raise HopfieldShapeError(
            f"values has {values.shape[0]} rows but keys has {keys.shape[0]}."
        )
    if values.device != keys.device:
        raise TypeError("values and keys must be on the same device.")

    _, weights = hopfield_retrieve(query, keys, beta=beta, normalize=normalize)
    output = torch.matmul(weights, values.to(dtype=weights.dtype))
    return output, weights


def one_hot_values(
    labels: torch.Tensor, num_classes: int, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """[N] integer labels -> [N, num_classes] one-hot values."""

    if labels.ndim != 1:
        raise HopfieldShapeError(f"labels must be [N], got {tuple(labels.shape)}.")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= num_classes):
        raise ValueError(f"labels must be in [0, {num_classes}).")
    return torch.nn.functional.one_hot(labels.long(), num_classes).to(dtype=dtype)


def class_prototypes(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Mean feature vector per class: [N, dim], [N] -> [num_classes, dim]."""

    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise HopfieldShapeError(
            "features must be [N, dim] and labels [N] with the same N, got "
            f"{tuple(features.shape)} and {tuple(labels.shape)}."
        )
    protos = torch.zeros(
        num_classes, features.shape[1], dtype=features.dtype, device=features.device
    )
    for c in range(num_classes):
        mask = labels == c
        if not bool(mask.any()):
            raise ValueError(f"class {c} has no samples.")
        protos[c] = features[mask].mean(dim=0)
    return protos
