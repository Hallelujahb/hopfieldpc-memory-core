"""HopfieldPC Memory Core.

Standalone modern Hopfield memory module for validating associative retrieval
before FabricPC integration.
"""

from .api import HopfieldMemoryOutput, HopfieldShapeContract, HopfieldShapeError
from .coupling import (
    MemoryBank,
    banks_energy,
    banks_grad,
    effective_lam,
    estimate_top_curvature,
    hopfield_curvature_bounds,
    hopfield_energy_grad,
    hopfield_smoothness,
    hopfield_trace,
    joint_energy,
    pull_step,
    safe_step_size,
    settle,
    settle_banks,
)
from .diagnostics import (
    argmax_memory_index,
    attention_entropy,
    build_retrieval_diagnostics,
    max_attention_weight,
    retrieval_distance,
    selected_memory_distance,
    topk_mass,
    topk_recall,
)
from .energy import hopfield_energy
from .functional import hopfield_retrieve, multi_step_retrieve
from .memory import HopfieldMemory

__all__ = [
    "HopfieldMemory",
    "HopfieldMemoryOutput",
    "HopfieldShapeContract",
    "HopfieldShapeError",
    "MemoryBank",
    "argmax_memory_index",
    "attention_entropy",
    "banks_energy",
    "banks_grad",
    "build_retrieval_diagnostics",
    "effective_lam",
    "estimate_top_curvature",
    "hopfield_curvature_bounds",
    "hopfield_energy",
    "hopfield_energy_grad",
    "hopfield_retrieve",
    "hopfield_smoothness",
    "hopfield_trace",
    "joint_energy",
    "max_attention_weight",
    "multi_step_retrieve",
    "pull_step",
    "retrieval_distance",
    "safe_step_size",
    "selected_memory_distance",
    "settle",
    "settle_banks",
    "topk_mass",
    "topk_recall",
]