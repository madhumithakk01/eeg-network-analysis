"""Functional connectivity computation and sparsification."""

from .pearson import compute_connectivity_batch
from .sparsify import (
    count_edges_per_matrix,
    sparsify_connectivity_dataset,
    sparsify_connectivity_matrix,
    validate_sparse_matrix,
)

__all__ = [
    "compute_connectivity_batch",
    "count_edges_per_matrix",
    "sparsify_connectivity_dataset",
    "sparsify_connectivity_matrix",
    "validate_sparse_matrix",
]
