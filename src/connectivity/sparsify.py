"""
Density-based sparsification of functional connectivity matrices.

Converts dense Pearson correlation matrices (n_windows × 19 × 19) into sparse
networks by retaining only the strongest (by absolute value) fraction of edges.
Uses partial sorting (argpartition) for performance at scale (~1.5M matrices).
"""

from __future__ import annotations

import numpy as np

# Fixed network size for I-CARE pipeline; upper triangle indices computed once.
N_NODES = 19
_N_EDGES = N_NODES * (N_NODES - 1) // 2  # 171
_TRI_I, _TRI_J = np.triu_indices(N_NODES, k=1)


def sparsify_connectivity_matrix(
    matrix: np.ndarray,
    density: float = 0.15,
) -> np.ndarray:
    """
    Sparsify a single connectivity matrix by retaining the strongest edges by density.

    Parameters
    ----------
    matrix : np.ndarray, shape (19, 19)
        Dense symmetric correlation matrix (diagonal typically ~1).
    density : float, optional
        Fraction of edges to retain (default 0.15). For 19 nodes, 171 edges → ~26 retained.

    Returns
    -------
    np.ndarray, shape (19, 19), dtype float32
        Sparse symmetric matrix with diagonal set to zero. Original signs preserved.

    Notes
    -----
    Uses argpartition (partial sort) to select top-k edges by absolute weight
    without fully sorting. Diagonal is set to zero; symmetry is preserved.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (N_NODES, N_NODES):
        raise ValueError(f"Expected shape ({N_NODES}, {N_NODES}), got {matrix.shape}")

    upper = matrix[_TRI_I, _TRI_J].copy()
    abs_upper = np.abs(upper)
    k = max(1, min(_N_EDGES, int(np.ceil(_N_EDGES * density))))

    # Partial sort: indices of the k largest (by absolute value)
    part_idx = np.argpartition(abs_upper, _N_EDGES - k)
    top_k_idx = part_idx[-k:]
    mask = np.zeros(_N_EDGES, dtype=bool)
    mask[top_k_idx] = True

    upper_sparse = np.zeros(_N_EDGES, dtype=np.float32)
    upper_sparse[mask] = upper[mask]

    out = np.zeros((N_NODES, N_NODES), dtype=np.float32)
    out[_TRI_I, _TRI_J] = upper_sparse
    out[_TRI_J, _TRI_I] = upper_sparse
    return out


def sparsify_connectivity_dataset(
    connectivity_dataset: np.ndarray,
    density: float = 0.15,
) -> np.ndarray:
    """
    Sparsify a batch of connectivity matrices in a vectorized way.

    Parameters
    ----------
    connectivity_dataset : np.ndarray, shape (n_windows, 19, 19)
        Dense connectivity matrices (e.g. Pearson correlation), float32.
    density : float, optional
        Fraction of edges to retain per matrix (default 0.15).

    Returns
    -------
    np.ndarray, shape (n_windows, 19, 19), dtype float32
        Sparse matrices: symmetric, diagonal zero, ~density edges retained.

    Notes
    -----
    Processes all windows using NumPy broadcasting and argpartition along the
    edge dimension. No Python loop over windows. Upper triangle indices are
    precomputed at module load.
    """
    data = np.asarray(connectivity_dataset, dtype=np.float32)
    if data.ndim != 3 or data.shape[1] != N_NODES or data.shape[2] != N_NODES:
        raise ValueError(
            f"Expected shape (n_windows, {N_NODES}, {N_NODES}), got {data.shape}"
        )

    n_windows = data.shape[0]
    # Extract upper triangle for all windows: (n_windows, 171)
    upper = data[:, _TRI_I, _TRI_J].copy()
    abs_upper = np.abs(upper)

    k = max(1, min(_N_EDGES, int(np.ceil(_N_EDGES * density))))

    # For each window, indices of the k largest edges (argpartition along axis=1)
    part_idx = np.argpartition(abs_upper, _N_EDGES - k, axis=1)
    top_k_idx = part_idx[:, -k:]  # (n_windows, k)

    # Build boolean mask (n_windows, 171): True for edges to keep
    mask = np.zeros((n_windows, _N_EDGES), dtype=bool)
    np.put_along_axis(mask, top_k_idx, True, axis=1)

    upper_sparse = np.zeros_like(upper)
    upper_sparse[mask] = upper[mask]

    out = np.zeros((n_windows, N_NODES, N_NODES), dtype=np.float32)
    out[:, _TRI_I, _TRI_J] = upper_sparse
    out[:, _TRI_J, _TRI_I] = upper_sparse
    return out


def validate_sparse_matrix(matrix: np.ndarray) -> None:
    """
    Validate a single sparse connectivity matrix for scientific correctness.

    Checks: symmetric, diagonal zero, dtype float32, no NaN, density ≈ 0.15.

    Parameters
    ----------
    matrix : np.ndarray, shape (19, 19)
        Sparse matrix to validate.

    Raises
    ------
    ValueError
        If any check fails.
    """
    if matrix.shape != (N_NODES, N_NODES):
        raise ValueError(f"Expected shape ({N_NODES}, {N_NODES}), got {matrix.shape}")
    if matrix.dtype != np.float32:
        raise ValueError(f"Expected dtype float32, got {matrix.dtype}")
    if not np.allclose(matrix, matrix.T, equal_nan=True):
        raise ValueError("Matrix is not symmetric")
    if not np.allclose(np.diag(matrix), 0.0):
        raise ValueError("Diagonal must be zero")
    if np.any(np.isnan(matrix)):
        raise ValueError("Matrix contains NaN")
    n_edges = np.count_nonzero(matrix[_TRI_I, _TRI_J])
    expected = int(np.ceil(_N_EDGES * 0.15))
    if abs(n_edges - expected) > 2:  # allow small tolerance
        raise ValueError(
            f"Expected ~{expected} edges (density 0.15), got {n_edges}"
        )


def count_edges_per_matrix(sparse_dataset: np.ndarray) -> int:
    """
    Return the number of non-zero upper-triangle edges in the first matrix.

    Used for logging; assumes all matrices in the dataset use the same density.
    """
    if sparse_dataset.size == 0:
        return 0
    return int(np.count_nonzero(sparse_dataset[0][_TRI_I, _TRI_J]))
