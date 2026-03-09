"""
Validation utilities for connectivity matrices.

Checks shape, symmetry, diagonal, value range, and absence of NaNs.
Used to verify preprocessing output before running full batch.
"""

import numpy as np

# Default tolerance for diagonal ~1 and symmetry checks
_DIAG_ATOL = 1e-4
_DIAG_RTOL = 1e-3
_SYMM_ATOL = 1e-5
_SYMM_RTOL = 1e-4


def validate_connectivity_matrix(matrix: np.ndarray) -> None:
    """
    Validate a single connectivity matrix (Pearson correlation).

    Checks:
        1. Shape is (n_channels, n_channels).
        2. Matrix is symmetric.
        3. Diagonal values are close to 1.
        4. All values in [-1, 1].
        5. No NaN or Inf.

    Parameters
    ----------
    matrix : np.ndarray
        Single connectivity matrix, shape (n_channels, n_channels).

    Raises
    ------
    ValueError
        If any check fails, with a clear message.
    """
    if matrix.ndim != 2:
        raise ValueError(
            f"connectivity matrix must be 2D, got ndim={matrix.ndim}"
        )
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"connectivity matrix must be square, got shape={matrix.shape}"
        )
    if np.any(np.isnan(matrix)):
        raise ValueError("connectivity matrix contains NaN")
    if np.any(np.isinf(matrix)):
        raise ValueError("connectivity matrix contains Inf")
    if np.any(matrix < -1.0) or np.any(matrix > 1.0):
        vmin, vmax = float(np.nanmin(matrix)), float(np.nanmax(matrix))
        raise ValueError(
            f"connectivity values must be in [-1, 1], got min={vmin}, max={vmax}"
        )
    if not np.allclose(matrix, matrix.T, atol=_SYMM_ATOL, rtol=_SYMM_RTOL):
        diff = np.abs(matrix - matrix.T)
        raise ValueError(
            f"connectivity matrix must be symmetric; max |M - M.T| = {np.max(diff)}"
        )
    diag = np.diag(matrix)
    if not np.allclose(diag, 1.0, atol=_DIAG_ATOL, rtol=_DIAG_RTOL):
        raise ValueError(
            f"diagonal must be close to 1, got min={np.min(diag)}, max={np.max(diag)}"
        )


def validate_connectivity_batch(batch: np.ndarray) -> None:
    """
    Validate a batch of connectivity matrices.

    Checks:
        - 3D shape (n_windows, n_channels, n_channels).
        - Each matrix is symmetric.
        - Diagonals close to 1, values in [-1, 1], no NaN/Inf.

    Parameters
    ----------
    batch : np.ndarray
        Connectivity batch, shape (n_windows, n_channels, n_channels).

    Raises
    ------
    ValueError
        If any check fails.
    """
    if batch.ndim != 3:
        raise ValueError(
            f"connectivity batch must be 3D (n_windows, n_ch, n_ch), got ndim={batch.ndim}"
        )
    n_w, c1, c2 = batch.shape
    if c1 != c2:
        raise ValueError(
            f"each matrix must be square, got shape ({n_w}, {c1}, {c2})"
        )
    if np.any(np.isnan(batch)):
        raise ValueError("connectivity batch contains NaN")
    if np.any(np.isinf(batch)):
        raise ValueError("connectivity batch contains Inf")
    if np.any(batch < -1.0) or np.any(batch > 1.0):
        vmin, vmax = float(np.nanmin(batch)), float(np.nanmax(batch))
        raise ValueError(
            f"connectivity values must be in [-1, 1], got min={vmin}, max={vmax}"
        )
    # Vectorized symmetry check: batch[w] == batch[w].T
    if not np.allclose(batch, batch.transpose(0, 2, 1), atol=_SYMM_ATOL, rtol=_SYMM_RTOL):
        raise ValueError("connectivity batch: one or more matrices are not symmetric")
    # Vectorized diagonal check: diag of each matrix ~ 1
    diags = np.diagonal(batch, axis1=1, axis2=2)
    if not np.allclose(diags, 1.0, atol=_DIAG_ATOL, rtol=_DIAG_RTOL):
        raise ValueError(
            f"connectivity batch: diagonals not close to 1 (min={np.min(diags)}, max={np.max(diags)})"
        )
