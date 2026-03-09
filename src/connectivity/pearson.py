"""
Vectorized Pearson correlation connectivity for EEG windows.

Computes connectivity matrices in batch (no Python loops over windows)
for use in the preprocessing pipeline. Output is float32 to reduce storage.
"""

import numpy as np


def compute_connectivity_batch(windows: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation matrices for a batch of EEG windows.

    Each window is normalized to zero mean and unit variance along the
    samples axis; then correlation between channel pairs is computed via
    the inner product over samples (equivalent to Pearson r).

    Parameters
    ----------
    windows : np.ndarray
        EEG windows, shape (n_windows, n_samples, n_channels).
        dtype: float32 or float64.

    Returns
    -------
    np.ndarray
        Connectivity matrices, shape (n_windows, n_channels, n_channels),
        dtype float32. Each matrix is symmetric with diagonal 1.0
        (up to numerical precision). Values in [-1, 1].
    """
    n_windows, n_samples, n_channels = windows.shape
    if n_samples < 2:
        # Not enough samples for correlation; return identity
        out = np.eye(n_channels, dtype=np.float32)
        out = np.broadcast_to(out[np.newaxis, :, :], (n_windows, n_channels, n_channels)).copy()
        return out

    # Center and scale along samples (axis=1); ddof=1 for sample std
    x = windows - windows.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True, ddof=1)
    std = np.where(std > 0, std, 1.0)
    x = x / std

    # Correlation: (1/(n-1)) * sum_s x[w,s,c]*x[w,s,d] -> (w,c,d)
    divisor = max(n_samples - 1, 1)
    corr = np.einsum("wsc,wsd->wcd", x, x, dtype=np.float64) / divisor
    corr = np.clip(corr, -1.0, 1.0)
    return corr.astype(np.float32)
