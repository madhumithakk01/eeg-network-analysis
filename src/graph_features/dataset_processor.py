"""
Process per-patient sparse connectivity files into feature matrices.

Loads (n_windows, 19, 19) sparse .npy, builds a graph per window,
extracts a fixed-length feature vector per window, returns (n_windows, n_features).
"""

from __future__ import annotations

import numpy as np

from .feature_extractor import extract_graph_features, get_feature_count
from .graph_builder import build_graph


def process_patient_sparse_file(
    patient_id: str,
    sparse_file_path: str,
) -> np.ndarray:
    """
    Load sparse connectivity matrices for one patient and extract graph features per window.

    Parameters
    ----------
    patient_id : str
        Patient identifier (for logging only).
    sparse_file_path : str
        Path to <patient_id>_sparse.npy, shape (n_windows, 19, 19).

    Returns
    -------
    np.ndarray, shape (n_windows, n_features), dtype float32
        One feature vector per window. n_features = get_feature_count() (~40).
    """
    data = np.load(sparse_file_path)
    if data.ndim != 3 or data.shape[1] != 19 or data.shape[2] != 19:
        raise ValueError(
            f"Expected shape (n_windows, 19, 19) for {sparse_file_path}, got {data.shape}"
        )
    n_windows = data.shape[0]
    n_features = get_feature_count()
    out = np.zeros((n_windows, n_features), dtype=np.float32)

    for i in range(n_windows):
        matrix = data[i]
        G = build_graph(matrix)
        out[i] = extract_graph_features(G)

    # Replace any remaining NaN/Inf for robustness
    bad = ~np.isfinite(out)
    if np.any(bad):
        out[bad] = 0.0

    return out
