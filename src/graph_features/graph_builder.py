"""
Build NetworkX weighted graphs from sparse connectivity matrices.

Converts (19, 19) symmetric connectivity matrices into graph objects
with edge weights; zero-weight edges are removed to represent sparsity.
"""

from __future__ import annotations

import numpy as np
import networkx as nx


def build_graph(connectivity_matrix: np.ndarray) -> nx.Graph:
    """
    Build a weighted undirected graph from a connectivity matrix.

    Parameters
    ----------
    connectivity_matrix : np.ndarray, shape (19, 19)
        Symmetric matrix with diagonal 0; non-zero entries are edge weights.

    Returns
    -------
    nx.Graph
        Undirected graph with edge attribute 'weight'. Zero-weight edges
        are removed. Node labels 0..18 correspond to matrix indices.
    """
    matrix = np.asarray(connectivity_matrix, dtype=np.float64)
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError(f"Expected square matrix, got {matrix.shape}")

    # Ensure diagonal is zero (do not modify input)
    mat = matrix.copy()
    np.fill_diagonal(mat, 0.0)

    G = nx.from_numpy_array(mat)
    # Remove edges with zero (or negligible) weight to reflect sparse network
    to_remove = [(u, v) for u, v, w in G.edges(data="weight") if w == 0 or (w != w)]
    G.remove_edges_from(to_remove)
    return G
