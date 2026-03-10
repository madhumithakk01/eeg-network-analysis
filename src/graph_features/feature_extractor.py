"""
Extract graph-theoretic feature vectors from NetworkX graphs.

Produces ~40 features in fixed order: global topology, node centrality
aggregates, fragmentation, edge weight distribution, spectral.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy import stats

# global_efficiency may live in efficiency_measures in some NetworkX versions
try:
    _global_efficiency = nx.global_efficiency
except AttributeError:
    from networkx.algorithms.efficiency_measures import global_efficiency as _global_efficiency

# Feature count and order are fixed for downstream ML.
N_FEATURES = 40


def _safe(fn, default: float = 0.0):
    """Return default on exception (e.g. empty graph, disconnected)."""
    try:
        v = fn()
        return float(v) if np.isfinite(v) else default
    except (ZeroDivisionError, nx.NetworkXError, ValueError, TypeError, KeyError):
        return default


def _agg(values, prefix: str) -> dict:
    """Aggregate stats for a sequence; return mean, std, max (and min for 4-tuple)."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0, f"{prefix}_max": 0.0}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)) if arr.size > 1 else 0.0,
        f"{prefix}_max": float(np.max(arr)),
    }


def _largest_component_subgraph(G: nx.Graph):
    """Largest connected component as subgraph; or G if connected."""
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return None
    comps = list(nx.connected_components(G))
    if not comps:
        return None
    largest = max(comps, key=len)
    return G.subgraph(largest).copy()


def _small_world_approx(G: nx.Graph, n_nodes: int, n_edges: int) -> float:
    """Approximate small-worldness: C/C_rand and L/L_rand; return ratio of ratios."""
    if n_edges == 0 or n_nodes < 2:
        return 0.0
    try:
        C = nx.average_clustering(G, weight="weight")
        L = _safe(lambda: nx.average_shortest_path_length(G, weight="weight"))
        if L <= 0 or not np.isfinite(L):
            L = 1.0
        # Random graph with same n and m (approximate same density)
        G_rand = nx.gnm_random_graph(n_nodes, n_edges)
        C_rand = nx.average_clustering(G_rand)
        L_rand = nx.average_shortest_path_length(G_rand)
        if C_rand <= 0 or L_rand <= 0:
            return 0.0
        sigma = (C / C_rand) / (L / L_rand)
        return float(sigma) if np.isfinite(sigma) else 0.0
    except Exception:
        return 0.0


def _entropy_of_sizes(sizes: list) -> float:
    """Entropy of discrete distribution (component sizes)."""
    if not sizes:
        return 0.0
    p = np.array(sizes, dtype=np.float64) / sum(sizes)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def extract_graph_features(G: nx.Graph) -> np.ndarray:
    """
    Extract a fixed-length feature vector from a weighted graph.

    Parameters
    ----------
    G : nx.Graph
        Undirected graph with edge attribute 'weight'.

    Returns
    -------
    np.ndarray, shape (N_FEATURES,), dtype float32
        Feature vector. If graph is empty or computation fails, missing
        values are filled with 0.0. Order: global, centrality, fragmentation,
        edge distribution, spectral.
    """
    out = np.zeros(N_FEATURES, dtype=np.float32)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes == 0 or n_edges == 0:
        return out

    idx = 0

    # ----- 1. Global topology (~10) -----
    out[idx] = _safe(lambda: nx.density(G))
    idx += 1
    out[idx] = _safe(lambda: nx.average_clustering(G, weight="weight"))
    idx += 1
    out[idx] = _safe(lambda: nx.transitivity(G))
    idx += 1
    out[idx] = _safe(lambda: _global_efficiency(G))
    idx += 1
    sub = _largest_component_subgraph(G)
    avg_path = 0.0
    if sub is not None and sub.number_of_edges() > 0:
        avg_path = _safe(lambda: nx.average_shortest_path_length(sub, weight="weight"))
    out[idx] = avg_path
    idx += 1
    out[idx] = avg_path  # characteristic_path_length same as average_shortest_path
    idx += 1
    out[idx] = _safe(lambda: nx.degree_assortativity_coefficient(G))
    idx += 1
    # Modularity (API varies by NetworkX version)
    try:
        try:
            comms = nx.community.greedy_modularity_communities(G, weight="weight")
            mod = nx.community.modularity(G, comms, weight="weight")
        except AttributeError:
            from networkx.algorithms.community.modularity_max import greedy_modularity_communities
            from networkx.algorithms.community.quality import modularity
            comms = greedy_modularity_communities(G, weight="weight")
            mod = modularity(G, comms, weight="weight")
        out[idx] = float(mod) if np.isfinite(mod) else 0.0
    except Exception:
        out[idx] = 0.0
    idx += 1
    out[idx] = _small_world_approx(G, n_nodes, n_edges)
    idx += 1
    # one more global to reach ~10: e.g. num_edges / max_edges
    max_edges = n_nodes * (n_nodes - 1) / 2
    out[idx] = (n_edges / max_edges) if max_edges > 0 else 0.0
    idx += 1

    # ----- 2. Node centrality aggregates (~15) -----
    deg = [d for _, d in G.degree()]
    deg_agg = _agg(deg, "degree")
    out[idx] = deg_agg["degree_mean"]
    idx += 1
    out[idx] = deg_agg["degree_std"]
    idx += 1
    out[idx] = deg_agg["degree_max"]
    idx += 1
    strength = [sum(G[u][v].get("weight", 1) for v in G.neighbors(u)) for u in G]
    str_agg = _agg(strength, "strength")
    out[idx] = str_agg["strength_mean"]
    idx += 1
    out[idx] = str_agg["strength_std"]
    idx += 1
    out[idx] = str_agg["strength_max"]
    idx += 1
    try:
        bet = nx.betweenness_centrality(G, weight="weight")
        bv = list(bet.values())
        out[idx] = float(np.mean(bv))
        idx += 1
        out[idx] = float(np.max(bv))
        idx += 1
    except Exception:
        out[idx] = 0.0
        idx += 1
        out[idx] = 0.0
        idx += 1
    try:
        close = nx.closeness_centrality(G)
        cv = list(close.values())
        out[idx] = float(np.mean(cv))
        idx += 1
        out[idx] = float(np.max(cv))
        idx += 1
    except Exception:
        out[idx] = 0.0
        idx += 1
        out[idx] = 0.0
        idx += 1
    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
        ev = list(eig.values())
        out[idx] = float(np.mean(ev))
        idx += 1
        out[idx] = float(np.max(ev))
        idx += 1
    except Exception:
        out[idx] = 0.0
        idx += 1
        out[idx] = 0.0
        idx += 1

    # ----- 3. Fragmentation (~6) -----
    comps = list(nx.connected_components(G))
    n_comp = len(comps)
    out[idx] = float(n_comp)
    idx += 1
    largest_size = max(len(c) for c in comps) if comps else 0
    out[idx] = float(largest_size)
    idx += 1
    out[idx] = (largest_size / n_nodes) if n_nodes > 0 else 0.0
    idx += 1
    sizes = [len(c) for c in comps]
    out[idx] = float(np.var(sizes)) if len(sizes) > 1 else 0.0
    idx += 1
    out[idx] = _entropy_of_sizes(sizes)
    idx += 1
    out[idx] = float(np.min(sizes)) if sizes else 0.0  # min_component_size
    idx += 1

    # ----- 4. Edge weight distribution (~6) -----
    weights = [G[u][v].get("weight", 0) for u, v in G.edges()]
    if weights:
        w = np.array(weights, dtype=np.float64)
        w = w[np.isfinite(w)]
        if w.size > 0:
            out[idx] = float(np.mean(w))
            idx += 1
            out[idx] = float(np.std(w)) if w.size > 1 else 0.0
            idx += 1
            out[idx] = float(np.max(w))
            idx += 1
            out[idx] = float(np.min(w))
            idx += 1
            if w.size >= 3:
                out[idx] = float(stats.skew(w, nan_policy="omit"))
                idx += 1
                out[idx] = float(stats.kurtosis(w, nan_policy="omit"))
                idx += 1
            else:
                out[idx] = 0.0
                idx += 1
                out[idx] = 0.0
                idx += 1
            # edge weight entropy (binned)
            try:
                hist, _ = np.histogram(w, bins=min(10, max(2, w.size // 2)))
                hist = hist.astype(np.float64) / hist.sum()
                hist = hist[hist > 0]
                out[idx] = float(-np.sum(hist * np.log2(hist)))
            except Exception:
                out[idx] = 0.0
            idx += 1
        else:
            for _ in range(7):
                out[idx] = 0.0
                idx += 1
    else:
        for _ in range(7):
            out[idx] = 0.0
            idx += 1

    # ----- 5. Spectral (~5) -----
    try:
        adj = nx.to_numpy_array(G, nodelist=list(G.nodes()), weight="weight")
        evals = np.linalg.eigvalsh(adj)
        evals = np.real(evals)
        evals = evals[np.isfinite(evals)]
        if evals.size > 0:
            out[idx] = float(np.max(np.abs(evals)))  # spectral radius / largest eigenvalue
            idx += 1
            sorted_ev = np.sort(evals)[::-1]
            gap = (sorted_ev[0] - sorted_ev[1]) if len(sorted_ev) > 1 else 0.0
            out[idx] = float(gap)
            idx += 1
            out[idx] = float(np.trace(adj))
            idx += 1
            out[idx] = float(np.sum(np.abs(evals)))  # graph energy
            idx += 1
            out[idx] = float(sorted_ev[0])  # largest eigenvalue
            idx += 1
        else:
            for _ in range(5):
                out[idx] = 0.0
                idx += 1
    except Exception:
        for _ in range(5):
            out[idx] = 0.0
            idx += 1

    # Ensure we filled exactly N_FEATURES
    while idx < N_FEATURES:
        out[idx] = 0.0
        idx += 1
    return out[:N_FEATURES]


def get_feature_count() -> int:
    """Return the fixed number of features per window."""
    return N_FEATURES
