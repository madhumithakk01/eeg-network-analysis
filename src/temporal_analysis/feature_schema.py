"""
Canonical names and indices for window-level graph features.

Must match the order produced by src.graph_features.feature_extractor (40 features).
Used by temporal aggregation and collapse index to identify columns by meaning.
"""

from __future__ import annotations

# Ordered names for each of the 40 graph feature columns (index = list position).
GRAPH_FEATURE_NAMES = [
    "density",
    "average_clustering",       # clustering coefficient (weighted)
    "transitivity",
    "global_efficiency",
    "average_shortest_path_length",
    "characteristic_path_length",
    "degree_assortativity",
    "modularity",
    "small_worldness",
    "edges_ratio",
    "degree_mean",
    "degree_std",
    "degree_max",
    "strength_mean",
    "strength_std",
    "strength_max",
    "betweenness_mean",
    "betweenness_max",
    "closeness_mean",
    "closeness_max",
    "eigenvector_mean",
    "eigenvector_max",
    "n_connected_components",
    "largest_component_size",
    "largest_component_ratio",
    "component_size_variance",
    "component_entropy",
    "min_component_size",
    "edge_weight_mean",
    "edge_weight_std",
    "edge_weight_max",
    "edge_weight_min",
    "edge_weight_skewness",
    "edge_weight_kurtosis",
    "edge_weight_entropy",
    "spectral_radius",
    "spectral_gap",
    "trace_adjacency",
    "graph_energy",
    "largest_eigenvalue",
]

# Indices for collapse-critical metrics (used by NCI and collapse indicators).
# Map semantic name -> column index.
COLLAPSE_CRITICAL_INDICES = {
    "global_efficiency": 3,
    "clustering_coefficient": 1,   # average_clustering
    "largest_component_ratio": 24,
    "path_length": 4,              # average_shortest_path_length (largest CC)
    "n_components": 22,
}

# Spectral feature indices (for SCI).
SPECTRAL_INDICES = {
    "spectral_radius": 35,
    "spectral_gap": 36,
    "graph_energy": 38,
    "largest_eigenvalue": 39,
}

# Fragmentation indices (for NFCI).
FRAGMENTATION_INDICES = {
    "n_components": 22,
    "largest_component_ratio": 24,
    "component_entropy": 26,
    "largest_component_size": 23,
}

# Stability/critical features for TNSI (temporal variance and derivative).
STABILITY_FEATURE_INDICES = [3, 1, 24, 13]  # global_efficiency, clustering, LCC_ratio, strength_mean

N_GRAPH_FEATURES = len(GRAPH_FEATURE_NAMES)
