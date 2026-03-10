"""Temporal network analysis: aggregation and Network Collapse Index."""

from .dataset_temporal_builder import build_patient_row, build_temporal_dataset
from .feature_schema import (
    COLLAPSE_CRITICAL_INDICES,
    FRAGMENTATION_INDICES,
    GRAPH_FEATURE_NAMES,
    N_GRAPH_FEATURES,
    SPECTRAL_INDICES,
    STABILITY_FEATURE_INDICES,
)
from .network_collapse_index import (
    CollapseResult,
    compute_nci_basic,
    compute_nci_fragmentation,
    compute_nci_spectral,
    compute_nci_temporal_stability,
    compute_recovery_score,
    detect_collapse_change_point,
)
from .temporal_feature_aggregator import (
    MIN_WINDOWS_TEMPORAL,
    aggregate_temporal_features,
    get_aggregated_feature_count,
)

__all__ = [
    "aggregate_temporal_features",
    "build_patient_row",
    "build_temporal_dataset",
    "CollapseResult",
    "COLLAPSE_CRITICAL_INDICES",
    "compute_nci_basic",
    "compute_nci_fragmentation",
    "compute_nci_spectral",
    "compute_nci_temporal_stability",
    "compute_recovery_score",
    "detect_collapse_change_point",
    "FRAGMENTATION_INDICES",
    "get_aggregated_feature_count",
    "GRAPH_FEATURE_NAMES",
    "MIN_WINDOWS_TEMPORAL",
    "N_GRAPH_FEATURES",
    "SPECTRAL_INDICES",
    "STABILITY_FEATURE_INDICES",
]
