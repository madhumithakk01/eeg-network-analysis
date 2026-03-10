"""
Build patient-level temporal dataset from window-level graph features.

Loads each patient's graph feature file, computes aggregated temporal descriptors
and all NCI variants, and assembles a single Parquet dataset for ML.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from .feature_schema import N_GRAPH_FEATURES
from .network_collapse_index import (
    CollapseResult,
    compute_nci_basic,
    compute_nci_fragmentation,
    compute_nci_spectral,
    compute_nci_temporal_stability,
    compute_recovery_score,
)
from .temporal_feature_aggregator import (
    MIN_WINDOWS_TEMPORAL,
    aggregate_temporal_features,
    get_aggregated_feature_count,
)


def build_patient_row(
    patient_id: str,
    features: np.ndarray,
    max_descriptors: int = 320,
    use_smoothing: bool = False,
) -> dict[str, Any] | None:
    """
    Build a single patient-level row: aggregated features + NCI variants + recovery.

    Parameters
    ----------
    patient_id : str
        Patient identifier.
    features : np.ndarray, shape (n_windows, 40)
        Window-level graph features.
    max_descriptors : int
        Max temporal descriptor features (trimmed from full set).
    use_smoothing : bool
        Apply smoothing before aggregation.

    Returns
    -------
    dict with keys patient_id, temporal feature names -> values, NCI_* columns,
    or None if window count < MIN_WINDOWS_TEMPORAL.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != N_GRAPH_FEATURES:
        return None
    n_win = features.shape[0]
    if n_win < MIN_WINDOWS_TEMPORAL:
        return None

    # Aggregated temporal descriptors
    agg_vec, agg_names = aggregate_temporal_features(
        features,
        use_smoothing=use_smoothing,
        max_descriptors=max_descriptors,
    )
    row = {"patient_id": patient_id}
    for i, name in enumerate(agg_names):
        if i < len(agg_vec):
            val = agg_vec[i]
            row[name] = float(val) if np.isfinite(val) else 0.0

    # NCI variants
    nci_basic: CollapseResult = compute_nci_basic(features)
    row["NCI_basic"] = nci_basic.collapse_score
    row["NCI_basic_collapse_time"] = nci_basic.collapse_time
    row["NCI_basic_severity"] = nci_basic.collapse_severity

    nci_spectral: CollapseResult = compute_nci_spectral(features)
    row["NCI_spectral"] = nci_spectral.collapse_score
    row["NCI_spectral_collapse_time"] = nci_spectral.collapse_time
    row["NCI_spectral_severity"] = nci_spectral.collapse_severity

    nci_fragmentation: CollapseResult = compute_nci_fragmentation(features)
    row["NCI_fragmentation"] = nci_fragmentation.collapse_score
    row["NCI_fragmentation_collapse_time"] = nci_fragmentation.collapse_time
    row["NCI_fragmentation_severity"] = nci_fragmentation.collapse_severity

    nci_temporal: CollapseResult = compute_nci_temporal_stability(features)
    row["NCI_temporal"] = nci_temporal.collapse_score
    row["NCI_temporal_collapse_time"] = nci_temporal.collapse_time
    row["NCI_temporal_severity"] = nci_temporal.collapse_severity

    # Recovery score (efficiency late/early)
    row["recovery_score_efficiency"] = compute_recovery_score(features, metric="global_efficiency")
    row["n_windows"] = n_win

    return row


def build_temporal_dataset(
    graph_features_dir: str,
    patient_ids: list[str],
    output_path: str,
    max_descriptors: int = 320,
    use_smoothing: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    """
    Build patient-level temporal dataset and save to Parquet.

    Parameters
    ----------
    graph_features_dir : str
        Directory containing <patient_id>_features.npy files.
    patient_ids : list[str]
        Patient IDs to include.
    output_path : str
        Output Parquet path.
    max_descriptors : int
        Max aggregated features per patient.
    use_smoothing : bool
        Smooth before aggregation.

    Returns
    -------
    df : pd.DataFrame
        One row per patient; columns = patient_id, temporal features, NCI_*.
    n_processed : int
        Number of patients successfully processed.
    n_skipped : int
        Number skipped (missing file or too few windows).
    """
    rows: list[dict[str, Any]] = []
    n_skipped = 0
    for patient_id in patient_ids:
        path = os.path.join(graph_features_dir, f"{patient_id}_features.npy")
        if not os.path.isfile(path):
            n_skipped += 1
            continue
        try:
            features = np.load(path)
        except Exception:
            n_skipped += 1
            continue
        row = build_patient_row(
            patient_id=patient_id,
            features=features,
            max_descriptors=max_descriptors,
            use_smoothing=use_smoothing,
        )
        if row is None:
            n_skipped += 1
            continue
        rows.append(row)

    if not rows:
        df = pd.DataFrame()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_parquet(output_path, index=False)
        return df, 0, n_skipped

    df = pd.DataFrame(rows)
    # Ensure no NaN in numeric columns
    for c in df.select_dtypes(include=[np.floating]).columns:
        df[c] = df[c].fillna(0.0)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df, len(rows), n_skipped
