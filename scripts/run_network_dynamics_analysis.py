#!/usr/bin/env python3
"""
Network dynamics / trajectory analysis: compare graph metric evolution between Good and Poor outcome.

Loads window-level graph features per patient from intermediate/graph_features/,
normalizes time to [0, 1], resamples to 200 steps, computes group mean ± std trajectories
for Good (CPC 1–2) vs Poor (CPC 3–5), and saves trajectory plots and summary statistics.

Outputs:
  analysis/network_dynamics/
    global_efficiency_trajectory.png
    clustering_coefficient_trajectory.png
    largest_component_ratio_trajectory.png
    n_connected_components_trajectory.png
    strength_mean_trajectory.png
    network_collapse_statistics.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import (
    AUDIT_PATH,
    GRAPH_FEATURES_DIR,
    NETWORK_DYNAMICS_OUTPUT_PATH,
    PATIENT_TEMPORAL_DATASET_PATH,
)
from src.modeling.dataset_loader import load_patient_outcomes
from src.temporal_analysis.trajectory_analysis import (
    load_patient_features,
    run_trajectory_analysis,
    TRAJECTORY_METRICS,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare network metric trajectories between Good and Poor neurological outcome.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to patient_temporal_dataset.parquet (default: PATIENT_TEMPORAL_DATASET_PATH).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to metadata CSV for outcome (default: AUDIT_PATH).",
    )
    parser.add_argument(
        "--graph-features-dir",
        type=str,
        default=None,
        help="Directory with *_features.npy (default: GRAPH_FEATURES_DIR).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: NETWORK_DYNAMICS_OUTPUT_PATH).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=200,
        help="Resampled time steps (default: 200).",
    )
    args = parser.parse_args()

    data_path = args.data or PATIENT_TEMPORAL_DATASET_PATH
    metadata_path = args.metadata or AUDIT_PATH
    graph_features_dir = args.graph_features_dir or GRAPH_FEATURES_DIR
    output_dir = args.output_dir or NETWORK_DYNAMICS_OUTPUT_PATH

    print("=" * 60)
    print("Network dynamics / trajectory analysis")
    print("=" * 60)
    print(f"Data:              {data_path}")
    print(f"Metadata:          {metadata_path}")
    print(f"Graph features:    {graph_features_dir}")
    print(f"Output:            {output_dir}")
    print(f"Resampled steps:   {args.n_steps}")

    if not os.path.isfile(data_path):
        print(f"Error: Dataset not found: {data_path}")
        return 1
    if not os.path.isdir(graph_features_dir):
        print(f"Error: Graph features directory not found: {graph_features_dir}")
        return 1

    patient_ids, y = load_patient_outcomes(data_path, metadata_path=metadata_path)
    # Keep only patients with valid window-level feature files
    available = []
    labels = []
    for i, pid in enumerate(patient_ids):
        if load_patient_features(pid, graph_features_dir) is not None:
            available.append(pid)
            labels.append(y.iloc[i])
    labels = np.array(labels, dtype=np.int64)
    print(f"\nPatients with window-level features: {len(available)} (Good={int(labels.sum())}, Poor={len(labels) - int(labels.sum())})")

    if len(available) < 10:
        print("Error: Too few patients with valid feature files.")
        return 1

    results = run_trajectory_analysis(
        patient_ids=available,
        labels=labels,
        graph_features_dir=graph_features_dir,
        output_dir=output_dir,
        n_steps=args.n_steps,
    )

    print(f"\nTrajectory plots saved to: {output_dir}")
    for name, _ in TRAJECTORY_METRICS:
        print(f"  {name}_trajectory.png")
    print(f"  network_collapse_statistics.csv")
    print(f"\nSummary: n_good={results['n_good']}, n_poor={results['n_poor']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
