#!/usr/bin/env python3
"""
Connectivity graph deep learning pipeline: learn from raw adjacency sequences.

Loads {patient_id}_sparse.npy (n_windows, 19, 19) from SPARSE_CONNECTIVITY_DIR.
Uses graph encoder per window + LSTM over time; no graph feature vectors as primary input.
Patient-level stratified 5-fold CV; metrics and fold predictions saved to a new output directory.

Usage:
  python scripts/run_connectivity_dl.py
  python scripts/run_connectivity_dl.py --sparse-dir /path/to/sparse_connectivity --output-dir /path/to/results
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

# Ensure graph_models package is present (avoids confusing ModuleNotFoundError in Colab)
_graph_models_dir = os.path.join(_PROJECT_ROOT, "src", "graph_models")
_models_file = os.path.join(_graph_models_dir, "models.py")
if not os.path.isfile(_models_file):
    raise ImportError(
        f"Missing src/graph_models/models.py. Project root is: {_PROJECT_ROOT!r}. "
        "Ensure you run from the repo root (directory containing 'src' and 'scripts') and that "
        "src/graph_models/models.py exists (e.g. git pull or copy the full repo)."
    )

from configs.config import (
    AUDIT_PATH,
    CONNECTIVITY_DL_BATCH_SIZE,
    CONNECTIVITY_DL_EPOCHS,
    CONNECTIVITY_DL_OUTPUT_PATH,
    CONNECTIVITY_DL_STRIDE,
    PATIENT_TEMPORAL_DATASET_PATH,
    SPARSE_CONNECTIVITY_DIR,
)
from src.graph_models.dataset import validate_connectivity_file
from src.graph_models.training import run_patient_cv
from src.modeling.dataset_loader import load_patient_outcomes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train connectivity graph model (patient-level CV on raw adjacency sequences).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to patient_temporal_dataset.parquet for outcomes (default: PATIENT_TEMPORAL_DATASET_PATH).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to metadata CSV (default: AUDIT_PATH).",
    )
    parser.add_argument(
        "--sparse-dir",
        type=str,
        default=None,
        help="Directory with *_sparse.npy (default: SPARSE_CONNECTIVITY_DIR).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for metrics (default: CONNECTIVITY_DL_OUTPUT_PATH).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=f"Subsample every stride-th window; preserves full recording (default: {CONNECTIVITY_DL_STRIDE}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Batch size (default: {CONNECTIVITY_DL_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Epochs per fold (default: {CONNECTIVITY_DL_EPOCHS}).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Stratified K-fold splits (default: 5).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, cpu, or empty for auto.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state (default: 42).",
    )
    args = parser.parse_args()

    data_path = args.data or PATIENT_TEMPORAL_DATASET_PATH
    metadata_path = args.metadata or AUDIT_PATH
    sparse_dir = args.sparse_dir or SPARSE_CONNECTIVITY_DIR
    output_dir = args.output_dir or CONNECTIVITY_DL_OUTPUT_PATH
    stride = args.stride or CONNECTIVITY_DL_STRIDE
    batch_size = args.batch_size or CONNECTIVITY_DL_BATCH_SIZE
    epochs = args.epochs or CONNECTIVITY_DL_EPOCHS

    print("=" * 60)
    print("Connectivity graph deep learning pipeline")
    print("=" * 60)
    print(f"Data (outcomes):   {data_path}")
    print(f"Metadata:          {metadata_path}")
    print(f"Sparse connectivity: {sparse_dir}")
    print(f"Output:           {output_dir}")
    print(f"Stride: {stride}  Batch: {batch_size}  Epochs: {epochs}  Folds: {args.n_folds} (full sequences, no truncation)")

    if not os.path.isfile(data_path):
        print(f"Error: Dataset not found: {data_path}")
        return 1
    if not os.path.isdir(sparse_dir):
        print(f"Error: Sparse connectivity directory not found: {sparse_dir}")
        return 1

    patient_ids, y = load_patient_outcomes(data_path, metadata_path=metadata_path)
    available = []
    labels = []
    for i, pid in enumerate(patient_ids):
        path = os.path.join(sparse_dir, f"{pid}_sparse.npy")
        if validate_connectivity_file(path)[0]:
            available.append(pid)
            labels.append(y.iloc[i])
    labels = np.array(labels, dtype=np.int64)
    print(f"\nPatients with connectivity files: {len(available)} (Good={int(labels.sum())}, Poor={len(labels) - int(labels.sum())})")

    if len(available) < 20:
        print("Error: Too few patients with valid connectivity files.")
        return 1

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    print(f"Device: {dev}")

    print("\nRunning stratified patient-level K-fold CV...")
    results = run_patient_cv(
        patient_ids=available,
        y=labels,
        sparse_connectivity_dir=sparse_dir,
        stride=stride,
        n_splits=args.n_folds,
        batch_size=batch_size,
        epochs=epochs,
        lr=args.lr,
        device=dev,
        output_dir=output_dir,
        random_state=args.random_state,
    )

    s = results["summary"]
    print("\n" + "=" * 60)
    print("Connectivity DL CV summary (mean ± std)")
    print("=" * 60)
    print(f"  ROC-AUC:      {s['mean_roc_auc']:.4f} ± {s['std_roc_auc']:.4f}")
    print(f"  F1:           {s['mean_f1']:.4f}")
    print(f"  Sensitivity:  {s['mean_sensitivity']:.4f}")
    print(f"  Specificity:  {s['mean_specificity']:.4f}")
    print(f"  Accuracy:     {s['mean_accuracy']:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print("  connectivity_dl_cv_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
