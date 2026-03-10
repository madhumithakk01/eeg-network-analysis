#!/usr/bin/env python3
"""
Temporal deep learning pipeline: train Temporal CNN on window-level graph features for outcome prediction.

Loads (n_windows, 40) per patient from existing *_features.npy in GRAPH_FEATURES_DIR.
No sequence truncation; all windows are used. Padding is applied per batch; masking ensures
padded timesteps never influence the model.
Uses patient-level stratified K-fold CV; no windows from the same patient leak across splits.
Designed for Google Colab with GPU; all data paths read from config (Drive).

Usage:
  python scripts/run_temporal_dl.py
  python scripts/run_temporal_dl.py --data /path/to/patient_temporal_dataset.parquet --graph-features-dir /path/to/graph_features --output-dir /path/to/results
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
    PATIENT_TEMPORAL_DATASET_PATH,
    TEMPORAL_DL_BATCH_SIZE,
    TEMPORAL_DL_EPOCHS,
    TEMPORAL_DL_OUTPUT_PATH,
)
from src.modeling.dataset_loader import load_patient_outcomes
from src.temporal_models.dataset import MIN_WINDOWS, validate_feature_file
from src.temporal_models.training import run_patient_cv, set_seeds


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Temporal CNN on window-level graph features (patient-level CV, variable-length sequences).",
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
        help="Path to metadata CSV for outcome merge (default: AUDIT_PATH).",
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
        help="Output directory for metrics and logs (default: TEMPORAL_DL_OUTPUT_PATH).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Batch size (default: {TEMPORAL_DL_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Epochs per fold (default: {TEMPORAL_DL_EPOCHS}).",
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
        help="Random state for CV and reproducibility (default: 42).",
    )
    args = parser.parse_args()

    set_seeds(args.random_state)

    data_path = args.data or PATIENT_TEMPORAL_DATASET_PATH
    metadata_path = args.metadata or AUDIT_PATH
    graph_features_dir = args.graph_features_dir or GRAPH_FEATURES_DIR
    output_dir = args.output_dir or TEMPORAL_DL_OUTPUT_PATH
    batch_size = args.batch_size or TEMPORAL_DL_BATCH_SIZE
    epochs = args.epochs or TEMPORAL_DL_EPOCHS

    print("=" * 60)
    print("Temporal CNN pipeline (window-level graph features)")
    print("=" * 60)
    print(f"Data (parquet):     {data_path}")
    print(f"Metadata:           {metadata_path}")
    print(f"Graph features dir: {graph_features_dir}")
    print(f"Output dir:         {output_dir}")
    print(f"Batch size: {batch_size}  Epochs: {epochs}  Folds: {args.n_folds}  Seed: {args.random_state}")

    if not os.path.isfile(data_path):
        print(f"Error: Dataset not found: {data_path}")
        return 1
    if not os.path.isdir(graph_features_dir):
        print(f"Error: Graph features directory not found: {graph_features_dir}")
        return 1

    patient_ids, y = load_patient_outcomes(
        data_path,
        metadata_path=metadata_path,
    )
    print(f"\nLoaded {len(patient_ids)} patients with outcomes from parquet/metadata.")

    # Dataset validation: verify each feature file shape (n_windows, 40), skip if n_windows < MIN_WINDOWS
    available = []
    labels = []
    seq_lengths = []
    for i, pid in enumerate(patient_ids):
        path = os.path.join(graph_features_dir, f"{pid}_features.npy")
        valid, msg = validate_feature_file(path, min_windows=MIN_WINDOWS, n_features=40)
        if not valid:
            continue
        available.append(pid)
        labels.append(y.iloc[i])
        arr = np.load(path)
        seq_lengths.append(arr.shape[0])

    if len(available) < 20:
        print(f"Error: Too few patients with valid window-level data ({len(available)}). Need >= 20.")
        return 1

    labels = np.array(labels, dtype=np.int64)
    seq_lengths = np.array(seq_lengths)

    # Sequence statistics
    print(f"\nDataset validation passed. Using {len(available)} patients.")
    print(f"  Sequence length: mean={seq_lengths.mean():.0f}, median={np.median(seq_lengths):.0f}, max={seq_lengths.max()}, min={seq_lengths.min()}")
    print(f"  Feature file shape: (n_windows, 40); MIN_WINDOWS={MIN_WINDOWS}")

    # Class distribution
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    print(f"  Class distribution: positive (Good)={n_pos}, negative (Poor)={n_neg}")

    # Padding verification: mask 1 = valid, 0 = padding; ensured in collate_patient_batch and model masked pooling
    print("  Padding: applied per batch to max length in batch; mask 1=valid, 0=pad; model uses masked pooling.")

    import torch
    device = args.device
    if device is None or device == "":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"\nDevice: {dev}")

    print("\nRunning stratified patient-level K-fold CV (Temporal CNN)...")
    results = run_patient_cv(
        patient_ids=available,
        y=labels,
        graph_features_dir=graph_features_dir,
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
    print("Temporal CNN CV summary (mean ± std)")
    print("=" * 60)
    print(f"  ROC-AUC:      {s['mean_roc_auc']:.4f} ± {s['std_roc_auc']:.4f}")
    print(f"  F1:           {s['mean_f1']:.4f}")
    print(f"  Sensitivity:  {s['mean_sensitivity']:.4f}")
    print(f"  Specificity:  {s['mean_specificity']:.4f}")
    print(f"  Accuracy:     {s['mean_accuracy']:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"  temporal_dl_cv_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
