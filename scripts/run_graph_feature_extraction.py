#!/usr/bin/env python3
"""
Graph feature extraction stage: sparse connectivity → per-window feature vectors.

Reads <patient_id>_sparse.npy from intermediate/sparse_connectivity/, builds a
weighted graph per window, extracts ~40 graph-theoretic features, and writes
<patient_id>_features.npy to intermediate/graph_features/.

Does not modify earlier pipeline stages. Supports --patient-split for parallel
workers and resume (skips patients that already have a features file).

  python scripts/run_graph_feature_extraction.py --patient-split patient_split_1.csv
"""

import argparse
import os
import sys
import time

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import BATCH_FOLDER, GRAPH_FEATURES_DIR, SPARSE_CONNECTIVITY_DIR
from src.data_loading.patient_list import load_patient_ids
from src.graph_features.dataset_processor import process_patient_sparse_file
from src.graph_features.feature_extractor import get_feature_count


def process_one_patient(
    patient_id: str,
    sparse_dir: str,
    features_dir: str,
) -> dict:
    """
    Load sparse .npy for one patient, extract features, save. Skip if output exists.

    Returns
    -------
    dict
        processed, skipped, n_windows, n_features, elapsed_sec, output_path, error
    """
    sparse_name = f"{patient_id}_sparse.npy"
    features_name = f"{patient_id}_features.npy"
    sparse_path = os.path.join(sparse_dir, sparse_name)
    features_path = os.path.join(features_dir, features_name)

    if os.path.isfile(features_path):
        return {
            "processed": False,
            "skipped": True,
            "n_windows": 0,
            "n_features": 0,
            "elapsed_sec": 0.0,
            "output_path": features_path,
            "error": None,
        }

    if not os.path.isfile(sparse_path):
        return {
            "processed": False,
            "skipped": False,
            "n_windows": 0,
            "n_features": 0,
            "elapsed_sec": 0.0,
            "output_path": None,
            "error": f"sparse file not found: {sparse_path}",
        }

    t0 = time.perf_counter()
    try:
        features = process_patient_sparse_file(patient_id, sparse_path)
    except Exception as e:
        return {
            "processed": False,
            "skipped": False,
            "n_windows": 0,
            "n_features": 0,
            "elapsed_sec": time.perf_counter() - t0,
            "output_path": None,
            "error": str(e),
        }

    n_windows, n_features = features.shape
    if np.any(~np.isfinite(features)):
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = features.astype(np.float32)

    os.makedirs(features_dir, exist_ok=True)
    try:
        np.save(features_path, features)
    except Exception as e:
        return {
            "processed": False,
            "skipped": False,
            "n_windows": n_windows,
            "n_features": n_features,
            "elapsed_sec": time.perf_counter() - t0,
            "output_path": None,
            "error": f"save failed: {e!r}",
        }

    elapsed = time.perf_counter() - t0
    return {
        "processed": True,
        "skipped": False,
        "n_windows": n_windows,
        "n_features": n_features,
        "elapsed_sec": elapsed,
        "output_path": features_path,
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract graph-theoretic features from sparse connectivity (per patient split)."
    )
    parser.add_argument(
        "--patient-split",
        type=str,
        default=None,
        help="Path to patient split CSV (e.g. patient_split_1.csv). Resolved against BATCH_FOLDER.",
    )
    parser.add_argument(
        "--sparse-dir",
        type=str,
        default=None,
        help="Directory with *_sparse.npy (default: SPARSE_CONNECTIVITY_DIR).",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=None,
        help="Output directory for *_features.npy (default: GRAPH_FEATURES_DIR).",
    )
    args = parser.parse_args()

    if args.patient_split is None:
        print("Error: --patient-split is required (e.g. patient_split_1.csv).")
        return 1

    split_path = args.patient_split
    if not os.path.isabs(split_path):
        split_path = os.path.join(BATCH_FOLDER, split_path)
    if not os.path.isfile(split_path):
        print(f"Error: Patient split file not found: {split_path}")
        return 1

    sparse_dir = args.sparse_dir or SPARSE_CONNECTIVITY_DIR
    features_dir = args.features_dir or GRAPH_FEATURES_DIR
    n_features_expected = get_feature_count()

    patient_ids = load_patient_ids(split_path)
    print(f"Loaded {len(patient_ids)} patients from {split_path}")
    print(f"Sparse dir:   {sparse_dir}")
    print(f"Features dir: {features_dir}")
    print(f"Features per window: {n_features_expected}")
    print()

    n_processed = 0
    n_skipped = 0
    n_failed = 0
    progress_interval = 10

    for i, patient_id in enumerate(patient_ids):
        result = process_one_patient(
            patient_id=patient_id,
            sparse_dir=sparse_dir,
            features_dir=features_dir,
        )
        if result["processed"]:
            n_processed += 1
            print(
                f"  [{i+1}/{len(patient_ids)}] {patient_id}: "
                f"windows={result['n_windows']} features={result['n_features']} "
                f"time={result['elapsed_sec']:.2f}s -> {result['output_path']}"
            )
        elif result["skipped"]:
            n_skipped += 1
        else:
            n_failed += 1
            print(
                f"  [{i+1}/{len(patient_ids)}] {patient_id}: failed - {result['error']}"
            )

        if (i + 1) % progress_interval == 0:
            print(
                f"  --- Progress: {i+1}/{len(patient_ids)} | "
                f"Processed: {n_processed} Skipped: {n_skipped} Failed: {n_failed}"
            )

    print(f"Done. Processed: {n_processed}, Skipped: {n_skipped}, Failed: {n_failed}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
