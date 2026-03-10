#!/usr/bin/env python3
"""
Sparsification stage: dense connectivity → sparse connectivity.

Reads patient_id_connectivity.npy from intermediate/windows/, applies
density-based sparsification (top 15% edges by absolute correlation),
and writes patient_id_sparse.npy to intermediate/sparse_connectivity/.

Does not modify the preprocessing pipeline or overwrite existing dense files.
Supports --patient-split for parallel workers and resume (skips existing sparse outputs).

  python scripts/run_sparsification.py --patient-split patient_split_1.csv
  python scripts/run_sparsification.py --patient-split batches/patient_split_2.csv
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

from configs.config import (
    BATCH_FOLDER,
    SPARSE_CONNECTIVITY_DIR,
    SPARSE_DENSITY,
    WINDOWS_OUTPUT_DIR,
)
from src.connectivity.sparsify import (
    count_edges_per_matrix,
    sparsify_connectivity_dataset,
    validate_sparse_matrix,
)
from src.data_loading.patient_list import load_patient_ids


def process_one_patient(
    patient_id: str,
    windows_dir: str,
    sparse_dir: str,
    density: float,
) -> dict:
    """
    Load dense connectivity, sparsify, validate, and save. Skip if output exists.

    Returns
    -------
    dict
        processed (bool), skipped (bool), n_windows (int), n_edges (int),
        elapsed_sec (float), output_path (str | None), error (str | None).
    """
    dense_name = f"{patient_id}_connectivity.npy"
    sparse_name = f"{patient_id}_sparse.npy"
    dense_path = os.path.join(windows_dir, dense_name)
    sparse_path = os.path.join(sparse_dir, sparse_name)

    if os.path.isfile(sparse_path):
        return {
            "processed": False,
            "skipped": True,
            "n_windows": 0,
            "n_edges": 0,
            "elapsed_sec": 0.0,
            "output_path": sparse_path,
            "error": None,
        }

    if not os.path.isfile(dense_path):
        return {
            "processed": False,
            "skipped": False,
            "n_windows": 0,
            "n_edges": 0,
            "elapsed_sec": 0.0,
            "output_path": None,
            "error": f"dense file not found: {dense_path}",
        }

    t0 = time.perf_counter()
    try:
        dense = np.load(dense_path)
    except Exception as e:
        return {
            "processed": False,
            "skipped": False,
            "n_windows": 0,
            "n_edges": 0,
            "elapsed_sec": time.perf_counter() - t0,
            "output_path": None,
            "error": f"load failed: {e!r}",
        }

    if dense.ndim != 3 or dense.shape[1] != 19 or dense.shape[2] != 19:
        return {
            "processed": False,
            "skipped": False,
            "n_windows": dense.shape[0] if dense.ndim >= 1 else 0,
            "n_edges": 0,
            "elapsed_sec": time.perf_counter() - t0,
            "output_path": None,
            "error": f"unexpected shape {getattr(dense, 'shape', '?')}",
        }

    sparse = sparsify_connectivity_dataset(dense, density=density)
    validate_sparse_matrix(sparse[0])
    n_edges = count_edges_per_matrix(sparse)

    os.makedirs(sparse_dir, exist_ok=True)
    try:
        np.save(sparse_path, sparse)
    except Exception as e:
        return {
            "processed": False,
            "skipped": False,
            "n_windows": sparse.shape[0],
            "n_edges": n_edges,
            "elapsed_sec": time.perf_counter() - t0,
            "output_path": None,
            "error": f"save failed: {e!r}",
        }

    elapsed = time.perf_counter() - t0
    return {
        "processed": True,
        "skipped": False,
        "n_windows": sparse.shape[0],
        "n_edges": n_edges,
        "elapsed_sec": elapsed,
        "output_path": sparse_path,
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sparsify connectivity matrices for a patient split (density-based)."
    )
    parser.add_argument(
        "--patient-split",
        type=str,
        default=None,
        help="Path to patient split CSV (e.g. patient_split_1.csv). Resolved against BATCH_FOLDER.",
    )
    parser.add_argument(
        "--windows-dir",
        type=str,
        default=None,
        help="Directory with *_connectivity.npy (default: WINDOWS_OUTPUT_DIR).",
    )
    parser.add_argument(
        "--sparse-dir",
        type=str,
        default=None,
        help="Output directory for *_sparse.npy (default: SPARSE_CONNECTIVITY_DIR).",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help=f"Fraction of edges to retain (default: config SPARSE_DENSITY={SPARSE_DENSITY}).",
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

    windows_dir = args.windows_dir or WINDOWS_OUTPUT_DIR
    sparse_dir = args.sparse_dir or SPARSE_CONNECTIVITY_DIR
    density = args.density if args.density is not None else SPARSE_DENSITY

    patient_ids = load_patient_ids(split_path)
    print(f"Loaded {len(patient_ids)} patients from {split_path}")
    print(f"Windows dir: {windows_dir}")
    print(f"Sparse dir:  {sparse_dir}")
    print(f"Density:     {density}")
    print()

    n_processed = 0
    n_skipped = 0
    n_failed = 0
    progress_interval = 10

    for i, patient_id in enumerate(patient_ids):
        result = process_one_patient(
            patient_id=patient_id,
            windows_dir=windows_dir,
            sparse_dir=sparse_dir,
            density=density,
        )
        if result["processed"]:
            n_processed += 1
            print(
                f"  [{i+1}/{len(patient_ids)}] {patient_id}: "
                f"windows={result['n_windows']} edges={result['n_edges']} "
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

    print(
        f"Done. Processed: {n_processed}, Skipped: {n_skipped}, Failed: {n_failed}"
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
