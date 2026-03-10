#!/usr/bin/env python3
"""
Temporal feature generation: window-level graph features -> patient-level dataset.

Loads <patient_id>_features.npy from intermediate/graph_features/, computes
aggregated temporal descriptors and Network Collapse Index variants, and writes
rows to analysis/patient_temporal_dataset.parquet.

Supports --patient-split for parallel workers. Resume: skips patients already
present in the output Parquet (merge new rows into existing file).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from configs.config import (
    BATCH_FOLDER,
    GRAPH_FEATURES_DIR,
    PATIENT_TEMPORAL_DATASET_PATH,
)
from src.data_loading.patient_list import load_patient_ids
from src.temporal_analysis.dataset_temporal_builder import build_patient_row
from src.temporal_analysis.temporal_feature_aggregator import MIN_WINDOWS_TEMPORAL


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build patient-level temporal dataset and NCI from graph features."
    )
    parser.add_argument(
        "--patient-split",
        type=str,
        default=None,
        help="Path to patient split CSV (e.g. patient_split_1.csv). Resolved against BATCH_FOLDER.",
    )
    parser.add_argument(
        "--graph-features-dir",
        type=str,
        default=None,
        help="Directory with *_features.npy (default: GRAPH_FEATURES_DIR).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output Parquet path (default: PATIENT_TEMPORAL_DATASET_PATH).",
    )
    parser.add_argument(
        "--max-descriptors",
        type=int,
        default=320,
        help="Max aggregated temporal features per patient (default: 320).",
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable temporal smoothing before aggregation.",
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

    graph_features_dir = args.graph_features_dir or GRAPH_FEATURES_DIR
    output_path = args.output or PATIENT_TEMPORAL_DATASET_PATH
    use_smoothing = not args.no_smoothing

    patient_ids = load_patient_ids(split_path)
    print(f"Loaded {len(patient_ids)} patients from {split_path}")
    print(f"Graph features dir: {graph_features_dir}")
    print(f"Output: {output_path}")
    print(f"Max descriptors: {args.max_descriptors}  Smoothing: {use_smoothing}")
    print()

    # Resume: load existing dataset and skip patients already present
    existing_ids: set[str] = set()
    existing_df: pd.DataFrame | None = None
    if os.path.isfile(output_path):
        try:
            existing_df = pd.read_parquet(output_path)
            if "patient_id" in existing_df.columns:
                existing_ids = set(existing_df["patient_id"].astype(str).tolist())
            print(f"Resume: found {len(existing_ids)} patients already in output.")
        except Exception as e:
            print(f"Warning: could not load existing output: {e}")

    to_process = [p for p in patient_ids if p not in existing_ids]
    if not to_process:
        print("All patients in split already processed. Nothing to do.")
        return 0

    n_processed = 0
    n_skipped = 0
    n_failed = 0
    rows: list[dict] = []
    progress_interval = 10
    t0_total = time.perf_counter()

    for i, patient_id in enumerate(to_process):
        path = os.path.join(graph_features_dir, f"{patient_id}_features.npy")
        if not os.path.isfile(path):
            n_skipped += 1
            continue
        t0 = time.perf_counter()
        try:
            features = np.load(path)
        except Exception as e:
            n_failed += 1
            print(f"  [{i+1}/{len(to_process)}] {patient_id}: load failed - {e}")
            continue
        n_win = features.shape[0] if features.ndim >= 1 else 0
        if features.ndim != 2 or features.shape[1] != 40 or n_win < MIN_WINDOWS_TEMPORAL:
            n_skipped += 1
            continue
        row = build_patient_row(
            patient_id=patient_id,
            features=features,
            max_descriptors=args.max_descriptors,
            use_smoothing=use_smoothing,
        )
        elapsed = time.perf_counter() - t0
        if row is None:
            n_skipped += 1
            continue
        rows.append(row)
        n_processed += 1
        n_feat = len(row) - 2  # exclude patient_id, n_windows
        print(
            f"  [{i+1}/{len(to_process)}] {patient_id}: windows={n_win} "
            f"features={n_feat} time={elapsed:.2f}s"
        )
        if (i + 1) % progress_interval == 0:
            print(
                f"  --- Progress: {i+1}/{len(to_process)} | "
                f"Processed: {n_processed} Skipped: {n_skipped} Failed: {n_failed}"
            )

    # Merge with existing and save
    if rows:
        new_df = pd.DataFrame(rows)
        for c in new_df.select_dtypes(include=[np.floating]).columns:
            new_df[c] = new_df[c].fillna(0.0)
        if existing_df is not None and len(existing_df) > 0:
            # Align columns: union, fill missing with 0
            all_cols = sorted(set(existing_df.columns) | set(new_df.columns))
            existing_df = existing_df.reindex(columns=all_cols, fill_value=0.0)
            new_df = new_df.reindex(columns=all_cols, fill_value=0.0)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        combined.to_parquet(output_path, index=False)
        total_elapsed = time.perf_counter() - t0_total
        print(f"Done. Processed: {n_processed}, Skipped: {n_skipped}, Failed: {n_failed}")
        print(f"Total patients in dataset: {len(combined)}")
        print(f"Output: {output_path}  (shape ~{combined.shape})")
        print(f"Runtime: {total_elapsed:.2f}s")
    else:
        print(f"Done. Processed: 0, Skipped: {n_skipped}, Failed: {n_failed} (no new rows)")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
