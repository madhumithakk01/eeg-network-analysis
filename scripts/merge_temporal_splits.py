#!/usr/bin/env python3
"""
Merge temporal feature Parquet files from multiple patient splits into one dataset.

Use when temporal feature generation was run separately per split (e.g. split_1, split_2, split_3)
and produced multiple Parquet files. Loads all inputs, concatenates row-wise, drops duplicate
patient_id (keeps first), and writes analysis/patient_temporal_dataset.parquet.

Example:
  python scripts/merge_temporal_splits.py analysis/temporal_split_1.parquet analysis/temporal_split_2.parquet analysis/temporal_split_3.parquet
  python scripts/merge_temporal_splits.py --input-dir analysis --glob "temporal_*.parquet" --output analysis/patient_temporal_dataset.parquet
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from configs.config import PATIENT_TEMPORAL_DATASET_PATH


def _normalize_patient_id(pid) -> str:
    s = str(pid).strip()
    try:
        return str(int(s)).zfill(4)
    except ValueError:
        return s


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge temporal feature Parquet files from multiple splits into one patient-level dataset."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input Parquet paths to merge (or use --input-dir + --glob).",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory to search for Parquet files (use with --glob).",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.parquet",
        help="Glob pattern for Parquet files under --input-dir (default: *.parquet).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output Parquet path (default: {PATIENT_TEMPORAL_DATASET_PATH}).",
    )
    args = parser.parse_args()

    paths = list(args.inputs)
    if args.input_dir and os.path.isdir(args.input_dir):
        pattern = os.path.join(args.input_dir, args.glob)
        paths.extend(glob.glob(pattern))
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        print("No input Parquet files found. Provide paths or --input-dir and --glob.")
        return 1

    output_path = args.output or PATIENT_TEMPORAL_DATASET_PATH
    print(f"Merge {len(paths)} file(s) -> {output_path}")
    for p in paths:
        print(f"  {p}")

    dfs = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")

    if not dfs:
        print("No data loaded.")
        return 1

    merged = pd.concat(dfs, ignore_index=True)
    before = len(merged)

    id_col = "patient_id"
    if "Patient" in merged.columns and id_col not in merged.columns:
        merged = merged.rename(columns={"Patient": id_col})
    if id_col not in merged.columns:
        print("No patient_id (or Patient) column found. Cannot deduplicate.")
        merged.to_parquet(output_path, index=False)
        print(f"Saved {len(merged)} rows to {output_path}")
        return 0

    merged[id_col] = merged[id_col].astype(str).str.strip().apply(_normalize_patient_id)
    merged = merged.drop_duplicates(subset=[id_col], keep="first")
    after = len(merged)
    n_dup = before - after

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    merged.to_parquet(output_path, index=False)
    n_patients = merged[id_col].nunique()
    print(f"Rows before dedup: {before}, after: {after} (dropped {n_dup} duplicate patient_id).")
    print(f"Unique patients: {n_patients}")
    print(f"Output: {output_path}  shape={merged.shape}")
    if n_patients < 250:
        print("WARNING: Fewer than 250 patients. Expected full cohort ~294.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
