#!/usr/bin/env python3
"""
Aggregate preprocessing audit CSVs and produce bias/retention summaries.
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from configs.config import AUDIT_PATH


def _normalize_patient_id(x: str) -> str:
    s = str(x).strip()
    try:
        return str(int(s)).zfill(4)
    except ValueError:
        return s.zfill(4) if len(s) <= 4 else s


def main() -> int:
    parser = argparse.ArgumentParser(description="Build preprocessing retention/bias audit tables.")
    parser.add_argument(
        "--audit-glob",
        type=str,
        required=True,
        help='Glob for per-split preprocessing audit CSVs, e.g. "/path/windows_tag/preprocessing_audit_*.csv".',
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=AUDIT_PATH,
        help=f"Metadata CSV for Outcome/Hospital merge (default: {AUDIT_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for aggregated audit outputs.",
    )
    args = parser.parse_args()

    paths = sorted(glob(args.audit_glob))
    if not paths:
        print(f"No audit CSV files matched: {args.audit_glob}")
        return 1

    dfs = []
    for p in paths:
        try:
            d = pd.read_csv(p)
            d["source_file"] = os.path.basename(p)
            dfs.append(d)
        except Exception as e:
            print(f"Warning: failed reading {p}: {e}")
    if not dfs:
        print("No audit data loaded.")
        return 1

    df = pd.concat(dfs, ignore_index=True)
    df["patient_id"] = df["patient_id"].astype(str).apply(_normalize_patient_id)
    # Keep best/last row per patient if duplicates
    df = df.sort_values(by=["patient_id", "processed", "n_windows_after_qc"], ascending=[True, False, False])
    df = df.drop_duplicates(subset=["patient_id"], keep="first").reset_index(drop=True)

    meta = pd.read_csv(args.metadata)
    if "Patient" in meta.columns and "patient_id" not in meta.columns:
        meta = meta.rename(columns={"Patient": "patient_id"})
    meta["patient_id"] = meta["patient_id"].astype(str).apply(_normalize_patient_id)
    keep_cols = [c for c in ["patient_id", "Outcome", "Hospital", "Sex", "Age", "CPC"] if c in meta.columns]
    merged = df.merge(meta[keep_cols].drop_duplicates("patient_id"), on="patient_id", how="left")

    os.makedirs(args.output_dir, exist_ok=True)
    patient_csv = os.path.join(args.output_dir, "preprocessing_patient_audit.csv")
    merged.to_csv(patient_csv, index=False)

    # Outcome summary
    if "Outcome" in merged.columns:
        outcome_summary = (
            merged.groupby("Outcome", dropna=False)
            .agg(
                n_patients=("patient_id", "count"),
                mean_windows=("n_windows_after_qc", "mean"),
                median_windows=("n_windows_after_qc", "median"),
                low_data_lt300=("n_windows_after_qc", lambda s: int((s < 300).sum())),
                mean_retention=("retention_ratio", "mean"),
            )
            .reset_index()
        )
        outcome_summary.to_csv(os.path.join(args.output_dir, "preprocessing_outcome_summary.csv"), index=False)

    # Hospital summary
    if "Hospital" in merged.columns:
        hosp_summary = (
            merged.groupby("Hospital", dropna=False)
            .agg(
                n_patients=("patient_id", "count"),
                mean_windows=("n_windows_after_qc", "mean"),
                median_windows=("n_windows_after_qc", "median"),
                low_data_lt300=("n_windows_after_qc", lambda s: int((s < 300).sum())),
                mean_retention=("retention_ratio", "mean"),
            )
            .reset_index()
        )
        hosp_summary.to_csv(os.path.join(args.output_dir, "preprocessing_hospital_summary.csv"), index=False)

    # Failure reason totals
    fail_cols = [c for c in merged.columns if c.startswith("fail_")]
    if fail_cols:
        totals = merged[fail_cols].sum().reset_index()
        totals.columns = ["failure_reason", "count"]
        totals.to_csv(os.path.join(args.output_dir, "preprocessing_failure_reason_totals.csv"), index=False)

    print(f"Saved: {patient_csv}")
    print(f"Rows: {len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

