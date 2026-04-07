#!/usr/bin/env python3
"""
Model-side bias checks for preprocessing attrition audit outputs.

Inputs:
  - preprocessing_patient_audit.csv (from run_preprocessing_audit.py)

Outputs:
  - preprocessing_metadata_probe.json
  - exclusion_sensitivity.csv
  - exclusion_sensitivity_by_hospital.csv (if Hospital exists)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _norm_outcome(x: str) -> int:
    s = str(x).strip().lower()
    if s in {"good", "0"}:
        return 0
    if s in {"poor", "1"}:
        return 1
    try:
        v = int(float(s))
        return 1 if v == 1 else 0
    except ValueError as e:
        raise ValueError(f"Unsupported Outcome value: {x!r}") from e


def _build_probe_features(df: pd.DataFrame) -> pd.DataFrame:
    base_numeric = [
        "n_segments_total",
        "n_segments_processed",
        "n_segments_failed",
        "n_windows_pre_qc",
        "n_windows_after_qc",
        "n_windows_rejected_qc",
        "retention_ratio",
        "n_connectivity_matrices",
    ]
    fail_cols = [c for c in df.columns if c.startswith("fail_")]
    cols = [c for c in base_numeric + fail_cols if c in df.columns]
    out = df[cols].copy()
    if "Hospital" in df.columns:
        out["Hospital"] = df["Hospital"].astype(str)
    if "Sex" in df.columns:
        out["Sex"] = df["Sex"].astype(str)
    return out


def _run_probe(df: pd.DataFrame, n_splits: int, random_state: int) -> dict:
    y = df["Outcome"].map(_norm_outcome).values
    x = _build_probe_features(df)

    numeric_cols = [c for c in x.columns if c not in {"Hospital", "Sex"}]
    categorical_cols = [c for c in ["Hospital", "Sex"] if c in x.columns]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    clf = Pipeline(
        steps=[
            ("pre", pre),
            ("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    p = cross_val_predict(clf, x, y, cv=skf, method="predict_proba")[:, 1]
    y_hat = (p >= 0.5).astype(int)

    return {
        "n_patients": int(len(df)),
        "outcome_positive_rate": float(np.mean(y)),
        "cv_roc_auc": float(roc_auc_score(y, p)),
        "cv_f1_at_0p5": float(f1_score(y, y_hat)),
        "cv_accuracy_at_0p5": float(accuracy_score(y, y_hat)),
        "features_used": list(x.columns),
        "interpretation_hint": (
            "If cv_roc_auc is high (>0.70), preprocessing metadata carries outcome signal; "
            "audit thresholds should be reviewed for potential selection bias."
        ),
    }


def _exclusion_sensitivity(df: pd.DataFrame, thresholds: List[int]) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        keep = df[df["n_windows_after_qc"] >= int(t)].copy()
        if keep.empty:
            rows.append(
                {
                    "min_windows_threshold": int(t),
                    "n_patients_kept": 0,
                    "keep_rate": 0.0,
                    "poor_rate_kept": np.nan,
                    "good_rate_kept": np.nan,
                }
            )
            continue
        out = keep["Outcome"].map(_norm_outcome).values
        poor_rate = float(np.mean(out))
        rows.append(
            {
                "min_windows_threshold": int(t),
                "n_patients_kept": int(len(keep)),
                "keep_rate": float(len(keep) / len(df)),
                "poor_rate_kept": poor_rate,
                "good_rate_kept": float(1.0 - poor_rate),
            }
        )
    return pd.DataFrame(rows)


def _exclusion_by_hospital(df: pd.DataFrame, thresholds: List[int]) -> pd.DataFrame:
    if "Hospital" not in df.columns:
        return pd.DataFrame()
    rows = []
    base_counts = df.groupby("Hospital").size().to_dict()
    for t in thresholds:
        keep = df[df["n_windows_after_qc"] >= int(t)].copy()
        kept_counts = keep.groupby("Hospital").size().to_dict()
        hospitals = sorted(set(base_counts.keys()) | set(kept_counts.keys()))
        for h in hospitals:
            n_base = int(base_counts.get(h, 0))
            n_kept = int(kept_counts.get(h, 0))
            rows.append(
                {
                    "min_windows_threshold": int(t),
                    "Hospital": h,
                    "n_base": n_base,
                    "n_kept": n_kept,
                    "keep_rate_within_hospital": float(n_kept / n_base) if n_base > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run preprocessing bias probe and exclusion sensitivity checks.")
    parser.add_argument(
        "--patient-audit-csv",
        type=str,
        required=True,
        help="Path to preprocessing_patient_audit.csv",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for bias checks.")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0,100,200,300,400,500,800,1000",
        help='Comma-separated minimum window thresholds, e.g. "0,100,300,500"',
    )
    parser.add_argument("--n-splits", type=int, default=5, help="CV folds for metadata probe model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.patient_audit_csv)
    df = df.dropna(subset=["Outcome"]).copy()

    thresholds = [int(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    probe = _run_probe(df, n_splits=args.n_splits, random_state=args.seed)
    probe_path = os.path.join(args.output_dir, "preprocessing_metadata_probe.json")
    with open(probe_path, "w", encoding="utf-8") as f:
        json.dump(probe, f, indent=2)

    sens = _exclusion_sensitivity(df, thresholds)
    sens_path = os.path.join(args.output_dir, "exclusion_sensitivity.csv")
    sens.to_csv(sens_path, index=False)

    by_hosp = _exclusion_by_hospital(df, thresholds)
    by_hosp_path = os.path.join(args.output_dir, "exclusion_sensitivity_by_hospital.csv")
    if not by_hosp.empty:
        by_hosp.to_csv(by_hosp_path, index=False)

    print(f"Saved: {probe_path}")
    print(f"Saved: {sens_path}")
    if not by_hosp.empty:
        print(f"Saved: {by_hosp_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

