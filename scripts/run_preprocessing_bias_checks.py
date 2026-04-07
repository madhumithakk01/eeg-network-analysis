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
from scipy.stats import chi2_contingency
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


def _build_probe_pipeline(x: pd.DataFrame) -> Pipeline:
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

    return Pipeline(
        steps=[
            ("pre", pre),
            ("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )


def _bootstrap_ci_metric(
    y: np.ndarray, p: np.ndarray, metric_name: str, n_boot: int = 1000, seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        pb = p[idx]
        if metric_name == "roc_auc":
            if len(np.unique(yb)) < 2:
                continue
            vals.append(roc_auc_score(yb, pb))
        elif metric_name == "f1":
            vals.append(f1_score(yb, (pb >= 0.5).astype(int)))
        elif metric_name == "accuracy":
            vals.append(accuracy_score(yb, (pb >= 0.5).astype(int)))
    if not vals:
        return {"low": np.nan, "high": np.nan}
    return {
        "low": float(np.percentile(vals, 2.5)),
        "high": float(np.percentile(vals, 97.5)),
    }


def _run_probe(df: pd.DataFrame, n_splits: int, random_state: int, include_hospital: bool) -> dict:
    y = df["Outcome"].map(_norm_outcome).values
    x = _build_probe_features(df)
    if not include_hospital and "Hospital" in x.columns:
        x = x.drop(columns=["Hospital"])
    clf = _build_probe_pipeline(x)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    p = cross_val_predict(clf, x, y, cv=skf, method="predict_proba")[:, 1]
    y_hat = (p >= 0.5).astype(int)

    auc = float(roc_auc_score(y, p))
    f1 = float(f1_score(y, y_hat))
    acc = float(accuracy_score(y, y_hat))
    return {
        "n_patients": int(len(df)),
        "outcome_positive_rate": float(np.mean(y)),
        "include_hospital": bool(include_hospital),
        "cv_roc_auc": auc,
        "cv_f1_at_0p5": f1,
        "cv_accuracy_at_0p5": acc,
        "ci95_roc_auc": _bootstrap_ci_metric(y, p, metric_name="roc_auc", seed=random_state),
        "ci95_f1_at_0p5": _bootstrap_ci_metric(y, p, metric_name="f1", seed=random_state),
        "ci95_accuracy_at_0p5": _bootstrap_ci_metric(y, p, metric_name="accuracy", seed=random_state),
        "features_used": list(x.columns),
    }


def _exclusion_sensitivity(df: pd.DataFrame, thresholds: List[int]) -> pd.DataFrame:
    base_out = df["Outcome"].map(_norm_outcome).values
    base_poor_rate = float(np.mean(base_out))
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
                    "delta_poor_rate_vs_baseline": np.nan,
                    "p_value_outcome_shift_vs_baseline": np.nan,
                }
            )
            continue
        out = keep["Outcome"].map(_norm_outcome).values
        poor_rate = float(np.mean(out))
        dropped = df[df["n_windows_after_qc"] < int(t)].copy()
        p_val = np.nan
        if not dropped.empty:
            a = int((keep["Outcome"].map(_norm_outcome) == 1).sum())
            b = int((keep["Outcome"].map(_norm_outcome) == 0).sum())
            c = int((dropped["Outcome"].map(_norm_outcome) == 1).sum())
            d = int((dropped["Outcome"].map(_norm_outcome) == 0).sum())
            table = np.array([[a, b], [c, d]], dtype=int)
            if np.all(table.sum(axis=1) > 0) and np.all(table.sum(axis=0) > 0):
                _, p_val, _, _ = chi2_contingency(table)
        rows.append(
            {
                "min_windows_threshold": int(t),
                "n_patients_kept": int(len(keep)),
                "keep_rate": float(len(keep) / len(df)),
                "poor_rate_kept": poor_rate,
                "good_rate_kept": float(1.0 - poor_rate),
                "delta_poor_rate_vs_baseline": float(poor_rate - base_poor_rate),
                "p_value_outcome_shift_vs_baseline": float(p_val) if pd.notna(p_val) else np.nan,
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


def _fairness_gap_summary(by_hosp: pd.DataFrame) -> pd.DataFrame:
    if by_hosp.empty:
        return pd.DataFrame()
    out_rows = []
    for t, g in by_hosp.groupby("min_windows_threshold"):
        vals = g["keep_rate_within_hospital"].dropna().values
        gap = float(np.max(vals) - np.min(vals)) if len(vals) > 0 else np.nan
        out_rows.append({"min_windows_threshold": int(t), "max_hospital_keep_rate_gap": gap})
    return pd.DataFrame(out_rows)


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

    probe_without_hospital = _run_probe(
        df, n_splits=args.n_splits, random_state=args.seed, include_hospital=False
    )
    probe_with_hospital = _run_probe(
        df, n_splits=args.n_splits, random_state=args.seed, include_hospital=True
    )
    probe_delta = {
        "roc_auc_delta_with_minus_without_hospital": float(
            probe_with_hospital["cv_roc_auc"] - probe_without_hospital["cv_roc_auc"]
        ),
        "f1_delta_with_minus_without_hospital": float(
            probe_with_hospital["cv_f1_at_0p5"] - probe_without_hospital["cv_f1_at_0p5"]
        ),
    }
    probe_path = os.path.join(args.output_dir, "preprocessing_metadata_probe.json")
    with open(probe_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "without_hospital": probe_without_hospital,
                "with_hospital": probe_with_hospital,
                "delta": probe_delta,
                "interpretation_hint": (
                    "Use without_hospital AUC as primary preprocessing-bias signal. "
                    "Large positive delta suggests site confounding contribution."
                ),
            },
            f,
            indent=2,
        )

    sens = _exclusion_sensitivity(df, thresholds)
    sens_path = os.path.join(args.output_dir, "exclusion_sensitivity.csv")
    sens.to_csv(sens_path, index=False)

    by_hosp = _exclusion_by_hospital(df, thresholds)
    by_hosp_path = os.path.join(args.output_dir, "exclusion_sensitivity_by_hospital.csv")
    if not by_hosp.empty:
        by_hosp.to_csv(by_hosp_path, index=False)
    gap = _fairness_gap_summary(by_hosp)
    gap_path = os.path.join(args.output_dir, "exclusion_fairness_gap_summary.csv")
    if not gap.empty:
        gap.to_csv(gap_path, index=False)

    print(f"Saved: {probe_path}")
    print(f"Saved: {sens_path}")
    if not by_hosp.empty:
        print(f"Saved: {by_hosp_path}")
    if not gap.empty:
        print(f"Saved: {gap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

