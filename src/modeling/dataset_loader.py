"""
Load and prepare patient-level temporal dataset for ML.

Loads Parquet, merges outcome labels from metadata if needed, separates
features (X) and target (y), and applies data quality checks.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd


# Columns that are never features (identifiers, metadata, target).
NON_FEATURE_COLUMNS = {"patient_id", "n_windows", "Outcome", "outcome", "CPC", "target", "label"}

# Target column name (binary: Good=1, Poor=0).
OUTCOME_COLUMN = "Outcome"

# Good outcome = CPC 1-2 -> 1; Poor = CPC 3-5 -> 0.
OUTCOME_GOOD_VALUES = {"Good", "good", "1", 1, "CPC1", "CPC2"}
OUTCOME_POOR_VALUES = {"Poor", "poor", "0", 0, "CPC3", "CPC4", "CPC5"}


def _normalize_patient_id(pid: Any) -> str:
    """Normalize patient ID to string, zero-padded 4 digits if numeric."""
    s = str(pid).strip()
    if not s:
        return s
    try:
        return str(int(s)).zfill(4)
    except ValueError:
        return s


def load_dataset(
    parquet_path: str,
    metadata_path: str | None = None,
    outcome_column: str = OUTCOME_COLUMN,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Load patient-level dataset and return X (features), y (outcome), feature_names.

    If outcome_column is not present in the Parquet, merges with metadata_path CSV
    on patient_id (column may be 'patient_id' or 'Patient'). Outcome is mapped to
    binary: Good/CPC1-2 -> 1, Poor/CPC3-5 -> 0.

    Parameters
    ----------
    parquet_path : str
        Path to patient_temporal_dataset.parquet.
    metadata_path : str, optional
        CSV with patient_id and Outcome/CPC for merge if outcome not in parquet.
    outcome_column : str
        Target column name (default 'Outcome').

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (patients x features).
    y : pd.Series
        Binary outcome (1 = Good, 0 = Poor).
    feature_names : list[str]
        Ordered feature column names.
    """
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Ensure patient identifier for merge
    if "patient_id" not in df.columns and "Patient" in df.columns:
        df = df.rename(columns={"Patient": "patient_id"})
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["patient_id"] = df["patient_id"].apply(_normalize_patient_id)

    # Get or merge outcome
    if outcome_column not in df.columns and "outcome" in df.columns:
        outcome_column = "outcome"
    if outcome_column not in df.columns:
        if metadata_path and os.path.isfile(metadata_path):
            meta = pd.read_csv(metadata_path)
            if "Patient" in meta.columns and "patient_id" not in meta.columns:
                meta = meta.rename(columns={"Patient": "patient_id"})
            if "patient_id" not in meta.columns:
                raise ValueError("Metadata must have patient_id or Patient column.")
            meta["patient_id"] = meta["patient_id"].astype(str).apply(_normalize_patient_id)
            if "Outcome" in meta.columns:
                out_col = "Outcome"
            elif "outcome" in meta.columns:
                out_col = "outcome"
            elif "CPC" in meta.columns:
                meta["Outcome"] = meta["CPC"].apply(
                    lambda x: 1 if str(x).strip() in ("1", "2") else 0
                )
                out_col = "Outcome"
            else:
                raise ValueError("Metadata must have Outcome, outcome, or CPC column.")
            if out_col != "Outcome":
                def _map_outcome(v):
                    s = str(v).strip()
                    if s in ("Good", "good", "1"):
                        return 1
                    if s in ("Poor", "poor", "0"):
                        return 0
                    try:
                        return 1 if int(float(v)) <= 2 else 0
                    except (ValueError, TypeError):
                        return np.nan
                meta["Outcome"] = meta[out_col].apply(_map_outcome)
            meta = meta[["patient_id", "Outcome"]].drop_duplicates(subset="patient_id")
            df = df.merge(meta, on="patient_id", how="left")
            outcome_column = "Outcome"
        else:
            raise ValueError(
                f"Outcome column '{outcome_column}' not in dataset and no metadata_path provided."
            )

    y_raw = df[outcome_column]
    # Map to binary
    y = pd.Series(index=df.index, dtype=int)
    for i, v in y_raw.items():
        v_str = str(v).strip()
        if v in (1, "1") or v_str in OUTCOME_GOOD_VALUES or v_str in ("1", "2"):
            y.loc[i] = 1
        elif v in (0, "0") or v_str in OUTCOME_POOR_VALUES or v_str in ("3", "4", "5"):
            y.loc[i] = 0
        else:
            try:
                y.loc[i] = 1 if int(float(v)) <= 2 else 0
            except (ValueError, TypeError):
                y.loc[i] = np.nan
    y = y.astype(float).astype("Int64")
    df = df.drop(columns=[outcome_column], errors="ignore")

    # Feature columns: exclude identifiers and known non-features
    exclude = NON_FEATURE_COLUMNS | {outcome_column}
    feature_names = [c for c in df.columns if c not in exclude and c != "patient_id"]
    # Only numeric
    X = df[feature_names].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.astype(np.float64)

    # Drop rows with missing target
    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True).astype(int)

    return X, y, feature_names


def data_quality_checks(
    X: pd.DataFrame,
    max_missing_frac: float = 0.5,
    remove_zero_variance: bool = True,
    impute_remaining: bool = True,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """
    Data quality: missing/inf detection, drop bad features, median imputation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    max_missing_frac : float
        Drop features with more than this fraction missing (default 0.5).
    remove_zero_variance : bool
        Drop zero-variance features.
    impute_remaining : bool
        Fill remaining NaN/Inf with column median.

    Returns
    -------
    X_clean : pd.DataFrame
        Cleaned feature matrix.
    dropped : list[str]
        Names of dropped columns.
    report : dict
        Counts of missing, inf, zero-var, etc.
    """
    report: dict[str, Any] = {
        "n_rows": len(X),
        "n_cols_initial": X.shape[1],
        "missing_per_col": {},
        "inf_per_col": {},
        "zero_var_cols": [],
        "high_missing_cols": [],
        "dropped": [],
    }

    X_clean = X.copy()
    for c in X_clean.columns:
        col = pd.to_numeric(X_clean[c], errors="coerce")
        report["missing_per_col"][c] = int(col.isna().sum())
        report["inf_per_col"][c] = int(np.isinf(col.replace({np.nan: 0})).sum())

    # Drop > max_missing_frac missing
    n = len(X_clean)
    for c in X_clean.columns:
        if report["missing_per_col"][c] / max(n, 1) > max_missing_frac:
            report["high_missing_cols"].append(c)
    X_clean = X_clean.drop(columns=report["high_missing_cols"], errors="ignore")

    # Replace inf with NaN then count
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

    if remove_zero_variance:
        var = X_clean.var()
        zero_var = var[var <= 1e-12].index.tolist()
        report["zero_var_cols"] = zero_var
        X_clean = X_clean.drop(columns=zero_var, errors="ignore")

    report["dropped"] = report["high_missing_cols"] + report["zero_var_cols"]

    if impute_remaining:
        for c in X_clean.columns:
            med = X_clean[c].median()
            if np.isnan(med) or not np.isfinite(med):
                med = 0.0
            X_clean[c] = X_clean[c].fillna(med)
        X_clean = X_clean.replace([np.inf, -np.inf], 0.0)

    report["n_cols_final"] = X_clean.shape[1]
    return X_clean, report["dropped"], report
