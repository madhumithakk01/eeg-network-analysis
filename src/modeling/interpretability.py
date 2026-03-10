"""
Model interpretability: SHAP values and Network Collapse Index standalone analysis.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# NCI and recovery columns for standalone analysis.
NCI_COLUMNS = [
    "NCI_basic",
    "NCI_spectral",
    "NCI_fragmentation",
    "NCI_temporal",
    "recovery_score_efficiency",
]


def compute_shap_importance(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str],
    n_samples: int | None = 200,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values and return mean absolute SHAP per feature (global importance).
    Uses TreeExplainer for tree models. Input is converted to a DataFrame with correct
    column names so SHAP receives consistent, 1D-per-column data.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap is required. pip install shap")
    # Ensure SHAP receives a pandas DataFrame with correct column names (avoids "Per-column arrays must each be 1-dimensional")
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X), columns=feature_names)
    else:
        X = X.copy()
        if list(X.columns) != feature_names:
            X = X[feature_names] if all(c in X.columns for c in feature_names) else pd.DataFrame(X.values, columns=feature_names)
    if len(feature_names) != X.shape[1]:
        X = pd.DataFrame(X.values[:, : len(feature_names)], columns=feature_names)
    if n_samples is not None and len(X) > n_samples:
        idx = np.random.RandomState(42).choice(len(X), size=n_samples, replace=False)
        X_sample = X.iloc[idx].copy()
    else:
        X_sample = X.copy()
    try:
        explainer = shap.TreeExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    except Exception:
        X_small = X_sample.iloc[: min(50, len(X_sample))]
        mask = shap.maskers.Independent(X_small)
        explainer = shap.KernelExplainer(model.predict_proba, X_small, link="identity")
        shap_values = explainer.shap_values(X_sample.iloc[: min(100, len(X_sample))], nsamples=50)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    imp_df = imp_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return mean_abs, imp_df


def shap_summary_plot(
    model: Any,
    X: pd.DataFrame,
    feature_names: list[str],
    save_path: str,
    n_samples: int = 200,
) -> None:
    """Generate and save SHAP summary plot (beeswarm). TreeExplainer with DataFrame input."""
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # Ensure SHAP receives a pandas DataFrame with correct column names
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X), columns=feature_names)
    else:
        X = X.copy()
        if list(X.columns) != feature_names and all(c in X.columns for c in feature_names):
            X = X[feature_names]
        elif list(X.columns) != feature_names:
            X = pd.DataFrame(X.values[:, : len(feature_names)], columns=feature_names)
    if n_samples and len(X) > n_samples:
        idx = np.random.RandomState(42).choice(len(X), size=n_samples, replace=False)
        X_sample = X.iloc[idx].copy()
    else:
        X_sample = X.copy()
    try:
        explainer = shap.TreeExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        return
    shap.summary_plot(shap_values, X_sample, show=False)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.gcf().savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def nci_standalone_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    nci_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Report standalone ROC-AUC for each NCI/recovery column (univariate predictive power).
    """
    if nci_columns is None:
        nci_columns = [c for c in NCI_COLUMNS if c in X.columns]
    if not nci_columns:
        return pd.DataFrame()
    y_arr = np.asarray(y)
    rows = []
    for col in nci_columns:
        if col not in X.columns:
            continue
        x_col = X[col].replace([np.inf, -np.inf], np.nan).fillna(X[col].median())
        if x_col.nunique() < 2:
            rows.append({"feature": col, "roc_auc": 0.5})
            continue
        try:
            auc = roc_auc_score(y_arr, x_col)
        except ValueError:
            auc = 0.5
        rows.append({"feature": col, "roc_auc": float(auc)})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
