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
    Uses TreeExplainer for tree models, else KernelExplainer with a sample.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap is required. pip install shap")
    X_arr = np.asarray(X)
    if n_samples is not None and len(X_arr) > n_samples:
        idx = np.random.RandomState(42).choice(len(X_arr), size=n_samples, replace=False)
        X_sample = X_arr[idx]
    else:
        X_sample = X_arr
    try:
        explainer = shap.TreeExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            # Binary: use positive class
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    except Exception:
        mask = shap.maskers.Independent(X_sample)
        explainer = shap.KernelExplainer(model.predict_proba, X_sample[:50], link="identity")
        shap_values = explainer.shap_values(X_sample[: min(100, len(X_sample))], nsamples=50)
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
    """Generate and save SHAP summary plot (beeswarm)."""
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        return
    X_arr = np.asarray(X)
    if n_samples and len(X_arr) > n_samples:
        idx = np.random.RandomState(42).choice(len(X_arr), size=n_samples, replace=False)
        X_sample = X_arr[idx]
    else:
        X_sample = X_arr
    try:
        explainer = shap.TreeExplainer(model, X_sample)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        return
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
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
