"""
Feature selection: correlation filtering and multi-method ranking (MI, RF, L1).

Removes highly correlated features (threshold 0.95) via hierarchical clustering,
then ranks by mutual information, Random Forest importance, and L1 (LogisticRegression)
and selects top k features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def remove_highly_correlated(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove features with correlation above threshold using hierarchical clustering.
    Keeps one representative feature per cluster (closest to cluster centroid).
    Constant features are dropped before correlation to avoid NaN/Inf in the distance matrix.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    X = X.copy()
    # Drop constant features to avoid NaN/Inf in correlation and distance matrix
    var = X.var()
    nonconst = var > 1e-12
    if not nonconst.any():
        return X, []
    X = X.loc[:, nonconst]
    if X.shape[1] < 2:
        return X, []

    corr = X.corr()
    corr = corr.replace([np.inf, -np.inf], np.nan).fillna(0)
    dist = 1 - np.abs(corr.values.astype(np.float64))
    dist = np.clip(dist, 0.0, None)
    np.fill_diagonal(dist, 0)
    dist = np.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
    condensed = squareform(dist, checks=False)
    if condensed.size == 0:
        return X, []
    if not np.isfinite(condensed).all():
        condensed = np.nan_to_num(condensed, nan=0.0, posinf=0.0, neginf=0.0)
    Z = linkage(condensed, method="average")
    clusters = fcluster(Z, t=1.0 - threshold, criterion="distance")
    names = list(X.columns)
    to_keep = []
    corr_arr = corr.values
    for c_id in np.unique(clusters):
        idx = np.where(clusters == c_id)[0]
        if len(idx) == 1:
            to_keep.append(names[idx[0]])
            continue
        sub = corr_arr[np.ix_(idx, idx)]
        max_abs = np.abs(sub).max(axis=1)
        rep = idx[np.argmin(max_abs)]
        to_keep.append(names[rep])
    dropped = [c for c in names if c not in to_keep]
    return X[to_keep], dropped


def rank_features_multi_method(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Rank features by Mutual Information, Random Forest importance, and L1 (Lasso).
    Returns a DataFrame with columns: feature, mi_score, rf_importance, l1_abs_coef, combined_rank.
    """
    X = X.copy()
    y = np.asarray(y, dtype=np.int64)
    names = list(X.columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Mutual information (discrete target)
    mi = mutual_info_classif(
        X_scaled, y, random_state=random_state, n_neighbors=min(5, len(X) // 2 - 1)
    )
    mi = np.nan_to_num(mi, nan=0.0)

    # Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10)
    rf.fit(X_scaled, y)
    rf_imp = rf.feature_importances_

    # L1 LogisticRegression
    lr = LogisticRegression(penalty="l1", solver="saga", C=0.1, random_state=random_state, max_iter=2000)
    lr.fit(X_scaled, y)
    l1_abs = np.abs(lr.coef_).ravel()

    # Normalize to [0,1] and average rank
    def _rank_norm(a: np.ndarray) -> np.ndarray:
        order = np.argsort(-a)
        rank = np.empty_like(a)
        rank[order] = np.arange(len(a))
        return rank / max(1, len(rank) - 1)

    r_mi = _rank_norm(mi)
    r_rf = _rank_norm(rf_imp)
    r_l1 = _rank_norm(l1_abs)
    combined = (r_mi + r_rf + r_l1) / 3.0

    df = pd.DataFrame({
        "feature": names,
        "mi_score": mi,
        "rf_importance": rf_imp,
        "l1_abs_coef": l1_abs,
        "combined_score": combined,
    })
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    return df


def select_top_k(
    ranking_df: pd.DataFrame,
    k: int = 40,
    feature_col: str = "feature",
) -> list[str]:
    """Return top k feature names from ranking DataFrame."""
    return ranking_df[feature_col].head(k).tolist()


# Core hypothesis-driven biomarkers to retain when possible (PART 5).
NCI_ALWAYS_INCLUDE = [
    "NCI_basic",
    "NCI_spectral",
    "NCI_fragmentation",
    "NCI_temporal",
    "recovery_score_efficiency",
]


def select_top_k_with_nci(
    ranking_df: pd.DataFrame,
    k: int,
    available_columns: list[str] | None = None,
    always_include: list[str] | None = None,
    feature_col: str = "feature",
) -> list[str]:
    """
    Return top k feature names, ensuring core NCI/recovery features are included if present.
    Fills remaining slots from ranking. If always_include is None, uses NCI_ALWAYS_INCLUDE.
    """
    if always_include is None:
        always_include = NCI_ALWAYS_INCLUDE
    ranked = ranking_df[feature_col].tolist()
    if available_columns is not None:
        ranked = [f for f in ranked if f in available_columns]
    selected = []
    for f in always_include:
        if f in ranked and f not in selected and (available_columns is None or f in available_columns):
            selected.append(f)
    for f in ranked:
        if f not in selected:
            selected.append(f)
        if len(selected) >= k:
            break
    return selected[:k]
