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

    Parameters
    ----------
    X : pd.DataFrame
        Numeric feature matrix.
    threshold : float
        Correlation threshold (default 0.95).

    Returns
    -------
    X_reduced : pd.DataFrame
        Matrix with correlated features removed.
    dropped : list[str]
        Names of dropped columns.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    corr = X.corr().values
    np.fill_diagonal(corr, 0)
    # Distance = 1 - |correlation| for clustering
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    if condensed.size == 0:
        return X, []
    Z = linkage(condensed, method="average")
    # Cut at height corresponding to threshold: clusters where max|corr| <= threshold
    # height in linkage is distance; we want clusters with dist >= 1 - threshold
    clusters = fcluster(Z, t=1.0 - threshold, criterion="distance")
    names = list(X.columns)
    to_keep = []
    for c_id in np.unique(clusters):
        idx = np.where(clusters == c_id)[0]
        if len(idx) == 1:
            to_keep.append(names[idx[0]])
            continue
        # Subset correlation matrix for this cluster
        sub = corr[np.ix_(idx, idx)]
        # Representative: feature with smallest max absolute correlation to others
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
