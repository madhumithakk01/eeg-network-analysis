"""
Cross-validation evaluation and metrics for patient-level ML.

Stratified 5-fold CV, per-fold and aggregate metrics (ROC-AUC, accuracy, F1,
sensitivity, specificity), ROC curves, and confusion matrices.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def stratified_kfold_splits(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
):
    """Yield (train_idx, val_idx) from StratifiedKFold."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    y_arr = np.asarray(y)
    for train_idx, val_idx in cv.split(X, y_arr):
        yield train_idx, val_idx


def find_best_threshold_youden(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> float:
    """
    Find probability threshold that maximizes Youden index J = sensitivity + specificity - 1.
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    best_j = -1.0
    best_t = 0.5
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_t = t
    return float(best_t)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict[str, float]:
    """Compute ROC-AUC, accuracy, F1, sensitivity, specificity."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }
    if y_proba is not None and len(np.unique(y_true)) >= 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = 0.5
    else:
        metrics["roc_auc"] = 0.5
    return metrics


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: callable,
    n_splits: int = 5,
    random_state: int = 42,
    scale: bool = True,
    use_youden_threshold: bool = True,
) -> dict[str, Any]:
    """
    Run Stratified K-Fold CV: fit scaler and model on train, evaluate on val per fold.
    If use_youden_threshold is True, the classification threshold is chosen to maximize
    Youden index (sensitivity + specificity - 1) on validation predictions.
    """
    X = X.copy()
    y = pd.Series(y).reset_index(drop=True)
    if X.index.tolist() != y.index.tolist():
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
    fold_metrics = []
    fold_roc_curves = []
    fold_confusions = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        if scale:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)
        else:
            X_train_s = np.asarray(X_train)
            X_val_s = np.asarray(X_val)
        X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_s = np.nan_to_num(X_val_s, nan=0.0, posinf=0.0, neginf=0.0)
        model = model_factory()
        model.fit(X_train_s, y_train)
        y_proba = model.predict_proba(X_val_s)[:, 1]
        if use_youden_threshold:
            thresh = find_best_threshold_youden(y_val.values, y_proba)
            y_pred = (y_proba >= thresh).astype(int)
        else:
            y_pred = (y_proba >= 0.5).astype(int)
        m = compute_metrics(y_val.values, y_pred, y_proba)
        m["fold"] = fold
        fold_metrics.append(m)
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        fold_roc_curves.append((fpr, tpr))
        fold_confusions.append(confusion_matrix(y_val, y_pred, labels=[0, 1]))
    df_metrics = pd.DataFrame(fold_metrics)
    agg = {
        "mean_roc_auc": df_metrics["roc_auc"].mean(),
        "std_roc_auc": df_metrics["roc_auc"].std(),
        "mean_accuracy": df_metrics["accuracy"].mean(),
        "mean_f1": df_metrics["f1"].mean(),
        "mean_sensitivity": df_metrics["sensitivity"].mean(),
        "mean_specificity": df_metrics["specificity"].mean(),
        "fold_metrics": df_metrics,
        "roc_curves": fold_roc_curves,
        "confusion_matrices": fold_confusions,
    }
    return agg
