"""
Threshold optimization for temporal DL (and binary classification) evaluation.

Evaluates performance across probability thresholds (0.0 to 1.0), finds optimal
threshold by Youden Index (Sensitivity + Specificity - 1) and by best F1,
and reports metrics at the optimized threshold.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def _metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute classification metrics at a given probability threshold."""
    y_pred = (y_proba >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    return {
        "threshold": float(threshold),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "youden": float(sens + spec - 1),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) else 0.0,
    }


def find_optimal_threshold_youden(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Find threshold that maximizes Youden Index J = Sensitivity + Specificity - 1.
    Returns (optimal_threshold, metrics_at_optimal).
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    best_j = -1.0
    best_t = 0.5
    best_metrics: Dict[str, float] = {}
    for t in thresholds:
        m = _metrics_at_threshold(y_true, y_proba, t)
        if m["youden"] > best_j:
            best_j = m["youden"]
            best_t = t
            best_metrics = m
    if not best_metrics:
        best_metrics = _metrics_at_threshold(y_true, y_proba, 0.5)
    try:
        best_metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        best_metrics["roc_auc"] = 0.5
    return float(best_t), best_metrics


def find_optimal_threshold_f1(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Find threshold that maximizes F1 score.
    Returns (optimal_threshold, metrics_at_optimal).
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    best_f1 = -1.0
    best_t = 0.5
    best_metrics: Dict[str, float] = {}
    for t in thresholds:
        m = _metrics_at_threshold(y_true, y_proba, t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = t
            best_metrics = m
    if not best_metrics:
        best_metrics = _metrics_at_threshold(y_true, y_proba, 0.5)
    try:
        best_metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        best_metrics["roc_auc"] = 0.5
    return float(best_t), best_metrics


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """
    Compute full metrics at a given threshold.
    Includes: optimal_threshold (same as input), roc_auc, f1, sensitivity, specificity, accuracy.
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    m = _metrics_at_threshold(y_true, y_proba, threshold)
    try:
        m["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        m["roc_auc"] = 0.5
    m["optimal_threshold"] = float(threshold)
    return m


def evaluate_across_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> List[Dict[str, float]]:
    """Evaluate metrics at each threshold. Returns list of dicts (one per threshold)."""
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    return [_metrics_at_threshold(y_true, y_proba, t) for t in thresholds]


def run_threshold_optimization(
    metrics_path: str,
    output_path: Optional[str] = None,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Load CV results (with fold_predictions), run threshold optimization per fold,
    and aggregate. Expects JSON with key "fold_predictions" (list of {fold, y_true, y_proba}).
    If "fold_predictions" is missing, returns a message that re-run of CV is needed to save predictions.

    Returns report with:
    - per_fold_youden: list of {fold, optimal_threshold, roc_auc, f1, sensitivity, specificity, accuracy}
    - per_fold_f1: same for F1-optimal threshold
    - summary_youden: mean ± std across folds for Youden-optimal metrics
    - summary_f1: mean ± std for F1-optimal metrics
    """
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with open(metrics_path) as f:
        data = json.load(f)

    fold_predictions = data.get("fold_predictions")
    if not fold_predictions:
        return {
            "error": "No fold_predictions in metrics file. Re-run CV (run_temporal_dl.py) to save predictions.",
            "hint": "Updated training pipeline saves y_true and y_proba per fold in temporal_dl_cv_metrics.json",
        }

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    per_fold_youden: List[Dict[str, Any]] = []
    per_fold_f1: List[Dict[str, Any]] = []

    for item in fold_predictions:
        fold = item["fold"]
        y_true = np.array(item["y_true"])
        y_proba = np.array(item["y_proba"])
        if len(y_true) == 0 or len(y_proba) == 0:
            continue

        t_youden, m_youden = find_optimal_threshold_youden(y_true, y_proba, thresholds)
        per_fold_youden.append({
            "fold": fold,
            "optimal_threshold": t_youden,
            "roc_auc": m_youden["roc_auc"],
            "f1": m_youden["f1"],
            "sensitivity": m_youden["sensitivity"],
            "specificity": m_youden["specificity"],
            "accuracy": m_youden["accuracy"],
        })

        t_f1, m_f1 = find_optimal_threshold_f1(y_true, y_proba, thresholds)
        per_fold_f1.append({
            "fold": fold,
            "optimal_threshold": t_f1,
            "roc_auc": m_f1["roc_auc"],
            "f1": m_f1["f1"],
            "sensitivity": m_f1["sensitivity"],
            "specificity": m_f1["specificity"],
            "accuracy": m_f1["accuracy"],
        })

    def _summary(per_fold: List[Dict]) -> Dict[str, float]:
        if not per_fold:
            return {}
        keys = ["roc_auc", "f1", "sensitivity", "specificity", "accuracy"]
        out = {}
        for k in keys:
            vals = [p[k] for p in per_fold]
            out[f"mean_{k}"] = float(np.mean(vals))
            out[f"std_{k}"] = float(np.std(vals))
        out["mean_optimal_threshold"] = float(np.mean([p["optimal_threshold"] for p in per_fold]))
        out["std_optimal_threshold"] = float(np.std([p["optimal_threshold"] for p in per_fold]))
        return out

    report = {
        "per_fold_youden": per_fold_youden,
        "per_fold_f1": per_fold_f1,
        "summary_youden": _summary(per_fold_youden),
        "summary_f1": _summary(per_fold_f1),
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    return report
