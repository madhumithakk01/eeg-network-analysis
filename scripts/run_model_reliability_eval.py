#!/usr/bin/env python3
"""
Unified model reliability evaluation across available model CV prediction files.

Supports:
- Temporal CNN (temporal_dl_cv_metrics.json with fold_predictions)
- Connectivity DL stride-8 (connectivity_dl_cv_metrics.json with fold_predictions)
- Connectivity DL stride-4 (connectivity_dl_cv_metrics.json with fold_predictions)

RF baseline is included from summary-only metrics when fold predictions are unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import (  # noqa: E402
    ANALYSIS_OUTPUT_PATH,
    CONNECTIVITY_DL_OUTPUT_PATH,
    CONNECTIVITY_DL_STRIDE4_OUTPUT_PATH,
    MODEL_RESULTS_PATH,
    TEMPORAL_DL_OUTPUT_PATH,
)


def _metrics_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_proba >= float(threshold)).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) else 0.0,
        "youden": float(sens + spec - 1.0),
    }


def _find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, objective: str) -> float:
    thresholds = np.linspace(0.0, 1.0, 101)
    best_t = 0.5
    best_v = -1e9
    for t in thresholds:
        m = _metrics_at_threshold(y_true, y_proba, float(t))
        v = m["youden"] if objective == "youden" else m["f1"]
        if v > best_v:
            best_v = float(v)
            best_t = float(t)
    return best_t


def _brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    return float(np.mean((y_proba - y_true) ** 2))


def _ece_score(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_proba >= lo) & (y_proba < hi if i < n_bins - 1 else y_proba <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_proba[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def _fit_calibrator(method: str, p_cal: np.ndarray, y_cal: np.ndarray):
    if method == "none":
        return None
    if method == "platt":
        lr = LogisticRegression(max_iter=2000)
        lr.fit(p_cal.reshape(-1, 1), y_cal)
        return lr
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal, y_cal)
        return iso
    raise ValueError(f"Unsupported calibration method: {method}")


def _apply_calibrator(calibrator, method: str, p: np.ndarray) -> np.ndarray:
    if method == "none" or calibrator is None:
        out = p
    elif method == "platt":
        out = calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        out = calibrator.predict(p)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return np.clip(np.asarray(out, dtype=np.float64), 0.0, 1.0)


def _evaluate_fold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    method: str,
    seed: int,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_proba = np.asarray(y_proba, dtype=np.float64).ravel()
    if len(y_true) < 10 or len(np.unique(y_true)) < 2:
        return {"error": "fold_too_small_or_single_class"}

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    idx_cal, idx_eval = next(sss.split(y_proba.reshape(-1, 1), y_true))
    y_cal, y_eval = y_true[idx_cal], y_true[idx_eval]
    p_cal_raw, p_eval_raw = y_proba[idx_cal], y_proba[idx_eval]

    calibrator = _fit_calibrator(method, p_cal_raw, y_cal)
    p_cal = _apply_calibrator(calibrator, method, p_cal_raw)
    p_eval = _apply_calibrator(calibrator, method, p_eval_raw)

    t_youden = _find_optimal_threshold(y_cal, p_cal, objective="youden")
    t_f1 = _find_optimal_threshold(y_cal, p_cal, objective="f1")

    out = {
        "thresholds": {
            "default_0p5": 0.5,
            "youden_on_cal": float(t_youden),
            "f1_on_cal": float(t_f1),
        },
        "eval_default_0p5": _metrics_at_threshold(y_eval, p_eval, 0.5),
        "eval_youden_on_cal": _metrics_at_threshold(y_eval, p_eval, t_youden),
        "eval_f1_on_cal": _metrics_at_threshold(y_eval, p_eval, t_f1),
        "eval_roc_auc": float(roc_auc_score(y_eval, p_eval)),
        "eval_brier": _brier_score(y_eval, p_eval),
        "eval_ece": _ece_score(y_eval, p_eval, n_bins=10),
    }
    return out


def _summarize_fold_runs(fold_runs: List[Dict[str, Any]]) -> Dict[str, float]:
    valid = [r for r in fold_runs if "error" not in r]
    if not valid:
        return {}

    def _mean_std(vals: List[float]) -> Tuple[float, float]:
        return float(np.mean(vals)), float(np.std(vals))

    keys_base = ["eval_roc_auc", "eval_brier", "eval_ece"]
    summary: Dict[str, float] = {"n_folds_used": float(len(valid))}
    for k in keys_base:
        m, s = _mean_std([float(v[k]) for v in valid])
        summary[f"mean_{k}"] = m
        summary[f"std_{k}"] = s

    for policy in ["eval_default_0p5", "eval_youden_on_cal", "eval_f1_on_cal"]:
        for metric in ["f1", "sensitivity", "specificity", "accuracy", "youden"]:
            vals = [float(v[policy][metric]) for v in valid]
            m, s = _mean_std(vals)
            summary[f"mean_{policy}_{metric}"] = m
            summary[f"std_{policy}_{metric}"] = s

    return summary


def _evaluate_model_with_predictions(
    model_name: str,
    metrics_path: str,
    seed: int,
) -> Dict[str, Any]:
    if not os.path.isfile(metrics_path):
        return {"model": model_name, "available": False, "error": f"missing_file: {metrics_path}"}
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fold_predictions = data.get("fold_predictions", [])
    if not fold_predictions:
        return {"model": model_name, "available": False, "error": "missing_fold_predictions"}

    methods = ["none", "platt", "isotonic"]
    by_method: Dict[str, Any] = {}
    for method in methods:
        fold_runs = []
        for item in fold_predictions:
            fold = int(item.get("fold", len(fold_runs)))
            y_true = np.array(item.get("y_true", []))
            y_proba = np.array(item.get("y_proba", []))
            run = _evaluate_fold(y_true, y_proba, method=method, seed=seed + fold)
            run["fold"] = fold
            fold_runs.append(run)
        by_method[method] = {
            "per_fold": fold_runs,
            "summary": _summarize_fold_runs(fold_runs),
        }

    return {
        "model": model_name,
        "available": True,
        "metrics_path": metrics_path,
        "calibration_methods": by_method,
    }


def _evaluate_rf_summary_only(model_name: str, metrics_path: str) -> Dict[str, Any]:
    if not os.path.isfile(metrics_path):
        return {"model": model_name, "available": False, "error": f"missing_file: {metrics_path}"}
    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "model": model_name,
        "available": True,
        "metrics_path": metrics_path,
        "summary_only": True,
        "summary": {
            "mean_roc_auc": float(data.get("mean_roc_auc", np.nan)),
            "mean_f1": float(data.get("mean_f1", np.nan)),
            "mean_sensitivity": float(data.get("mean_sensitivity", np.nan)),
            "mean_specificity": float(data.get("mean_specificity", np.nan)),
            "mean_accuracy": float(data.get("mean_accuracy", np.nan)),
        },
        "note": "No fold_predictions found for RF; calibration/Brier/ECE unavailable.",
    }


def _build_comparison_rows(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in results:
        if not r.get("available"):
            continue
        if r.get("summary_only"):
            s = r["summary"]
            rows.append(
                {
                    "model": r["model"],
                    "calibration": "n/a",
                    "threshold_policy": "n/a",
                    "mean_roc_auc": s.get("mean_roc_auc"),
                    "mean_f1": s.get("mean_f1"),
                    "mean_sensitivity": s.get("mean_sensitivity"),
                    "mean_specificity": s.get("mean_specificity"),
                    "mean_accuracy": s.get("mean_accuracy"),
                    "mean_brier": np.nan,
                    "mean_ece": np.nan,
                }
            )
            continue

        for method, mres in r["calibration_methods"].items():
            s = mres.get("summary", {})
            if not s:
                continue
            rows.append(
                {
                    "model": r["model"],
                    "calibration": method,
                    "threshold_policy": "default_0p5",
                    "mean_roc_auc": s.get("mean_eval_roc_auc"),
                    "mean_f1": s.get("mean_eval_default_0p5_f1"),
                    "mean_sensitivity": s.get("mean_eval_default_0p5_sensitivity"),
                    "mean_specificity": s.get("mean_eval_default_0p5_specificity"),
                    "mean_accuracy": s.get("mean_eval_default_0p5_accuracy"),
                    "mean_brier": s.get("mean_eval_brier"),
                    "mean_ece": s.get("mean_eval_ece"),
                }
            )
            rows.append(
                {
                    "model": r["model"],
                    "calibration": method,
                    "threshold_policy": "youden_on_cal",
                    "mean_roc_auc": s.get("mean_eval_roc_auc"),
                    "mean_f1": s.get("mean_eval_youden_on_cal_f1"),
                    "mean_sensitivity": s.get("mean_eval_youden_on_cal_sensitivity"),
                    "mean_specificity": s.get("mean_eval_youden_on_cal_specificity"),
                    "mean_accuracy": s.get("mean_eval_youden_on_cal_accuracy"),
                    "mean_brier": s.get("mean_eval_brier"),
                    "mean_ece": s.get("mean_eval_ece"),
                }
            )
            rows.append(
                {
                    "model": r["model"],
                    "calibration": method,
                    "threshold_policy": "f1_on_cal",
                    "mean_roc_auc": s.get("mean_eval_roc_auc"),
                    "mean_f1": s.get("mean_eval_f1_on_cal_f1"),
                    "mean_sensitivity": s.get("mean_eval_f1_on_cal_sensitivity"),
                    "mean_specificity": s.get("mean_eval_f1_on_cal_specificity"),
                    "mean_accuracy": s.get("mean_eval_f1_on_cal_accuracy"),
                    "mean_brier": s.get("mean_eval_brier"),
                    "mean_ece": s.get("mean_eval_ece"),
                }
            )
    return pd.DataFrame(rows)


def _plot_metric(df: pd.DataFrame, metric: str, out_path: str, title: str) -> None:
    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return
    plot_df["label"] = (
        plot_df["model"].astype(str)
        + " | "
        + plot_df["calibration"].astype(str)
        + " | "
        + plot_df["threshold_policy"].astype(str)
    )
    plot_df = plot_df.sort_values(metric, ascending=(metric in {"mean_brier", "mean_ece"}))

    plt.figure(figsize=(12, max(4, 0.35 * len(plot_df))))
    plt.barh(plot_df["label"], plot_df[metric])
    plt.xlabel(metric)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified reliability eval: calibration + threshold policies.")
    default_out = os.path.join(ANALYSIS_OUTPUT_PATH, "model_reliability")
    parser.add_argument("--output-dir", type=str, default=default_out)
    parser.add_argument(
        "--rf-metrics",
        type=str,
        default=os.path.join(MODEL_RESULTS_PATH, "cv_metrics.json"),
    )
    parser.add_argument(
        "--temporal-dl-metrics",
        type=str,
        default=os.path.join(TEMPORAL_DL_OUTPUT_PATH, "temporal_dl_cv_metrics.json"),
    )
    parser.add_argument(
        "--connectivity-dl-metrics",
        type=str,
        default=os.path.join(CONNECTIVITY_DL_OUTPUT_PATH, "connectivity_dl_cv_metrics.json"),
    )
    parser.add_argument(
        "--connectivity-dl-stride4-metrics",
        type=str,
        default=os.path.join(CONNECTIVITY_DL_STRIDE4_OUTPUT_PATH, "connectivity_dl_cv_metrics.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []
    results.append(_evaluate_rf_summary_only("rf_baseline", args.rf_metrics))
    results.append(_evaluate_model_with_predictions("temporal_cnn", args.temporal_dl_metrics, seed=args.seed))
    results.append(_evaluate_model_with_predictions("connectivity_dl", args.connectivity_dl_metrics, seed=args.seed))
    results.append(
        _evaluate_model_with_predictions(
            "connectivity_dl_stride4",
            args.connectivity_dl_stride4_metrics,
            seed=args.seed,
        )
    )

    full_json = os.path.join(args.output_dir, "model_reliability_full_report.json")
    with open(full_json, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    comparison_df = _build_comparison_rows(results)
    comparison_csv = os.path.join(args.output_dir, "model_reliability_comparison.csv")
    comparison_df.to_csv(comparison_csv, index=False)

    plot_auc = os.path.join(args.output_dir, "model_reliability_auc.png")
    plot_brier = os.path.join(args.output_dir, "model_reliability_brier.png")
    plot_ece = os.path.join(args.output_dir, "model_reliability_ece.png")
    _plot_metric(comparison_df, "mean_roc_auc", plot_auc, "Model Reliability: ROC-AUC")
    _plot_metric(comparison_df, "mean_brier", plot_brier, "Model Reliability: Brier (lower is better)")
    _plot_metric(comparison_df, "mean_ece", plot_ece, "Model Reliability: ECE (lower is better)")

    print("Unified model reliability evaluation")
    print("=" * 60)
    for r in results:
        if r.get("available"):
            if r.get("summary_only"):
                print(f"  {r['model']}: summary-only (no fold predictions)")
            else:
                print(f"  {r['model']}: calibrated reliability computed")
        else:
            print(f"  {r['model']}: unavailable ({r.get('error')})")
    print("=" * 60)
    print(f"Saved: {full_json}")
    print(f"Saved: {comparison_csv}")
    if os.path.isfile(plot_auc):
        print(f"Saved: {plot_auc}")
    if os.path.isfile(plot_brier):
        print(f"Saved: {plot_brier}")
    if os.path.isfile(plot_ece):
        print(f"Saved: {plot_ece}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

