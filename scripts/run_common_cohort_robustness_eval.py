#!/usr/bin/env python3
"""
Common-cohort robustness evaluation for model comparison.

Compares Temporal CNN, Connectivity DL, Connectivity DL stride-4, and an ensemble
on the aligned fold prediction cohort. Generates fold-wise and summary diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def _brier(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def _ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not np.any(mask):
            continue
        out += (np.sum(mask) / n) * abs(float(np.mean(y_true[mask])) - float(np.mean(p[mask])))
    return float(out)


def _load_fold_predictions(path: str) -> Dict[int, Dict[str, np.ndarray]]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for item in data.get("fold_predictions", []):
        fold = int(item.get("fold", -1))
        if fold < 0:
            continue
        out[fold] = {
            "y_true": np.asarray(item.get("y_true", []), dtype=np.int64),
            "y_proba": np.asarray(item.get("y_proba", []), dtype=np.float64),
        }
    return out


def _candidate_weights(n_models: int, step: float) -> List[np.ndarray]:
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    out: List[np.ndarray] = []
    if n_models == 2:
        for a in vals:
            out.append(np.array([a, 1.0 - a], dtype=np.float64))
    elif n_models == 3:
        for a in vals:
            for b in vals:
                c = 1.0 - a - b
                if c < -1e-9:
                    continue
                if c < 0:
                    c = 0.0
                out.append(np.array([a, b, c], dtype=np.float64))
    else:
        out.append(np.ones(n_models, dtype=np.float64) / n_models)
    return out


def _select_ensemble_weights(y_cal: np.ndarray, p_cal_models: np.ndarray, step: float) -> np.ndarray:
    best_w = None
    best_auc = -1e9
    for w in _candidate_weights(p_cal_models.shape[1], step=step):
        p = np.clip(p_cal_models @ w, 0.0, 1.0)
        auc = 0.5 if len(np.unique(y_cal)) < 2 else float(roc_auc_score(y_cal, p))
        if auc > best_auc:
            best_auc = auc
            best_w = w
    assert best_w is not None
    return best_w


def _summary(df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {"n_folds_used": float(len(df))}
    for c in metric_cols:
        out[f"mean_{c}"] = float(df[c].mean())
        out[f"std_{c}"] = float(df[c].std())
    return out


def _plot_gain_hist(df: pd.DataFrame, out_path: str) -> None:
    if df.empty:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(df["ensemble_auc_gain_vs_best_single"].values, bins=min(10, len(df)), edgecolor="black")
    plt.title("Ensemble ROC-AUC gain vs best single model")
    plt.xlabel("AUC gain")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Common-cohort robustness evaluation for all available deep models.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(ANALYSIS_OUTPUT_PATH, "common_cohort_robustness"),
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
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_paths = {
        "temporal_cnn": args.temporal_dl_metrics,
        "connectivity_dl": args.connectivity_dl_metrics,
        "connectivity_dl_stride4": args.connectivity_dl_stride4_metrics,
    }
    fold_data = {m: _load_fold_predictions(p) for m, p in model_paths.items()}
    available_models = [m for m, d in fold_data.items() if d]
    if len(available_models) < 2:
        print("Need at least two models with fold_predictions for common-cohort evaluation.")
        return 1

    common_folds = sorted(set.intersection(*[set(fold_data[m].keys()) for m in available_models]))
    if not common_folds:
        print("No common folds across selected models.")
        return 1

    fold_rows: List[Dict[str, Any]] = []
    for fold in common_folds:
        ys = [fold_data[m][fold]["y_true"] for m in available_models]
        lens = [len(y) for y in ys]
        if len(set(lens)) != 1:
            continue
        y_true = ys[0]
        if any(not np.array_equal(y_true, y2) for y2 in ys[1:]):
            continue
        if len(np.unique(y_true)) < 2:
            continue

        p_models = np.column_stack([fold_data[m][fold]["y_proba"] for m in available_models])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed + int(fold))
        idx_cal, idx_eval = next(sss.split(p_models, y_true))
        y_cal, y_eval = y_true[idx_cal], y_true[idx_eval]
        p_cal, p_eval = p_models[idx_cal, :], p_models[idx_eval, :]

        weights = _select_ensemble_weights(y_cal, p_cal, step=args.weight_step)
        p_ens_eval = np.clip(p_eval @ weights, 0.0, 1.0)
        p_ens_cal = np.clip(p_cal @ weights, 0.0, 1.0)

        row: Dict[str, Any] = {"fold": int(fold), "n_eval": int(len(y_eval))}
        # single model eval on held-out half
        single_aucs = []
        for mi, model_name in enumerate(available_models):
            p = p_eval[:, mi]
            auc = float(roc_auc_score(y_eval, p))
            single_aucs.append(auc)
            row[f"{model_name}_auc"] = auc
            row[f"{model_name}_brier"] = _brier(y_eval, p)
            row[f"{model_name}_ece"] = _ece(y_eval, p, n_bins=10)

            t = _find_optimal_threshold(y_cal, p_cal[:, mi], objective="youden")
            m = _metrics_at_threshold(y_eval, p, t)
            row[f"{model_name}_youden_f1"] = m["f1"]
            row[f"{model_name}_youden_sensitivity"] = m["sensitivity"]
            row[f"{model_name}_youden_specificity"] = m["specificity"]
            row[f"{model_name}_youden_accuracy"] = m["accuracy"]

        # ensemble eval
        row["ensemble_auc"] = float(roc_auc_score(y_eval, p_ens_eval))
        row["ensemble_brier"] = _brier(y_eval, p_ens_eval)
        row["ensemble_ece"] = _ece(y_eval, p_ens_eval, n_bins=10)
        t_ens = _find_optimal_threshold(y_cal, p_ens_cal, objective="youden")
        m_ens = _metrics_at_threshold(y_eval, p_ens_eval, t_ens)
        row["ensemble_youden_f1"] = m_ens["f1"]
        row["ensemble_youden_sensitivity"] = m_ens["sensitivity"]
        row["ensemble_youden_specificity"] = m_ens["specificity"]
        row["ensemble_youden_accuracy"] = m_ens["accuracy"]
        row["ensemble_auc_gain_vs_best_single"] = float(row["ensemble_auc"] - max(single_aucs))

        for mi, model_name in enumerate(available_models):
            row[f"w_{model_name}"] = float(weights[mi])
        fold_rows.append(row)

    if not fold_rows:
        print("No fold passed alignment checks (same length + identical y_true).")
        return 1

    fold_df = pd.DataFrame(fold_rows)
    fold_csv = os.path.join(args.output_dir, "common_cohort_per_fold_metrics.csv")
    fold_df.to_csv(fold_csv, index=False)

    metric_cols = [c for c in fold_df.columns if c.endswith("_auc") or c.endswith("_brier") or c.endswith("_ece")]
    summary = _summary(fold_df, metric_cols)
    summary["models_used"] = available_models
    summary["n_folds_requested"] = int(len(common_folds))
    summary["n_folds_used"] = int(len(fold_df))
    summary["mean_ensemble_auc_gain_vs_best_single"] = float(fold_df["ensemble_auc_gain_vs_best_single"].mean())
    summary["std_ensemble_auc_gain_vs_best_single"] = float(fold_df["ensemble_auc_gain_vs_best_single"].std())
    summary["notes"] = [
        "Hospital-level subgroup analysis is unavailable unless fold predictions include patient IDs.",
        "This report guarantees same-cohort comparison only for folds passing strict y_true alignment checks.",
    ]

    summary_json = os.path.join(args.output_dir, "common_cohort_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_gain_hist(fold_df, os.path.join(args.output_dir, "ensemble_auc_gain_hist.png"))

    # Compact model comparison table
    comp_rows: List[Dict[str, Any]] = []
    for model_name in available_models + ["ensemble"]:
        comp_rows.append(
            {
                "model": model_name,
                "mean_auc": float(fold_df[f"{model_name}_auc"].mean()),
                "mean_brier": float(fold_df[f"{model_name}_brier"].mean()),
                "mean_ece": float(fold_df[f"{model_name}_ece"].mean()),
                "mean_youden_f1": float(fold_df[f"{model_name}_youden_f1"].mean()),
                "mean_youden_sensitivity": float(fold_df[f"{model_name}_youden_sensitivity"].mean()),
                "mean_youden_specificity": float(fold_df[f"{model_name}_youden_specificity"].mean()),
                "mean_youden_accuracy": float(fold_df[f"{model_name}_youden_accuracy"].mean()),
            }
        )
    comp_df = pd.DataFrame(comp_rows).sort_values("mean_auc", ascending=False)
    comp_csv = os.path.join(args.output_dir, "common_cohort_model_comparison.csv")
    comp_df.to_csv(comp_csv, index=False)

    print("Common-cohort robustness evaluation complete")
    print("=" * 60)
    print(f"Models used: {available_models}")
    print(f"Requested folds: {len(common_folds)} | Used folds: {len(fold_df)}")
    print(f"Mean ensemble AUC gain vs best single: {summary['mean_ensemble_auc_gain_vs_best_single']:.4f}")
    print(f"Saved: {fold_csv}")
    print(f"Saved: {comp_csv}")
    print(f"Saved: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

