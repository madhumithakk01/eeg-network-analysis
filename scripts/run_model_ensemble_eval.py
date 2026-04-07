#!/usr/bin/env python3
"""
Leakage-safe fold-wise ensemble evaluation across deep models.

Reads fold predictions from:
- temporal_dl_cv_metrics.json
- connectivity_dl_cv_metrics.json (stride 8)
- connectivity_dl_cv_metrics.json (stride 4, optional)

For each fold:
1) Split fold validation predictions into calibration/eval halves (stratified).
2) Calibrate each model on calibration half (none/platt/isotonic).
3) Optimize nonnegative blend weights (sum=1) on calibration half.
4) Evaluate ensemble on eval half with fixed threshold policies.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import ANALYSIS_OUTPUT_PATH, CONNECTIVITY_DL_OUTPUT_PATH, CONNECTIVITY_DL_STRIDE4_OUTPUT_PATH, TEMPORAL_DL_OUTPUT_PATH


def _metrics_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(np.int64)
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


def _fit_calibrator(method: str, p_cal: np.ndarray, y_cal: np.ndarray):
    if method == "none":
        return None
    if method == "platt":
        m = LogisticRegression(max_iter=2000)
        m.fit(p_cal.reshape(-1, 1), y_cal)
        return m
    if method == "isotonic":
        m = IsotonicRegression(out_of_bounds="clip")
        m.fit(p_cal, y_cal)
        return m
    raise ValueError(f"Unsupported calibration method: {method}")


def _apply_calibrator(calibrator, method: str, p: np.ndarray) -> np.ndarray:
    if method == "none" or calibrator is None:
        out = p
    elif method == "platt":
        out = calibrator.predict_proba(p.reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        out = calibrator.predict(p)
    else:
        raise ValueError(method)
    return np.clip(np.asarray(out, dtype=np.float64), 0.0, 1.0)


def _load_fold_preds(path: str) -> Dict[int, Dict[str, np.ndarray]]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[int, Dict[str, np.ndarray]] = {}
    for item in data.get("fold_predictions", []):
        fold = int(item["fold"])
        out[fold] = {
            "y_true": np.asarray(item["y_true"], dtype=np.int64),
            "y_proba": np.asarray(item["y_proba"], dtype=np.float64),
        }
    return out


def _candidate_weights(n_models: int, step: float) -> List[np.ndarray]:
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    out = []
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
        # Fallback uniform for unsupported size
        out.append(np.ones(n_models, dtype=np.float64) / n_models)
    return out


def _select_weights(
    y_cal: np.ndarray,
    p_cal_models: np.ndarray,
    objective: str,
    step: float,
) -> np.ndarray:
    cand = _candidate_weights(p_cal_models.shape[1], step=step)
    best_w = cand[0]
    best_v = -1e9
    for w in cand:
        p = np.clip(p_cal_models @ w, 0.0, 1.0)
        if objective == "roc_auc":
            if len(np.unique(y_cal)) < 2:
                v = 0.5
            else:
                v = float(roc_auc_score(y_cal, p))
        elif objective == "f1":
            t = _find_optimal_threshold(y_cal, p, objective="f1")
            v = _metrics_at_threshold(y_cal, p, t)["f1"]
        else:
            raise ValueError(f"Unsupported objective: {objective}")
        if v > best_v:
            best_v = float(v)
            best_w = w
    return best_w


def _plot_weights(df: pd.DataFrame, out_path: str) -> None:
    cols = [c for c in df.columns if c.startswith("w_")]
    if not cols:
        return
    m = df[cols].mean(axis=0).sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    plt.bar(m.index, m.values)
    plt.ylabel("Mean selected weight")
    plt.title("Ensemble weights (mean across folds)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate calibrated weighted ensemble from fold predictions.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(ANALYSIS_OUTPUT_PATH, "model_ensemble"),
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
    parser.add_argument("--calibration", type=str, default="platt", choices=["none", "platt", "isotonic"])
    parser.add_argument("--objective", type=str, default="roc_auc", choices=["roc_auc", "f1"])
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_paths = {
        "temporal_cnn": args.temporal_dl_metrics,
        "connectivity_dl": args.connectivity_dl_metrics,
        "connectivity_dl_stride4": args.connectivity_dl_stride4_metrics,
    }
    model_fold_data = {k: _load_fold_preds(v) for k, v in model_paths.items()}
    available_models = [k for k, d in model_fold_data.items() if d]
    if len(available_models) < 2:
        print("Need at least two models with fold_predictions.")
        return 1

    common_folds = sorted(set.intersection(*[set(model_fold_data[m].keys()) for m in available_models]))
    if not common_folds:
        print("No common folds found across available models.")
        return 1

    per_fold_rows: List[Dict[str, Any]] = []
    for fold in common_folds:
        # Align and verify y_true consistency
        y_refs = [model_fold_data[m][fold]["y_true"] for m in available_models]
        lengths = [len(y) for y in y_refs]
        if len(set(lengths)) != 1:
            print(f"Skip fold {fold}: unequal prediction lengths {dict(zip(available_models, lengths))}")
            continue
        y_true = y_refs[0]
        if any(not np.array_equal(y_true, yr) for yr in y_refs[1:]):
            print(f"Skip fold {fold}: y_true mismatch across models.")
            continue
        if len(np.unique(y_true)) < 2:
            print(f"Skip fold {fold}: single-class labels.")
            continue

        p_raw_models = np.column_stack([model_fold_data[m][fold]["y_proba"] for m in available_models])

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed + int(fold))
        idx_cal, idx_eval = next(sss.split(p_raw_models, y_true))
        y_cal, y_eval = y_true[idx_cal], y_true[idx_eval]
        p_cal_raw = p_raw_models[idx_cal, :]
        p_eval_raw = p_raw_models[idx_eval, :]

        # calibrate per model using calibration split only
        p_cal_list = []
        p_eval_list = []
        for mi in range(p_cal_raw.shape[1]):
            cal = _fit_calibrator(args.calibration, p_cal_raw[:, mi], y_cal)
            p_cal_list.append(_apply_calibrator(cal, args.calibration, p_cal_raw[:, mi]))
            p_eval_list.append(_apply_calibrator(cal, args.calibration, p_eval_raw[:, mi]))
        p_cal = np.column_stack(p_cal_list)
        p_eval = np.column_stack(p_eval_list)

        w = _select_weights(y_cal, p_cal, objective=args.objective, step=args.weight_step)
        p_cal_ens = np.clip(p_cal @ w, 0.0, 1.0)
        p_eval_ens = np.clip(p_eval @ w, 0.0, 1.0)

        t_youden = _find_optimal_threshold(y_cal, p_cal_ens, objective="youden")
        t_f1 = _find_optimal_threshold(y_cal, p_cal_ens, objective="f1")

        row: Dict[str, Any] = {
            "fold": int(fold),
            "roc_auc": float(roc_auc_score(y_eval, p_eval_ens)),
            "brier": _brier(y_eval, p_eval_ens),
            "ece": _ece(y_eval, p_eval_ens, n_bins=10),
            "threshold_youden": float(t_youden),
            "threshold_f1": float(t_f1),
        }
        m05 = _metrics_at_threshold(y_eval, p_eval_ens, 0.5)
        my = _metrics_at_threshold(y_eval, p_eval_ens, t_youden)
        mf = _metrics_at_threshold(y_eval, p_eval_ens, t_f1)
        for prefix, m in [("default05", m05), ("youden", my), ("f1", mf)]:
            for k, v in m.items():
                row[f"{prefix}_{k}"] = float(v)
        for i, mname in enumerate(available_models):
            row[f"w_{mname}"] = float(w[i])
        per_fold_rows.append(row)

    if not per_fold_rows:
        print("No valid folds could be evaluated.")
        return 1

    df = pd.DataFrame(per_fold_rows)
    fold_csv = os.path.join(args.output_dir, "ensemble_per_fold_metrics.csv")
    df.to_csv(fold_csv, index=False)

    summary = {
        "n_folds_used": int(len(df)),
        "models_used": available_models,
        "calibration": args.calibration,
        "weight_objective": args.objective,
        "weight_step": args.weight_step,
        "mean_roc_auc": float(df["roc_auc"].mean()),
        "std_roc_auc": float(df["roc_auc"].std()),
        "mean_brier": float(df["brier"].mean()),
        "std_brier": float(df["brier"].std()),
        "mean_ece": float(df["ece"].mean()),
        "std_ece": float(df["ece"].std()),
        "mean_default05_f1": float(df["default05_f1"].mean()),
        "mean_default05_sensitivity": float(df["default05_sensitivity"].mean()),
        "mean_default05_specificity": float(df["default05_specificity"].mean()),
        "mean_default05_accuracy": float(df["default05_accuracy"].mean()),
        "mean_youden_f1": float(df["youden_f1"].mean()),
        "mean_youden_sensitivity": float(df["youden_sensitivity"].mean()),
        "mean_youden_specificity": float(df["youden_specificity"].mean()),
        "mean_youden_accuracy": float(df["youden_accuracy"].mean()),
        "mean_f1policy_f1": float(df["f1_f1"].mean()),
        "mean_f1policy_sensitivity": float(df["f1_sensitivity"].mean()),
        "mean_f1policy_specificity": float(df["f1_specificity"].mean()),
        "mean_f1policy_accuracy": float(df["f1_accuracy"].mean()),
    }
    summary_json = os.path.join(args.output_dir, "ensemble_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_weights(df, os.path.join(args.output_dir, "ensemble_weight_means.png"))
    plt.figure(figsize=(7, 4))
    plt.hist(df["roc_auc"].values, bins=min(10, len(df)), edgecolor="black")
    plt.title("Ensemble ROC-AUC across folds")
    plt.xlabel("ROC-AUC")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "ensemble_auc_hist.png"), dpi=150)
    plt.close()

    print("Ensemble evaluation complete")
    print("=" * 60)
    print(f"Models used: {available_models}")
    print(f"Calibration: {args.calibration} | Objective: {args.objective}")
    print(f"Folds used: {summary['n_folds_used']}")
    print(f"Mean ROC-AUC: {summary['mean_roc_auc']:.4f} ± {summary['std_roc_auc']:.4f}")
    print(f"Mean Brier:   {summary['mean_brier']:.4f}")
    print(f"Mean ECE:     {summary['mean_ece']:.4f}")
    print(f"Saved: {fold_csv}")
    print(f"Saved: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

