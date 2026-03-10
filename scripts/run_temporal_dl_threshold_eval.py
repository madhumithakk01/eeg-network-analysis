#!/usr/bin/env python3
"""
Threshold optimization for Temporal CNN CV results.

Loads temporal_dl_cv_metrics.json (must contain fold_predictions: y_true, y_proba per fold),
evaluates metrics across thresholds 0.0--1.0, finds optimal threshold by Youden Index and by F1,
and saves temporal_dl_threshold_metrics.json with per-fold and summary metrics at optimized thresholds.

If the metrics file was produced by an older run without fold_predictions, re-run:
  python scripts/run_temporal_dl.py
then run this script again.
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import TEMPORAL_DL_OUTPUT_PATH
from src.temporal_models.threshold_evaluation import run_threshold_optimization


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run threshold optimization on Temporal CNN CV results.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Path to temporal_dl_cv_metrics.json (default: TEMPORAL_DL_OUTPUT_PATH/temporal_dl_cv_metrics.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save temporal_dl_threshold_metrics.json (default: same dir as --metrics).",
    )
    args = parser.parse_args()

    metrics_path = args.metrics or os.path.join(TEMPORAL_DL_OUTPUT_PATH, "temporal_dl_cv_metrics.json")
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(os.path.dirname(metrics_path), "temporal_dl_threshold_metrics.json")

    print("Threshold optimization (Youden Index & best F1)")
    print(f"  Metrics:  {metrics_path}")
    print(f"  Output:  {output_path}")

    report = run_threshold_optimization(metrics_path, output_path=output_path)

    if "error" in report:
        print(f"\nError: {report['error']}")
        if report.get("hint"):
            print(f"  {report['hint']}")
        return 1

    sy = report["summary_youden"]
    sf = report["summary_f1"]
    print("\n--- Youden Index (max Sensitivity + Specificity - 1) ---")
    print(f"  Mean optimal threshold: {sy['mean_optimal_threshold']:.3f} ± {sy['std_optimal_threshold']:.3f}")
    print(f"  ROC-AUC:    {sy['mean_roc_auc']:.4f} ± {sy['std_roc_auc']:.4f}")
    print(f"  F1:         {sy['mean_f1']:.4f} ± {sy['std_f1']:.4f}")
    print(f"  Sensitivity: {sy['mean_sensitivity']:.4f} ± {sy['std_sensitivity']:.4f}")
    print(f"  Specificity: {sy['mean_specificity']:.4f} ± {sy['std_specificity']:.4f}")
    print(f"  Accuracy:   {sy['mean_accuracy']:.4f} ± {sy['std_accuracy']:.4f}")

    print("\n--- Best F1 threshold ---")
    print(f"  Mean optimal threshold: {sf['mean_optimal_threshold']:.3f} ± {sf['std_optimal_threshold']:.3f}")
    print(f"  ROC-AUC:    {sf['mean_roc_auc']:.4f} ± {sf['std_roc_auc']:.4f}")
    print(f"  F1:         {sf['mean_f1']:.4f} ± {sf['std_f1']:.4f}")
    print(f"  Sensitivity: {sf['mean_sensitivity']:.4f} ± {sf['std_sensitivity']:.4f}")
    print(f"  Specificity: {sf['mean_specificity']:.4f} ± {sf['std_specificity']:.4f}")
    print(f"  Accuracy:   {sf['mean_accuracy']:.4f} ± {sf['std_accuracy']:.4f}")

    print(f"\nSaved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
