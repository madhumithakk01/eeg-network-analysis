#!/usr/bin/env python3
"""
Unified model comparison: aggregate metrics from RF baseline, Temporal CNN, Connectivity DL, and Connectivity DL (stride=4).

Loads cv_metrics.json and *_cv_metrics.json from each pipeline, builds a summary table (ROC-AUC, F1, Sensitivity, Specificity, Accuracy),
saves analysis/model_comparison/model_performance_comparison.csv and model_auc_comparison.png.

Run after all pipelines have produced their metrics files.
"""

from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import (
    ANALYSIS_OUTPUT_PATH,
    CONNECTIVITY_DL_OUTPUT_PATH,
    CONNECTIVITY_DL_STRIDE4_OUTPUT_PATH,
    MODEL_RESULTS_PATH,
    TEMPORAL_DL_OUTPUT_PATH,
)
from src.model_comparison.comparison import run_model_comparison

MODEL_COMPARISON_OUTPUT_DIR = os.path.join(ANALYSIS_OUTPUT_PATH, "model_comparison")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate metrics from all pipelines and produce comparison table and ROC-AUC plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {MODEL_COMPARISON_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--rf-metrics",
        type=str,
        default=None,
        help="Path to RF cv_metrics.json (default: MODEL_RESULTS_PATH/cv_metrics.json).",
    )
    parser.add_argument(
        "--temporal-dl-metrics",
        type=str,
        default=None,
        help="Path to temporal_dl_cv_metrics.json (default: TEMPORAL_DL_OUTPUT_PATH/temporal_dl_cv_metrics.json).",
    )
    parser.add_argument(
        "--connectivity-dl-metrics",
        type=str,
        default=None,
        help="Path to connectivity_dl_cv_metrics.json (default: CONNECTIVITY_DL_OUTPUT_PATH/connectivity_dl_cv_metrics.json).",
    )
    parser.add_argument(
        "--connectivity-dl-stride4-metrics",
        type=str,
        default=None,
        help="Path to connectivity_dl_cv_metrics.json for stride=4 (default: CONNECTIVITY_DL_STRIDE4_OUTPUT_PATH/connectivity_dl_cv_metrics.json).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or MODEL_COMPARISON_OUTPUT_DIR
    rf_path = args.rf_metrics or os.path.join(MODEL_RESULTS_PATH, "cv_metrics.json")
    temporal_path = args.temporal_dl_metrics or os.path.join(TEMPORAL_DL_OUTPUT_PATH, "temporal_dl_cv_metrics.json")
    conn_path = args.connectivity_dl_metrics or os.path.join(CONNECTIVITY_DL_OUTPUT_PATH, "connectivity_dl_cv_metrics.json")
    conn_s4_path = args.connectivity_dl_stride4_metrics or os.path.join(
        CONNECTIVITY_DL_STRIDE4_OUTPUT_PATH, "connectivity_dl_cv_metrics.json"
    )

    print("Unified model comparison")
    print("=" * 60)
    print(f"  RF metrics:              {rf_path}")
    print(f"  Temporal DL metrics:     {temporal_path}")
    print(f"  Connectivity DL metrics:  {conn_path}")
    print(f"  Connectivity DL stride4: {conn_s4_path}")
    print(f"  Output dir:              {output_dir}")
    print("=" * 60)

    result = run_model_comparison(
        output_dir=output_dir,
        rf_metrics_path=rf_path,
        temporal_dl_metrics_path=temporal_path,
        connectivity_dl_metrics_path=conn_path,
        connectivity_dl_stride4_metrics_path=conn_s4_path,
    )

    if result.get("error"):
        print(f"\nError: {result['error']}")
        return 1

    print("\nMetrics loaded:")
    for name, loaded in result["metrics_loaded"].items():
        print(f"  {name}: {'yes' if loaded else 'no'}")

    print(f"\nTable saved: {result['csv_path']}")
    print(f"Plot saved:  {result['plot_path']}")
    print("\nComparison table:")
    for row in result["table"]:
        print("  ", row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
