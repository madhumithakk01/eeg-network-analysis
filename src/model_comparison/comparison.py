"""
Unified model comparison: load metrics from all pipelines and produce summary table and ROC-AUC visualization.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


# Default paths (from config); can be overridden by caller or script.
DEFAULT_PATHS = {
    "rf_baseline": None,  # MODEL_RESULTS_PATH/cv_metrics.json
    "temporal_cnn": None,  # TEMPORAL_DL_OUTPUT_PATH/temporal_dl_cv_metrics.json
    "connectivity_dl": None,  # CONNECTIVITY_DL_OUTPUT_PATH/connectivity_dl_cv_metrics.json
    "connectivity_dl_stride4": None,  # CONNECTIVITY_DL_STRIDE4_OUTPUT_PATH/connectivity_dl_cv_metrics.json
}

METRIC_KEYS = ["roc_auc", "f1", "sensitivity", "specificity", "accuracy"]
TABLE_COLUMNS = ["Model", "ROC-AUC", "F1", "Sensitivity", "Specificity", "Accuracy"]
KEY_TO_COLUMN = {
    "roc_auc": "ROC-AUC",
    "f1": "F1",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "accuracy": "Accuracy",
}


def _load_rf_metrics(path: str) -> Optional[Dict[str, float]]:
    """Load RF baseline cv_metrics.json (flat keys: mean_roc_auc, mean_f1, ...)."""
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return {
        "roc_auc": data.get("mean_roc_auc", np.nan),
        "f1": data.get("mean_f1", np.nan),
        "sensitivity": data.get("mean_sensitivity", np.nan),
        "specificity": data.get("mean_specificity", np.nan),
        "accuracy": data.get("mean_accuracy", np.nan),
    }


def _load_dl_metrics(path: str, summary_key: str = "summary") -> Optional[Dict[str, float]]:
    """Load DL metrics JSON (has 'summary' with mean_roc_auc, mean_f1, ...)."""
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    summary = data.get(summary_key)
    if not summary:
        return None
    return {
        "roc_auc": summary.get("mean_roc_auc", np.nan),
        "f1": summary.get("mean_f1", np.nan),
        "sensitivity": summary.get("mean_sensitivity", np.nan),
        "specificity": summary.get("mean_specificity", np.nan),
        "accuracy": summary.get("mean_accuracy", np.nan),
    }


def run_model_comparison(
    output_dir: str,
    rf_metrics_path: Optional[str] = None,
    temporal_dl_metrics_path: Optional[str] = None,
    connectivity_dl_metrics_path: Optional[str] = None,
    connectivity_dl_stride4_metrics_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load metrics from all four pipelines, build comparison table, save CSV and ROC-AUC plot.

    Parameters
    ----------
    output_dir : str
        Directory for outputs: model_performance_comparison.csv, model_auc_comparison.png.
    rf_metrics_path : str, optional
        Path to model_results/cv_metrics.json.
    temporal_dl_metrics_path : str, optional
        Path to temporal_dl_cv_metrics.json.
    connectivity_dl_metrics_path : str, optional
        Path to connectivity_dl_cv_metrics.json (stride 8).
    connectivity_dl_stride4_metrics_path : str, optional
        Path to connectivity_dl_cv_metrics.json (stride 4).

    Returns
    -------
    dict
        results with "table" (list of dicts), "csv_path", "plot_path", "metrics_loaded" (dict of model -> bool).
    """
    try:
        import pandas as pd
    except ImportError:
        pd = None

    rf = _load_rf_metrics(rf_metrics_path) if rf_metrics_path else None
    temporal = _load_dl_metrics(temporal_dl_metrics_path) if temporal_dl_metrics_path else None
    conn = _load_dl_metrics(connectivity_dl_metrics_path) if connectivity_dl_metrics_path else None
    conn_s4 = _load_dl_metrics(connectivity_dl_stride4_metrics_path) if connectivity_dl_stride4_metrics_path else None

    models: List[Dict[str, Any]] = []
    if rf is not None:
        models.append({"name": "Random Forest baseline", "metrics": rf})
    if temporal is not None:
        models.append({"name": "Temporal CNN", "metrics": temporal})
    if conn is not None:
        models.append({"name": "Connectivity DL", "metrics": conn})
    if conn_s4 is not None:
        models.append({"name": "Connectivity DL (stride=4)", "metrics": conn_s4})

    if not models:
        return {
            "table": [],
            "csv_path": None,
            "plot_path": None,
            "metrics_loaded": {},
            "error": "No metrics files found. Provide paths to cv_metrics.json and/or *_cv_metrics.json.",
        }

    rows = []
    for m in models:
        row = {"Model": m["name"]}
        for k in METRIC_KEYS:
            v = m["metrics"].get(k, np.nan)
            row[KEY_TO_COLUMN[k]] = round(float(v), 4) if np.isfinite(v) else ""
        rows.append(row)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "model_performance_comparison.csv")
    plot_path = os.path.join(output_dir, "model_auc_comparison.png")

    if pd is not None:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
    else:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TABLE_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    # ROC-AUC bar chart
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    if plt is not None and models:
        names = [m["name"] for m in models]
        aucs = [m["metrics"].get("roc_auc", 0.0) for m in models]
        aucs = [float(a) if np.isfinite(a) else 0.0 for a in aucs]
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(names))
        bars = ax.bar(x, aucs, color=["#2ecc71", "#3498db", "#9b59b6", "#e67e22"][: len(names)], edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Model performance comparison (ROC-AUC)")
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        for bar, val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

    metrics_loaded = {
        "Random Forest baseline": rf is not None,
        "Temporal CNN": temporal is not None,
        "Connectivity DL": conn is not None,
        "Connectivity DL (stride=4)": conn_s4 is not None,
    }
    return {
        "table": rows,
        "csv_path": csv_path,
        "plot_path": plot_path,
        "metrics_loaded": metrics_loaded,
    }
