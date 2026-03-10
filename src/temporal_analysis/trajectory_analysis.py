"""
Temporal trajectory analysis: compare evolution of graph metrics between Good and Poor outcome groups.

Loads window-level graph features per patient, normalizes time to [0, 1], resamples to fixed length,
computes group mean ± std trajectories, and summary statistics (early/late mean, slope).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .feature_schema import GRAPH_FEATURE_NAMES, N_GRAPH_FEATURES

# Biologically relevant metrics for network collapse (name, column index).
TRAJECTORY_METRICS = [
    ("global_efficiency", 3),
    ("clustering_coefficient", 1),   # average_clustering
    ("largest_component_ratio", 24),
    ("n_connected_components", 22),
    ("strength_mean", 14),            # average node strength
]

# Resampled time steps (normalized 0 -> 1).
DEFAULT_N_TIME_STEPS = 200

# Early/late window fraction for summary stats.
EARLY_QUANTILE = 0.25
LATE_QUANTILE = 0.75


def load_patient_features(
    patient_id: str,
    graph_features_dir: str,
    n_features: int = N_GRAPH_FEATURES,
) -> Optional[np.ndarray]:
    """
    Load window-level graph features for one patient.
    Returns (n_windows, n_features) or None if file missing/invalid.
    """
    path = os.path.join(graph_features_dir.rstrip("/"), f"{patient_id}_features.npy")
    if not os.path.isfile(path):
        return None
    try:
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[1] != n_features or arr.shape[0] < 2:
            return None
        return np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return None


def resample_trajectory(
    trajectory: np.ndarray,
    n_steps: int = DEFAULT_N_TIME_STEPS,
) -> np.ndarray:
    """
    Normalize time axis to [0, 1] and resample to n_steps.
    trajectory: (n_windows, n_features)
    Returns: (n_steps, n_features)
    """
    T, F = trajectory.shape
    if T == 0:
        return np.zeros((n_steps, F), dtype=np.float64)
    old_t = np.linspace(0.0, 1.0, T, dtype=np.float64)
    new_t = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
    out = np.zeros((n_steps, F), dtype=np.float64)
    for j in range(F):
        out[:, j] = np.interp(new_t, old_t, trajectory[:, j])
    return out


def load_and_resample_group(
    patient_ids: List[str],
    labels: np.ndarray,
    graph_features_dir: str,
    n_steps: int = DEFAULT_N_TIME_STEPS,
    outcome_good: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load features for all patients, split by outcome, resample to n_steps.
    Returns:
      good_trajectories: (n_good, n_steps, n_features)
      poor_trajectories: (n_poor, n_steps, n_features)
    """
    good_list: List[np.ndarray] = []
    poor_list: List[np.ndarray] = []
    labels = np.asarray(labels).ravel()
    for i, pid in enumerate(patient_ids):
        feat = load_patient_features(pid, graph_features_dir)
        if feat is None:
            continue
        resampled = resample_trajectory(feat, n_steps=n_steps)
        if labels[i] == outcome_good:
            good_list.append(resampled)
        else:
            poor_list.append(resampled)
    good_trajectories = np.stack(good_list, axis=0) if good_list else np.zeros((0, n_steps, N_GRAPH_FEATURES))
    poor_trajectories = np.stack(poor_list, axis=0) if poor_list else np.zeros((0, n_steps, N_GRAPH_FEATURES))
    return good_trajectories, poor_trajectories


def compute_group_trajectories(
    trajectories: np.ndarray,
    feature_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    trajectories: (n_patients, n_steps, n_features)
    Returns mean and std over patients for the given feature index: (n_steps,) each.
    """
    if trajectories.size == 0:
        n_steps = 200
        return np.zeros(n_steps), np.zeros(n_steps)
    col = trajectories[:, :, feature_idx]
    mean = np.mean(col, axis=0)
    std = np.std(col, axis=0)
    return mean, std


def compute_summary_statistics(
    trajectories: np.ndarray,
    feature_idx: int,
) -> Dict[str, float]:
    """
    trajectories: (n_patients, n_steps, n_features)
    Returns early_mean, late_mean, slope (linear regression over normalized time).
    """
    if trajectories.size == 0:
        return {"early_mean": np.nan, "late_mean": np.nan, "slope": np.nan}
    col = trajectories[:, :, feature_idx]
    n_steps = col.shape[1]
    early_end = max(1, int(n_steps * EARLY_QUANTILE))
    late_start = min(n_steps - 1, int(n_steps * LATE_QUANTILE))
    early_mean = float(np.nanmean(col[:, :early_end]))
    late_mean = float(np.nanmean(col[:, late_start:]))
    # Slope: regress col on time [0, 1] per patient, then average slope
    t = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
    slopes = []
    for i in range(col.shape[0]):
        y = col[i]
        if np.isfinite(y).all():
            cov = np.cov(t, y)[0, 1]
            var_t = np.var(t)
            if var_t > 1e-20:
                slopes.append(cov / var_t)
    slope = float(np.mean(slopes)) if slopes else np.nan
    return {"early_mean": early_mean, "late_mean": late_mean, "slope": slope}


def plot_trajectory(
    good_mean: np.ndarray,
    good_std: np.ndarray,
    poor_mean: np.ndarray,
    poor_std: np.ndarray,
    metric_name: str,
    save_path: str,
    n_steps: int = DEFAULT_N_TIME_STEPS,
) -> None:
    """
    Plot Good vs Poor trajectories with confidence bands (± std).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    t = np.linspace(0.0, 1.0, n_steps)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(t, good_mean - good_std, good_mean + good_std, alpha=0.3, color="C0")
    ax.plot(t, good_mean, color="C0", label="Good outcome (CPC 1–2)", linewidth=2)
    ax.fill_between(t, poor_mean - poor_std, poor_mean + poor_std, alpha=0.3, color="C1")
    ax.plot(t, poor_mean, color="C1", label="Poor outcome (CPC 3–5)", linewidth=2)
    ax.set_xlabel("Normalized time (0 = recording start, 1 = end)")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"Network trajectory: {metric_name.replace('_', ' ').title()}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_trajectory_analysis(
    patient_ids: List[str],
    labels: np.ndarray,
    graph_features_dir: str,
    output_dir: str,
    n_steps: int = DEFAULT_N_TIME_STEPS,
    metrics: Optional[List[Tuple[str, int]]] = None,
) -> Dict[str, Any]:
    """
    Load features, resample, compute group trajectories and summary stats, save plots and CSV.
    """
    try:
        import pandas as pd
    except ImportError:
        pd = None
    if metrics is None:
        metrics = TRAJECTORY_METRICS
    good_traj, poor_traj = load_and_resample_group(
        patient_ids, labels, graph_features_dir, n_steps=n_steps,
    )
    n_good, n_poor = good_traj.shape[0], poor_traj.shape[0]
    results = {"n_good": n_good, "n_poor": n_poor}

    rows: List[Dict[str, Any]] = []
    for metric_name, feat_idx in metrics:
        good_mean, good_std = compute_group_trajectories(good_traj, feat_idx)
        poor_mean, poor_std = compute_group_trajectories(poor_traj, feat_idx)
        plot_trajectory(
            good_mean, good_std, poor_mean, poor_std,
            metric_name=metric_name,
            save_path=os.path.join(output_dir, f"{metric_name}_trajectory.png"),
            n_steps=n_steps,
        )
        good_stats = compute_summary_statistics(good_traj, feat_idx)
        poor_stats = compute_summary_statistics(poor_traj, feat_idx)
        rows.append({
            "metric": metric_name,
            "good_early_mean": good_stats["early_mean"],
            "good_late_mean": good_stats["late_mean"],
            "good_slope": good_stats["slope"],
            "poor_early_mean": poor_stats["early_mean"],
            "poor_late_mean": poor_stats["late_mean"],
            "poor_slope": poor_stats["slope"],
        })
    results["statistics"] = rows

    csv_path = os.path.join(output_dir, "network_collapse_statistics.csv")
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    if pd is not None:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    return results
