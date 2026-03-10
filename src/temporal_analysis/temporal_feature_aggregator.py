"""
Temporal aggregation of window-level graph features into patient-level descriptors.

Transforms (n_windows, 40) into a fixed-length vector (~200-300) of temporal
statistics, trends, early/late contrasts, variability, and collapse indicators.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.signal import savgol_filter

from .feature_schema import (
    COLLAPSE_CRITICAL_INDICES,
    N_GRAPH_FEATURES,
)

# Minimum windows required for temporal analysis (trend, early/late split).
MIN_WINDOWS_TEMPORAL = 10
# Rolling window for variance and optional smoothing.
ROLLING_WINDOW = 50
# Early/late phase: first and last 25%.
EARLY_QUANTILE = 0.25
LATE_QUANTILE = 0.75
# Savitzky-Golay for optional smoothing (window length must be odd).
SAVGOL_WINDOW = 21
SAVGOL_POLY = 3


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Numerically stable division."""
    if b == 0 or not np.isfinite(b):
        return default
    return float(a / b) if np.isfinite(a) else default


def _nan_to_zero(x: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0 in place or copy."""
    out = np.asarray(x, dtype=np.float64)
    out[~np.isfinite(out)] = 0.0
    return out


def _basic_stats_per_feature(features: np.ndarray) -> dict[str, np.ndarray]:
    """Compute mean, std, median, min, max, IQR per column. Shape (n_windows, n_feat) -> 6 * n_feat."""
    x = _nan_to_zero(features)
    n_feat = x.shape[1]
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0] = 1e-12  # avoid 0 for later ratios
    median = np.median(x, axis=0)
    min_ = np.min(x, axis=0)
    max_ = np.max(x, axis=0)
    q1 = np.percentile(x, 25, axis=0)
    q3 = np.percentile(x, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1e-12
    return {
        "mean": mean, "std": std, "median": median,
        "min": min_, "max": max_, "iqr": iqr,
    }


def _temporal_trend_features(features: np.ndarray) -> dict[str, np.ndarray]:
    """Linear regression slope, intercept, R² vs normalized time [0,1] per column."""
    n_win, n_feat = features.shape
    x = _nan_to_zero(features)
    t = np.linspace(0.0, 1.0, n_win, dtype=np.float64)
    t_var = np.var(t)
    if t_var < 1e-20:
        t_var = 1e-12
    slope = np.zeros(n_feat)
    intercept = np.zeros(n_feat)
    r_squared = np.zeros(n_feat)
    for j in range(n_feat):
        y = x[:, j]
        mean_y = np.mean(y)
        cov_ty = np.cov(t, y)[0, 1] if n_win > 1 else 0.0
        slope[j] = cov_ty / t_var if t_var else 0.0
        intercept[j] = mean_y - slope[j] * np.mean(t)
        y_pred = intercept[j] + slope[j] * t
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - mean_y) ** 2)
        r_squared[j] = 1.0 - _safe_div(ss_res, ss_tot, 0.0)
    return {"slope": slope, "intercept": intercept, "r_squared": r_squared}


def _early_late_features(features: np.ndarray) -> dict[str, np.ndarray]:
    """First 25% vs last 25%: early_mean, late_mean, late_minus_early, late_over_early."""
    n_win, n_feat = features.shape
    x = _nan_to_zero(features)
    n_early = max(1, int(n_win * EARLY_QUANTILE))
    n_late = max(1, int(n_win * (1 - LATE_QUANTILE)))
    early = x[:n_early]
    late = x[-n_late:]
    early_mean = np.mean(early, axis=0)
    late_mean = np.mean(late, axis=0)
    diff = late_mean - early_mean
    ratio = np.zeros(n_feat)
    for j in range(n_feat):
        ratio[j] = _safe_div(late_mean[j], early_mean[j], 1.0)
    return {
        "early_mean": early_mean,
        "late_mean": late_mean,
        "late_minus_early": diff,
        "late_over_early": ratio,
    }


def _rolling_variance_and_derivative(features: np.ndarray) -> dict[str, np.ndarray]:
    """Mean rolling variance (window=50) and mean absolute first derivative per column."""
    n_win, n_feat = features.shape
    x = _nan_to_zero(features)
    w = min(ROLLING_WINDOW, n_win - 1) if n_win > 1 else 1
    if w < 2:
        return {"mean_rolling_var": np.zeros(n_feat), "mean_abs_derivative": np.zeros(n_feat)}
    mean_rolling_var = np.zeros(n_feat)
    for j in range(n_feat):
        col = x[:, j]
        pad = np.pad(col, (w // 2, w - 1 - w // 2), mode="edge")
        rolling_var = np.array([np.var(pad[i : i + w]) for i in range(n_win)])
        mean_rolling_var[j] = np.mean(rolling_var) if np.isfinite(rolling_var).all() else 0.0
    # First derivative (central difference)
    deriv = np.zeros_like(x)
    deriv[1:-1] = (x[2:] - x[:-2]) / 2.0
    if n_win > 1:
        deriv[0] = x[1] - x[0]
        deriv[-1] = x[-1] - x[-2]
    mean_abs_derivative = np.mean(np.abs(deriv), axis=0)
    return {"mean_rolling_var": mean_rolling_var, "mean_abs_derivative": mean_abs_derivative}


def _collapse_indicators_for_metric(series: np.ndarray) -> dict[str, float]:
    """For one time series: min_value, time_of_min, max_drop_magnitude, max_negative_slope."""
    s = _nan_to_zero(series.flatten())
    n = len(s)
    out = {
        "min_value": float(np.min(s)) if n else 0.0,
        "time_of_min": 0.0,
        "max_drop_magnitude": 0.0,
        "max_negative_slope": 0.0,
    }
    if n == 0:
        return out
    out["time_of_min"] = float(np.argmin(s)) / max(1, n - 1)  # normalized [0,1]
    # Max drop: max over t of (max(s[:t+1]) - s[t])
    cummax = np.maximum.accumulate(s)
    drop = cummax - s
    out["max_drop_magnitude"] = float(np.max(drop)) if n else 0.0
    # Max negative slope (first difference)
    if n >= 2:
        diff = np.diff(s)
        neg_slopes = diff[diff < 0]
        out["max_negative_slope"] = float(np.max(np.abs(neg_slopes))) if len(neg_slopes) else 0.0
    return out


def _smooth_optional(features: np.ndarray, use_smoothing: bool, method: str = "rolling") -> np.ndarray:
    """Optional smoothing: rolling mean (window=20) or Savitzky-Golay."""
    if not use_smoothing or features.shape[0] < 10:
        return features.copy()
    x = _nan_to_zero(features)
    n_win = x.shape[0]
    if method == "savgol":
        w = min(SAVGOL_WINDOW, n_win if n_win % 2 == 1 else n_win - 1)
        if w < SAVGOL_POLY + 2:
            return x
        try:
            return savgol_filter(x, w, SAVGOL_POLY, axis=0)
        except Exception:
            return x
    # rolling mean
    w = min(20, n_win)
    kernel = np.ones(w) / w
    out = np.zeros_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.convolve(x[:, j], kernel, mode="same")
    return out


def aggregate_temporal_features(
    features: np.ndarray,
    collapse_critical_indices: dict[str, int] | None = None,
    use_smoothing: bool = False,
    smoothing_method: str = "rolling",
    max_descriptors: int | None = 320,
) -> tuple[np.ndarray, list[str]]:
    """
    Transform window-level graph features into patient-level temporal descriptor vector.

    Parameters
    ----------
    features : np.ndarray, shape (n_windows, 40)
        Graph feature matrix, float32/64.
    collapse_critical_indices : dict, optional
        Map metric name -> column index for collapse indicators. Default: feature_schema.COLLAPSE_CRITICAL_INDICES.
    use_smoothing : bool
        If True, smooth series before trend/collapse (rolling or savgol).
    smoothing_method : str
        'rolling' or 'savgol'.
    max_descriptors : int, optional
        If set, keep only first max_descriptors (for ~250-320 target). None = return all.

    Returns
    -------
    aggregated : np.ndarray, shape (n_descriptors,), dtype float64
        Patient-level feature vector (no NaN).
    names : list[str]
        Feature names in order (for DataFrame columns).
    """
    if collapse_critical_indices is None:
        collapse_critical_indices = COLLAPSE_CRITICAL_INDICES

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != N_GRAPH_FEATURES:
        raise ValueError(f"Expected shape (n_windows, {N_GRAPH_FEATURES}), got {features.shape}")

    n_win, n_feat = features.shape
    if n_win < MIN_WINDOWS_TEMPORAL:
        pass

    x = _smooth_optional(features, use_smoothing, smoothing_method)

    values: list[np.ndarray] = []
    names: list[str] = []

    # 1. Basic statistics per feature: mean, std, median, min, max, IQR (6 * 40 = 240)
    basic = _basic_stats_per_feature(x)
    for stat in ("mean", "std", "median", "min", "max", "iqr"):
        values.append(basic[stat])
        names.extend([f"f{j:02d}_{stat}" for j in range(n_feat)])

    # 2. Temporal trend: slope, intercept, R² (3 * 40 = 120)
    trend = _temporal_trend_features(x)
    for key in ("slope", "intercept", "r_squared"):
        values.append(trend[key])
        names.extend([f"f{j:02d}_trend_{key}" for j in range(n_feat)])

    # 3. Early vs late (4 * 40 = 160)
    early_late = _early_late_features(x)
    for key in ("early_mean", "late_mean", "late_minus_early", "late_over_early"):
        values.append(early_late[key])
        names.extend([f"f{j:02d}_{key}" for j in range(n_feat)])

    # 4. Variability (2 * 40 = 80)
    var_deriv = _rolling_variance_and_derivative(x)
    values.append(var_deriv["mean_rolling_var"])
    names.extend([f"f{j:02d}_mean_rolling_var" for j in range(n_feat)])
    values.append(var_deriv["mean_abs_derivative"])
    names.extend([f"f{j:02d}_mean_abs_derivative" for j in range(n_feat)])

    # 5. Collapse indicators for critical metrics (4 per metric)
    for metric_name, col_idx in collapse_critical_indices.items():
        if col_idx >= n_feat:
            continue
        ind = _collapse_indicators_for_metric(x[:, col_idx])
        for k, v in ind.items():
            values.append(np.array([v]))
            names.append(f"collapse_{metric_name}_{k}")

    aggregated = np.concatenate([v.ravel() for v in values])
    aggregated = _nan_to_zero(aggregated)

    if n_win < MIN_WINDOWS_TEMPORAL:
        aggregated = np.zeros_like(aggregated)

    if max_descriptors is not None and len(aggregated) > max_descriptors:
        aggregated = aggregated[:max_descriptors]
        names = names[:max_descriptors]

    return aggregated.astype(np.float64), names


def get_aggregated_feature_count(max_descriptors: int | None = 320) -> int:
    """Return length of aggregated vector (for validation)."""
    n_feat = N_GRAPH_FEATURES
    n_basic = 6 * n_feat
    n_trend = 3 * n_feat
    n_early_late = 4 * n_feat
    n_var = 2 * n_feat
    n_collapse = 4 * len(COLLAPSE_CRITICAL_INDICES)
    total = n_basic + n_trend + n_early_late + n_var + n_collapse
    if max_descriptors is not None and total > max_descriptors:
        return max_descriptors
    return total
