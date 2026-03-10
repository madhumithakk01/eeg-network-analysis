"""
Network Collapse Index (NCI) and variants for temporal network deterioration.

Computes composite collapse scores from window-level graph features:
- NCI (basic): efficiency drop, clustering drop, fragmentation, path length increase.
- SCI (Spectral Collapse Index): spectral radius/energy/gap changes.
- NFCI (Network Fragmentation Collapse Index): component/fragmentation growth.
- TNSI (Temporal Network Stability Index): temporal variance and derivative magnitude.

All implementations return (collapse_score, collapse_time, collapse_severity)
and are numerically stable for constant signals, NaNs, and short series.
"""

from __future__ import annotations

import numpy as np
from typing import NamedTuple

from .feature_schema import (
    COLLAPSE_CRITICAL_INDICES,
    FRAGMENTATION_INDICES,
    SPECTRAL_INDICES,
    STABILITY_FEATURE_INDICES,
    N_GRAPH_FEATURES,
)

# Optional change-point detection (ruptures); use try/except for import.
try:
    import ruptures as rpt
    _HAS_RUPTURES = True
except ImportError:
    _HAS_RUPTURES = False


class CollapseResult(NamedTuple):
    """Result of collapse index computation."""
    collapse_score: float
    collapse_time: float   # normalized [0,1] or index
    collapse_severity: float


def _safe_norm(x: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Z-score normalize; constant or zero-variance -> 0."""
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.nan)
    mu = np.nanmean(x, axis=axis, keepdims=True)
    sigma = np.nanstd(x, axis=axis, keepdims=True)
    if axis is None:
        if sigma == 0 or not np.isfinite(sigma):
            return np.zeros_like(x)
        return (x - mu) / sigma
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    out = (x - mu) / sigma
    return np.where(np.isfinite(out), out, 0.0)


def _extract_series(features: np.ndarray, col_idx: int) -> np.ndarray:
    """Get column, replace NaN with 0."""
    if features.shape[1] <= col_idx:
        return np.zeros(features.shape[0])
    s = np.asarray(features[:, col_idx], dtype=np.float64)
    s[~np.isfinite(s)] = 0.0
    return s


def compute_nci_basic(
    features: np.ndarray,
    weights: tuple[float, float, float, float] | None = None,
) -> CollapseResult:
    """
    Basic Network Collapse Index from critical metrics.

    NCI = w1*efficiency_drop + w2*clustering_drop + w3*fragmentation + w4*path_increase.
    Drops and increases are computed as (max - min) or (min - max) so higher = worse.
    All terms are z-score normalized across time then combined.
    """
    if weights is None:
        weights = (0.25, 0.25, 0.25, 0.25)
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.size == 0:
        return CollapseResult(0.0, 0.0, 0.0)

    n_win = features.shape[0]
    if n_win < 2:
        return CollapseResult(0.0, 0.0, 0.0)

    eff_idx = COLLAPSE_CRITICAL_INDICES["global_efficiency"]
    clust_idx = COLLAPSE_CRITICAL_INDICES["clustering_coefficient"]
    lcc_idx = COLLAPSE_CRITICAL_INDICES["largest_component_ratio"]
    path_idx = COLLAPSE_CRITICAL_INDICES["path_length"]
    n_comp_idx = COLLAPSE_CRITICAL_INDICES["n_components"]

    eff = _extract_series(features, eff_idx)
    clust = _extract_series(features, clust_idx)
    lcc = _extract_series(features, lcc_idx)
    path = _extract_series(features, path_idx)
    n_comp = _extract_series(features, n_comp_idx)

    # Degradation: lower efficiency/clustering/LCC = worse; higher path/n_comp = worse
    efficiency_drop = np.max(eff) - np.min(eff)  # range; then we want "drop" as severity
    clustering_drop = np.max(clust) - np.min(clust)
    largest_component_loss = np.max(lcc) - np.min(lcc)  # LCC going down
    path_length_increase = np.max(path) - np.min(path)
    component_fragmentation = np.max(n_comp) - np.min(n_comp)

    # Normalize each to [0,1] scale by their range so they're comparable
    def _norm01(x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))

    # Collapse score: weighted sum of normalized degradation magnitudes
    c1 = _norm01(efficiency_drop, 0, 1.0)
    c2 = _norm01(clustering_drop, 0, 1.0)
    c3 = _norm01(component_fragmentation, 0, 19.0)  # n_components max ~19
    c4 = _norm01(path_length_increase, 0, 10.0)     # path length scale
    collapse_score = weights[0] * c1 + weights[1] * c2 + weights[2] * c3 + weights[3] * c4

    # Collapse time: when efficiency is minimum (worst moment)
    collapse_time_idx = np.argmin(eff)
    collapse_time = float(collapse_time_idx) / max(1, n_win - 1)

    # Severity: max drop in efficiency over time (cummax - current)
    cummax_eff = np.maximum.accumulate(eff)
    severity = np.max(cummax_eff - eff)
    collapse_severity = float(severity) if np.isfinite(severity) else 0.0

    return CollapseResult(
        collapse_score=float(np.clip(collapse_score, 0.0, 1.0)),
        collapse_time=collapse_time,
        collapse_severity=collapse_severity,
    )


def compute_nci_spectral(features: np.ndarray) -> CollapseResult:
    """
    Spectral Collapse Index: radius_drop + energy_drop + gap_std.
    Brain networks losing integration show shrinking spectral radius and energy.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.size == 0:
        return CollapseResult(0.0, 0.0, 0.0)
    n_win = features.shape[0]
    if n_win < 2:
        return CollapseResult(0.0, 0.0, 0.0)

    rad_idx = SPECTRAL_INDICES["spectral_radius"]
    energy_idx = SPECTRAL_INDICES["graph_energy"]
    gap_idx = SPECTRAL_INDICES["spectral_gap"]

    radius = _extract_series(features, rad_idx)
    energy = _extract_series(features, energy_idx)
    gap = _extract_series(features, gap_idx)

    radius_drop = np.max(radius) - np.min(radius)
    energy_drop = np.max(energy) - np.min(energy)
    gap_change = np.std(gap) if len(gap) > 1 else 0.0

    # Normalize to comparable scale
    r_norm = radius_drop / (np.max(radius) + 1e-12)
    e_norm = energy_drop / (np.max(energy) + 1e-12)
    g_norm = gap_change / (np.std(radius) + 1e-12) if np.std(radius) > 1e-12 else 0.0
    sci = float(np.clip(r_norm + e_norm + g_norm, 0.0, 3.0) / 3.0)

    collapse_time_idx = np.argmin(radius)
    collapse_time = float(collapse_time_idx) / max(1, n_win - 1)
    severity = np.max(radius) - np.min(radius)
    return CollapseResult(
        collapse_score=sci,
        collapse_time=collapse_time,
        collapse_severity=float(severity),
    )


def compute_nci_fragmentation(features: np.ndarray) -> CollapseResult:
    """
    Network Fragmentation Collapse Index: growth in components, loss in LCC ratio, entropy increase.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.size == 0:
        return CollapseResult(0.0, 0.0, 0.0)
    n_win = features.shape[0]
    if n_win < 2:
        return CollapseResult(0.0, 0.0, 0.0)

    n_comp = _extract_series(features, FRAGMENTATION_INDICES["n_components"])
    lcc_ratio = _extract_series(features, FRAGMENTATION_INDICES["largest_component_ratio"])
    entropy = _extract_series(features, FRAGMENTATION_INDICES["component_entropy"])

    frag_growth = np.max(n_comp) - np.min(n_comp)
    lcc_loss = np.max(lcc_ratio) - np.min(lcc_ratio)
    entropy_inc = np.max(entropy) - np.min(entropy)

    w1, w2, w3 = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    f_norm = frag_growth / 19.0
    l_norm = lcc_loss / 1.0
    e_norm = entropy_inc / (np.max(entropy) + 1e-12)
    nfci = float(np.clip(w1 * f_norm + w2 * l_norm + w3 * e_norm, 0.0, 1.0))

    collapse_time_idx = np.argmin(lcc_ratio)
    collapse_time = float(collapse_time_idx) / max(1, n_win - 1)
    severity = lcc_loss
    return CollapseResult(
        collapse_score=nfci,
        collapse_time=collapse_time,
        collapse_severity=float(severity),
    )


def compute_nci_temporal_stability(features: np.ndarray) -> CollapseResult:
    """
    Temporal Network Stability Index: average temporal variance + average derivative magnitude
    across critical features. High instability suggests disrupted regulation.
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.size == 0:
        return CollapseResult(0.0, 0.0, 0.0)
    n_win = features.shape[0]
    if n_win < 2:
        return CollapseResult(0.0, 0.0, 0.0)

    indices = [i for i in STABILITY_FEATURE_INDICES if i < features.shape[1]]
    if not indices:
        return CollapseResult(0.0, 0.0, 0.0)

    variances = []
    deriv_mags = []
    for j in indices:
        s = _extract_series(features, j)
        variances.append(np.var(s))
        d = np.abs(np.diff(s))
        deriv_mags.append(np.mean(d) if len(d) else 0.0)
    tnsi = float(np.mean(variances) + np.mean(deriv_mags))
    # Normalize to [0,1] heuristically
    tnsi_norm = np.clip(tnsi / (1.0 + tnsi), 0.0, 1.0)

    # Collapse time: window with max combined derivative (most unstable moment)
    combined_deriv = np.zeros(n_win)
    for j in indices:
        s = _extract_series(features, j)
        d = np.abs(np.diff(s, prepend=s[0]))
        combined_deriv += d[:n_win] if len(d) >= n_win else np.pad(d, (0, n_win - len(d)))
    collapse_time_idx = np.argmax(combined_deriv)
    collapse_time = float(collapse_time_idx) / max(1, n_win - 1)
    collapse_severity = float(np.max(combined_deriv))
    return CollapseResult(
        collapse_score=float(tnsi_norm),
        collapse_time=collapse_time,
        collapse_severity=collapse_severity,
    )


def detect_collapse_change_point(
    features: np.ndarray,
    metric: str = "global_efficiency",
    n_bkps: int = 1,
) -> tuple[float, list[int]]:
    """
    Optional change-point detection for collapse time (ruptures).
    Returns (collapse_time_normalized, list of break indices).
    """
    if not _HAS_RUPTURES:
        eff_idx = COLLAPSE_CRITICAL_INDICES.get(metric, 3)
        s = _extract_series(features, eff_idx)
        if len(s) < 10:
            return 0.0, []
        collapse_time_idx = np.argmin(s)
        return float(collapse_time_idx) / max(1, len(s) - 1), [collapse_time_idx]

    col_idx = COLLAPSE_CRITICAL_INDICES.get(metric, 3)
    if features.shape[1] <= col_idx:
        return 0.0, []
    signal = _extract_series(features, col_idx).reshape(-1, 1).astype(np.float64)
    n = len(signal)
    if n < 2 * (n_bkps + 1):
        return 0.0, []
    try:
        algo = rpt.Pelt(model="rbf").fit(signal)
        bkps = algo.predict(pen=1.0)
        if not bkps:
            return 0.0, []
        # First break as collapse time
        bkp = bkps[0] if isinstance(bkps[0], int) else bkps[0][0]
        return float(bkp) / max(1, n - 1), list(bkps) if isinstance(bkps, list) else [bkp]
    except Exception:
        collapse_time_idx = np.argmin(signal.ravel())
        return float(collapse_time_idx) / max(1, n - 1), [collapse_time_idx]


def compute_recovery_score(
    features: np.ndarray,
    metric: str = "global_efficiency",
    early_quantile: float = 0.25,
    late_quantile: float = 0.75,
) -> float:
    """
    Recovery score: do late-phase values recover toward early-phase (good)?
    Return ratio late_mean / early_mean ( > 1 = recovery, < 1 = further decline).
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.size == 0:
        return 1.0
    col_idx = COLLAPSE_CRITICAL_INDICES.get(metric, 3)
    if features.shape[1] <= col_idx:
        return 1.0
    s = _extract_series(features, col_idx)
    n = len(s)
    n_early = max(1, int(n * early_quantile))
    n_late = max(1, int(n * (1 - late_quantile)))
    early_mean = np.mean(s[:n_early])
    late_mean = np.mean(s[-n_late:])
    if early_mean == 0 or not np.isfinite(early_mean):
        return 1.0
    return float(late_mean / early_mean)
