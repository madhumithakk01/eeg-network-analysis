"""
Window-level EEG quality checks for preprocessing.

Designed for I-CARE v2.1 where amplitudes may be unitless ("nu"), so checks
rely on robust relative statistics rather than fixed microvolt thresholds.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _mains_power_ratio(
    window: np.ndarray,
    fs: float,
    mains_hz: float,
    mains_band_hz: float,
    low_hz: float = 1.0,
    high_hz: float = 40.0,
) -> float:
    """Compute ratio of power around mains frequency to total band power."""
    if fs <= 0 or window.size == 0:
        return 0.0
    n_samples = window.shape[0]
    if n_samples < 16:
        return 0.0
    x = window - np.mean(window, axis=0, keepdims=True)
    spec = np.fft.rfft(x, axis=0)
    pxx = (np.abs(spec) ** 2).mean(axis=1)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band_mask):
        return 0.0
    total_power = float(np.sum(pxx[band_mask]))
    if total_power <= 1e-20:
        return 0.0

    mains_mask = (freqs >= mains_hz - mains_band_hz) & (freqs <= mains_hz + mains_band_hz)
    mains_power = float(np.sum(pxx[mains_mask]))
    return mains_power / total_power


def evaluate_window_quality(
    window: np.ndarray,
    fs: float,
    *,
    flat_std_min_abs: float = 1e-8,
    flat_std_min_ratio: float = 1e-3,
    max_flat_channel_frac: float = 0.2,
    min_unique_value_ratio: float = 0.02,
    max_low_unique_channel_frac: float = 0.3,
    high_amp_robust_z: float = 12.0,
    max_high_amp_frac: float = 0.02,
    mains_hz: float = 50.0,
    mains_band_hz: float = 1.0,
    max_mains_ratio: float = 0.35,
) -> Tuple[bool, Dict[str, float], str]:
    """
    Return (is_valid, diagnostics, reason).

    reason is "ok" when valid, otherwise one of:
      - too_many_flat_channels
      - too_many_low_unique_channels
      - excessive_amplitude_outliers
      - excessive_mains_power
    """
    n_samples, n_channels = window.shape
    if n_samples < 8 or n_channels < 2:
        return False, {"n_samples": float(n_samples), "n_channels": float(n_channels)}, "too_short_window"

    ch_std = np.std(window, axis=0)
    med_std = float(np.median(ch_std))
    flat_thr = max(flat_std_min_abs, flat_std_min_ratio * max(med_std, 1e-12))
    flat_mask = ch_std < flat_thr
    flat_frac = float(np.mean(flat_mask))

    # Detect clipping/quantization/flat traces via low uniqueness.
    low_unique = 0
    for c in range(n_channels):
        uniq_ratio = float(np.unique(window[:, c]).size) / float(n_samples)
        if uniq_ratio < min_unique_value_ratio:
            low_unique += 1
    low_unique_frac = float(low_unique) / float(n_channels)

    # Robust amplitude outlier fraction (global, unit-agnostic).
    x = window.astype(np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = max(1.4826 * mad, 1e-12)
    robust_z = np.abs((x - med) / scale)
    high_amp_frac = float(np.mean(robust_z > high_amp_robust_z))

    mains_ratio = _mains_power_ratio(
        window=window,
        fs=fs,
        mains_hz=mains_hz,
        mains_band_hz=mains_band_hz,
    )

    diagnostics = {
        "flat_frac": flat_frac,
        "low_unique_frac": low_unique_frac,
        "high_amp_frac": high_amp_frac,
        "mains_ratio": mains_ratio,
        "median_channel_std": med_std,
    }

    if flat_frac > max_flat_channel_frac:
        return False, diagnostics, "too_many_flat_channels"
    if low_unique_frac > max_low_unique_channel_frac:
        return False, diagnostics, "too_many_low_unique_channels"
    if high_amp_frac > max_high_amp_frac:
        return False, diagnostics, "excessive_amplitude_outliers"
    if mains_ratio > max_mains_ratio:
        return False, diagnostics, "excessive_mains_power"
    return True, diagnostics, "ok"


def filter_windows_by_quality(
    windows_list: List[np.ndarray],
    fs: float,
    *,
    flat_std_min_abs: float = 1e-8,
    flat_std_min_ratio: float = 1e-3,
    max_flat_channel_frac: float = 0.2,
    min_unique_value_ratio: float = 0.02,
    max_low_unique_channel_frac: float = 0.3,
    high_amp_robust_z: float = 12.0,
    max_high_amp_frac: float = 0.02,
    mains_hz: float = 50.0,
    mains_band_hz: float = 1.0,
    max_mains_ratio: float = 0.35,
) -> Tuple[List[np.ndarray], Dict[str, int]]:
    """Filter windows using evaluate_window_quality and return kept windows + stats."""
    kept: List[np.ndarray] = []
    stats = {
        "n_windows_total": len(windows_list),
        "n_windows_kept": 0,
        "n_reject_too_short_window": 0,
        "n_reject_too_many_flat_channels": 0,
        "n_reject_too_many_low_unique_channels": 0,
        "n_reject_excessive_amplitude_outliers": 0,
        "n_reject_excessive_mains_power": 0,
    }
    for w in windows_list:
        ok, _, reason = evaluate_window_quality(
            w,
            fs,
            flat_std_min_abs=flat_std_min_abs,
            flat_std_min_ratio=flat_std_min_ratio,
            max_flat_channel_frac=max_flat_channel_frac,
            min_unique_value_ratio=min_unique_value_ratio,
            max_low_unique_channel_frac=max_low_unique_channel_frac,
            high_amp_robust_z=high_amp_robust_z,
            max_high_amp_frac=max_high_amp_frac,
            mains_hz=mains_hz,
            mains_band_hz=mains_band_hz,
            max_mains_ratio=max_mains_ratio,
        )
        if ok:
            kept.append(w)
        else:
            key = f"n_reject_{reason}"
            if key in stats:
                stats[key] += 1
    stats["n_windows_kept"] = len(kept)
    return kept, stats

