"""
Sampling-rate harmonization utilities for EEG preprocessing.

Uses polyphase resampling for stable, anti-aliased conversion to a target fs.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Tuple

import numpy as np
from scipy.signal import resample_poly


def resample_signal_poly(
    signal: np.ndarray,
    fs_in: float,
    fs_target: float,
    max_denominator: int = 1000,
) -> Tuple[np.ndarray, float]:
    """
    Resample signal from fs_in to fs_target with polyphase filtering.

    Parameters
    ----------
    signal : np.ndarray
        Shape (n_samples, n_channels).
    fs_in : float
        Input sampling frequency in Hz.
    fs_target : float
        Target sampling frequency in Hz.
    max_denominator : int
        Max denominator for rational approximation of fs_target/fs_in.

    Returns
    -------
    (resampled_signal, fs_out)
        resampled_signal has shape (~n_samples * fs_target/fs_in, n_channels).
        fs_out equals fs_target.
    """
    if signal.ndim != 2:
        raise ValueError(f"signal must be 2D (n_samples, n_channels), got shape={signal.shape}")
    if fs_in <= 0 or fs_target <= 0:
        raise ValueError(f"fs_in and fs_target must be positive, got fs_in={fs_in}, fs_target={fs_target}")
    if abs(fs_in - fs_target) < 1e-6:
        return signal, float(fs_in)

    ratio = Fraction(fs_target / fs_in).limit_denominator(max_denominator)
    up, down = ratio.numerator, ratio.denominator

    out = resample_poly(signal, up=up, down=down, axis=0)
    return np.asarray(out, dtype=np.float64), float(fs_target)

