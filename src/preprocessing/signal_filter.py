"""
Apply bandpass filtering to EEG signals.

Uses a Butterworth filter (scipy) with zero-phase filtering (filtfilt)
to avoid phase distortion. Filter parameters come from config
(BANDPASS_LOW, BANDPASS_HIGH).
"""

from typing import Union

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 4,
    notch_freqs: list[float] | None = None,
    notch_q: float = 30.0,
) -> np.ndarray:
    """
    Apply a zero-phase bandpass filter to EEG signal(s).

    Args:
        signal: Array of shape (n_samples,) or (n_samples, n_channels).
        fs: Sampling frequency in Hz.
        low_hz: Lower cutoff frequency in Hz (default 0.5).
        high_hz: Upper cutoff frequency in Hz (default 40.0).
        order: Butterworth filter order (default 4).

    Returns:
        Filtered array with the same shape as signal.
    """
    nyq = 0.5 * fs
    low = max(low_hz / nyq, 1e-9)
    high = min(high_hz / nyq, 1.0 - 1e-9)
    if low >= high:
        return signal

    b, a = butter(order, [low, high], btype="band")
    single_channel = signal.ndim == 1
    if single_channel:
        signal = signal[:, np.newaxis]

    out = np.empty_like(signal)
    for ch in range(signal.shape[1]):
        x = filtfilt(b, a, signal[:, ch], axis=0)
        # Optional utility-frequency notch suppression (e.g., 50/60 Hz)
        if notch_freqs:
            for f0 in notch_freqs:
                if f0 <= 0 or f0 >= nyq:
                    continue
                bn, an = iirnotch(w0=f0, Q=max(float(notch_q), 1.0), fs=fs)
                x = filtfilt(bn, an, x, axis=0)
        out[:, ch] = x

    if single_channel:
        out = out.squeeze(axis=1)
    return out
