"""
Segment filtered EEG signals into fixed-length windows.

Produces non-overlapping windows of configurable length (e.g. 30 seconds).
Designed so that each window can be processed independently (e.g. for
future connectivity computation inside the loop).
"""

from typing import Iterator, List

import numpy as np


def segment_into_windows(
    signal: np.ndarray,
    fs: float,
    window_seconds: float = 30.0,
) -> Iterator[np.ndarray]:
    """
    Yield non-overlapping windows of fixed duration.

    Args:
        signal: Array of shape (n_samples, n_channels).
        fs: Sampling frequency in Hz.
        window_seconds: Duration of each window in seconds (default 30.0).

    Yields:
        Windows of shape (window_samples, n_channels). Partial final
        window is dropped if shorter than window_seconds.
    """
    n_samples, n_channels = signal.shape
    window_samples = int(round(window_seconds * fs))
    if window_samples <= 0:
        return

    n_full = n_samples // window_samples
    for i in range(n_full):
        start = i * window_samples
        yield signal[start : start + window_samples, :].copy()


def segment_into_windows_list(
    signal: np.ndarray,
    fs: float,
    window_seconds: float = 30.0,
) -> List[np.ndarray]:
    """
    Segment signal into fixed-length windows and return as a list.

    Convenience function when all windows are needed in memory.
    For large signals, prefer segment_into_windows() to process one at a time.

    Args:
        signal: Array of shape (n_samples, n_channels).
        fs: Sampling frequency in Hz.
        window_seconds: Duration of each window in seconds (default 30.0).

    Returns:
        List of arrays, each of shape (window_samples, n_channels).
    """
    return list(segment_into_windows(signal, fs, window_seconds))
