"""
Load EEG signals from WFDB record files.

Reads .mat/.hea pairs, selects channels by name, and returns a signal array
plus sampling rate. Used for segment-by-segment processing in the
preprocessing pipeline.
"""

import os
from typing import List, Tuple

import numpy as np
import wfdb


def load_eeg_segment(
    record_path: str,
    channel_names: List[str],
) -> Tuple[np.ndarray, float]:
    """
    Load one EEG segment from a WFDB record and return signals for the given channels.

    Args:
        record_path: Path to the record (with or without .hea/.mat extension).
                     Example: /path/to/0284_001_000_EEG or /path/to/0284_001_000_EEG.hea
        channel_names: List of channel names to load (e.g. from common_eeg_channels.json).
                       Order in the returned array matches this list. Channels missing
                       in the record are skipped; at least one must be present.

    Returns:
        signals: numpy array of shape (n_samples, n_channels) in physical units.
                 Channels are in the order of channel_names (only those present).
        fs: Sampling frequency in Hz.

    Raises:
        FileNotFoundError: If the record files are missing.
        ValueError: If no requested channel is found in the record.
    """
    path = os.path.abspath(record_path)
    if path.lower().endswith(".hea"):
        path = path[:-4]
    if path.lower().endswith(".mat"):
        path = path[:-4]

    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path)
    if dir_name:
        record = wfdb.rdrecord(
            record_name=base_name,
            pn_dir=dir_name,
            channel_names=channel_names,
        )
    else:
        record = wfdb.rdrecord(record_name=base_name, channel_names=channel_names)

    if record.p_signal is not None:
        data = np.asarray(record.p_signal, dtype=np.float64)
    else:
        data = np.asarray(record.d_signal, dtype=np.float64)

    fs = float(record.fs)
    if data.size == 0:
        raise ValueError(f"No signal data for record {record_path}")

    # rdrecord with channel_names returns only requested channels in order
    # Shape is (n_samples, n_channels)
    return data, fs
