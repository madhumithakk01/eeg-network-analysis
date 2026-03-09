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
    Load one EEG segment from a WFDB record on the local filesystem.

    Reads .hea/.mat from the given path only; no remote (PhysioNet) access.
    Uses record_path directly so WFDB loads from disk without pn_dir.

    Args:
        record_path: Full path to the record (with or without .hea/.mat extension).
                     Example: /content/drive/MyDrive/icare_project/data/raw/eeg/0284/0284_001_004_EEG
        channel_names: List of channel names to load (e.g. from common_eeg_channels.json).
                       Order in the returned array matches this list.

    Returns:
        signals: numpy array of shape (n_samples, n_channels) in physical units.
        fs: Sampling frequency in Hz.

    Raises:
        FileNotFoundError: If the record files are missing.
        ValueError: If any requested channel is missing (message includes record_path).
    """
    path = os.path.abspath(record_path)
    if path.lower().endswith(".hea"):
        path = path[:-4]
    if path.lower().endswith(".mat"):
        path = path[:-4]
    if not path:
        raise ValueError(f"Invalid record_path: {record_path}")

    # Load from local filesystem only; do not use pn_dir (avoids PhysioNet fetch)
    record = wfdb.rdrecord(path)

    if record.sig_name is None:
        raise ValueError(f"No channel names in record: {record_path}")

    for ch in channel_names:
        if ch not in record.sig_name:
            raise ValueError(f"Channel {ch} not found in {record_path}")

    indices = [record.sig_name.index(ch) for ch in channel_names]

    if record.p_signal is None:
        raise ValueError(f"No p_signal in record: {record_path}")
    signals = np.asarray(record.p_signal[:, indices], dtype=np.float64)

    if signals.size == 0:
        raise ValueError(f"No signal data for record {record_path}")

    fs = float(record.fs)
    return signals, fs
