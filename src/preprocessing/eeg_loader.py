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

    Uses directory + basename for WFDB: rdrecord(record_name=basename, pn_dir=directory).
    Loads physical units (p_signal), then selects only requested channels by name.
    Fails with ValueError if any required channel is missing (no silent skip).

    Args:
        record_path: Path to the record (with or without .hea/.mat extension).
                     Example: /path/to/0284_001_004_EEG or .../0284_001_004_EEG.hea
        channel_names: List of channel names to load (e.g. from common_eeg_channels.json).
                       Order in the returned array matches this list.

    Returns:
        signals: numpy array of shape (n_samples, n_channels) in physical units.
                 Channels in the order of channel_names.
        fs: Sampling frequency in Hz.

    Raises:
        FileNotFoundError: If the record files are missing.
        ValueError: If any requested channel is missing in the record (message includes record_path).
    """
    path = os.path.abspath(record_path)
    if path.lower().endswith(".hea"):
        path = path[:-4]
    if path.lower().endswith(".mat"):
        path = path[:-4]

    dir_name = os.path.dirname(path)
    base_name = os.path.basename(path)
    if not base_name:
        raise ValueError(f"Invalid record_path (no basename): {record_path}")

    if dir_name:
        record = wfdb.rdrecord(record_name=base_name, pn_dir=dir_name)
    else:
        record = wfdb.rdrecord(record_name=base_name)

    if record.sig_name is None:
        raise ValueError(f"No channel names in record: {record_path}")

    for ch in channel_names:
        if ch not in record.sig_name:
            raise ValueError(f"Missing required channel '{ch}' in {record_path}")

    channel_indices = [record.sig_name.index(ch) for ch in channel_names]

    if record.p_signal is not None:
        signals = np.asarray(record.p_signal, dtype=np.float64)
    else:
        signals = np.asarray(record.d_signal, dtype=np.float64)

    if signals.size == 0:
        raise ValueError(f"No signal data for record {record_path}")

    signals = signals[:, channel_indices]
    fs = float(record.fs)
    return signals, fs
