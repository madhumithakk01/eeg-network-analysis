"""
Filter channel lists to retain only valid EEG channels.

Excludes non-EEG channels (e.g. EKG, ECG, EMG, RESP, SpO2, Pulse, Temp)
so that connectivity analysis uses only brain signals. Supports either
a blacklist of non-EEG patterns or a whitelist of known EEG electrode names.
"""

from typing import List


def filter_eeg_channels(channel_names: List[str]) -> List[str]:
    """
    Return only channel names that are considered valid EEG electrodes.

    Non-EEG channels (physiological sensors) are excluded so they do not
    corrupt brain connectivity matrices. Exclusion is by name pattern
    (e.g. EKG, ECG, EMG, RESP, SpO2, Pulse, Temp) or by whitelist of
    known EEG labels (e.g. 10-20 system).

    Args:
        channel_names: Full list of channel names from a WFDB header.

    Returns:
        Subset of channel_names that are valid EEG. Order may be preserved
        or sorted; implementation-defined.
    """
    raise NotImplementedError("Channel filter not yet implemented.")
