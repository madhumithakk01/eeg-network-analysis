"""
Filter channel lists to retain only valid EEG channels.

Uses the official I-CARE EEG electrode set as a whitelist. Only channels
in this set are kept; all others (ECG, REF, SpO2, EMG, etc.) are excluded
so that connectivity analysis uses only brain signals.
"""

from typing import List

# Official EEG electrode set from I-CARE dataset documentation.
# Only these channels are valid for brain connectivity analysis.
VALID_EEG_CHANNELS = (
    "Fp1",
    "Fp2",
    "F7",
    "F8",
    "F3",
    "F4",
    "T3",
    "T4",
    "C3",
    "C4",
    "T5",
    "T6",
    "P3",
    "P4",
    "O1",
    "O2",
    "Fz",
    "Cz",
    "Pz",
    "Fpz",
    "Oz",
    "F9",
)

# Normalized (uppercase, stripped) set for fast lookup
_VALID_EEG_NORMALIZED = frozenset(c.strip().upper() for c in VALID_EEG_CHANNELS)

# Map normalized name -> canonical spelling for consistent output
_NORMALIZED_TO_CANONICAL = {c.strip().upper(): c for c in VALID_EEG_CHANNELS}


def filter_eeg_channels(channel_names: List[str]) -> List[str]:
    """
    Return only channel names that belong to the valid EEG whitelist.

    Channel names are normalized (strip whitespace, uppercase) for matching.
    Output uses canonical spelling from the I-CARE electrode set.
    Non-EEG channels (ECG, REF, SpO2, EMG, etc.) are excluded.

    Args:
        channel_names: Full list of channel names from a WFDB header.

    Returns:
        Subset of channel_names that are in VALID_EEG_CHANNELS, in order
        of first appearance, with canonical spelling.
    """
    result: List[str] = []
    seen = set()
    for name in channel_names:
        key = name.strip().upper()
        if key in _VALID_EEG_NORMALIZED and key not in seen:
            seen.add(key)
            result.append(_NORMALIZED_TO_CANONICAL[key])
    return result
