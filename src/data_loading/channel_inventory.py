"""
Orchestrate channel consistency analysis across all patients.

Scans patient directories, reads one .hea per patient, collects channel
lists (after non-EEG filter), computes channel frequency and the intersection
of EEG channels across patients, and writes outputs to ANALYSIS_OUTPUT_PATH.
"""

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .channel_filter import filter_eeg_channels
from .hea_parsing import get_channel_names_from_hea


def _find_first_hea(patient_dir: str) -> Optional[str]:
    """Return path to the first .hea file in patient_dir, or None."""
    try:
        entries = os.listdir(patient_dir)
    except OSError:
        return None
    hea_files = sorted(f for f in entries if f.lower().endswith(".hea"))
    if not hea_files:
        return None
    return os.path.join(patient_dir, hea_files[0])


def run_channel_inventory(
    patient_ids: List[str],
    eeg_raw_root: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Run full channel inventory: per-patient channel lists, frequency stats,
    and common EEG channel set. Writes channel_inventory.csv,
    channel_frequency.csv, and common_eeg_channels.json to output_dir.

    Args:
        patient_ids: List of patient ID strings (e.g. from load_patient_ids).
                     Should be four-digit zero-padded (e.g. "0284"); load_patient_ids
                     returns IDs already normalized.
        eeg_raw_root: Root directory containing one subdir per patient_id
                      with .hea/.mat files inside.
        output_dir: Directory for output artifacts (e.g. ANALYSIS_OUTPUT_PATH).

    Returns:
        Summary dict with keys: n_patients_processed, n_patients_skipped,
        n_common_channels, common_channel_list.

    Side effects:
        Writes channel_inventory.csv, channel_frequency.csv,
        common_eeg_channels.json under output_dir. Creates output_dir
        if it does not exist.
    """
    os.makedirs(output_dir, exist_ok=True)

    per_patient_channels: List[Tuple[str, List[str]]] = []
    skipped: List[str] = []
    frequency_counter: Counter[str] = Counter()

    for patient_id in patient_ids:
        # patient_id is expected to be four-digit zero-padded (e.g. "0284")
        patient_dir = os.path.join(eeg_raw_root, patient_id.strip())
        hea_path = _find_first_hea(patient_dir)
        if hea_path is None:
            skipped.append(patient_id)
            continue
        try:
            raw_channels = get_channel_names_from_hea(hea_path)
        except Exception:
            skipped.append(patient_id)
            continue
        eeg_channels = filter_eeg_channels(raw_channels)
        per_patient_channels.append((patient_id, eeg_channels))
        for ch in eeg_channels:
            frequency_counter[ch] += 1

    # Intersection: channels present in every processed patient
    if not per_patient_channels:
        common_channels: List[str] = []
    else:
        common_set = set(per_patient_channels[0][1])
        for _, ch_list in per_patient_channels[1:]:
            common_set &= set(ch_list)
        common_channels = sorted(common_set)

    # channel_inventory.csv: patient_id, channels (comma-separated)
    inventory_path = os.path.join(output_dir, "channel_inventory.csv")
    with open(inventory_path, "w", newline="", encoding="utf-8") as f:
        f.write("patient_id,channels\n")
        for pid, ch_list in per_patient_channels:
            channels_str = ",".join(ch_list)
            f.write(f'"{pid}","{channels_str}"\n')

    # channel_frequency.csv: channel, count
    frequency_path = os.path.join(output_dir, "channel_frequency.csv")
    with open(frequency_path, "w", newline="", encoding="utf-8") as f:
        f.write("channel,count\n")
        for ch, count in frequency_counter.most_common():
            f.write(f'"{ch}",{count}\n')

    # common_eeg_channels.json: list of channel names
    common_path = os.path.join(output_dir, "common_eeg_channels.json")
    with open(common_path, "w", encoding="utf-8") as f:
        json.dump(common_channels, f, indent=2)

    return {
        "n_patients_processed": len(per_patient_channels),
        "n_patients_skipped": len(skipped),
        "n_common_channels": len(common_channels),
        "common_channel_list": common_channels,
        "skipped_patient_ids": skipped,
    }
