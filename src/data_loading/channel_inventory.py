"""
Orchestrate channel consistency analysis across all patients.

Scans patient directories, reads one .hea per patient, collects channel
lists (after non-EEG filter), computes channel frequency and the intersection
of EEG channels across patients, and writes outputs to ANALYSIS_OUTPUT_PATH.
"""

from typing import Any, Dict, List


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
        patient_ids: List of patient ID strings (e.g. from patient_list.load).
        eeg_raw_root: Root directory containing one subdir per patient_id
                      with .hea/.mat files inside.
        output_dir: Directory for output artifacts (e.g. ANALYSIS_OUTPUT_PATH).

    Returns:
        Summary dict with keys such as: n_patients_processed, n_patients_skipped,
        n_common_channels, common_channel_list. Exact keys are
        implementation-defined.

    Side effects:
        Writes channel_inventory.csv, channel_frequency.csv,
        common_eeg_channels.json under output_dir. Creates output_dir
        if it does not exist.
    """
    raise NotImplementedError("Channel inventory orchestration not yet implemented.")
