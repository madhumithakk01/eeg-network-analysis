#!/usr/bin/env python3
"""
Entry point for EEG preprocessing (filter, average reference, window, connectivity).

Loads a patient split file, common channel list, and processes each patient:
load segments (first 48), bandpass filter, average reference, segment into 30s
windows, compute Pearson connectivity matrices, save to intermediate/windows/
as patient_id_connectivity.npy (shape n_windows × 19 × 19, float32).

Fault-tolerant: skips patients whose output already exists. Writes to temp
then moves to output dir. Supports parallel workers via different split files:

  python scripts/run_preprocessing.py --patient-split patient_split_1.csv
  python scripts/run_preprocessing.py --patient-split patient_split_2.csv
  python scripts/run_preprocessing.py --patient-split patient_split_3.csv
"""

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    BATCH_FOLDER,
    COMMON_CHANNELS_PATH,
    EEG_RAW_ROOT,
    MAX_EEG_SEGMENTS,
    TEMP_DIR,
    WINDOWS_OUTPUT_DIR,
    WINDOW_SECONDS,
)
from src.data_loading.patient_list import load_patient_ids
from src.preprocessing.patient_processor import process_patient


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run EEG preprocessing (filter, window) for a patient split."
    )
    parser.add_argument(
        "--patient-split",
        type=str,
        default=None,
        help="Path to patient split CSV (e.g. patient_split_1.csv). If relative, resolved against BATCH_FOLDER.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for connectivity .npy files (default: WINDOWS_OUTPUT_DIR).",
    )
    args = parser.parse_args()

    if args.patient_split is None:
        print("Error: --patient-split is required (e.g. patient_split_1.csv).")
        return 1

    split_path = args.patient_split
    if not os.path.isabs(split_path):
        split_path = os.path.join(BATCH_FOLDER, split_path)
    if not os.path.isfile(split_path):
        print(f"Error: Patient split file not found: {split_path}")
        return 1

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = WINDOWS_OUTPUT_DIR

    if not os.path.isfile(COMMON_CHANNELS_PATH):
        print(f"Error: Common channels file not found: {COMMON_CHANNELS_PATH}")
        return 1
    with open(COMMON_CHANNELS_PATH, "r", encoding="utf-8") as f:
        common_channels = json.load(f)
    if not common_channels:
        print("Error: common_eeg_channels.json is empty.")
        return 1

    patient_ids = load_patient_ids(split_path)
    print(f"Loaded {len(patient_ids)} patients from {split_path}")
    print(f"Common channels: {len(common_channels)}")
    print(f"EEG root: {EEG_RAW_ROOT}")
    print(f"Output dir: {output_dir}")

    n_processed = 0
    n_skipped = 0
    n_failed = 0
    for i, patient_id in enumerate(patient_ids):
        result = process_patient(
            patient_id=patient_id,
            eeg_raw_root=EEG_RAW_ROOT,
            output_dir=output_dir,
            common_channel_names=common_channels,
            window_seconds=WINDOW_SECONDS,
            bandpass_low=BANDPASS_LOW,
            bandpass_high=BANDPASS_HIGH,
            max_segments=MAX_EEG_SEGMENTS,
            temp_dir=TEMP_DIR,
        )
        if result["processed"]:
            n_processed += 1
            n_win = result["n_windows"]
            n_mat = result["n_connectivity_matrices"]
            print(f"  [{i+1}/{len(patient_ids)}] {patient_id}: {n_win} windows processed, {n_mat} connectivity matrices saved -> {result['output_path']}")
        elif result["skipped"]:
            n_skipped += 1
        else:
            n_failed += 1
            print(f"  [{i+1}/{len(patient_ids)}] {patient_id}: failed - {result.get('error', result.get('reason', 'unknown'))}")

    print(f"Done. Processed: {n_processed}, Skipped: {n_skipped}, Failed: {n_failed}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
