#!/usr/bin/env python3
"""
Test preprocessing on a single patient to validate the pipeline before full batch.

SAFE TEST WORKFLOW (Steps 1-6 before parallel batch):

  1. Run preprocessing for one patient (this script with validate_connectivity=True):
       python scripts/test_single_patient.py --patient-id 0284

  2. Inspect the connectivity output:
       python scripts/inspect_connectivity.py --path /path/to/intermediate/windows/0284_connectivity.npy

  3. Confirm:
       - shape is (n_windows, 19, 19)
       - dtype is float32
       - diagonal ~1, symmetric, values in [-1, 1], no NaNs

  4. Then run the full batch (e.g. parallel workers):
       python scripts/run_preprocessing.py --patient-split patient_split_1.csv
       python scripts/run_preprocessing.py --patient-split patient_split_2.csv
       python scripts/run_preprocessing.py --patient-split patient_split_3.csv

Example (Colab/Drive paths):
  python scripts/test_single_patient.py --patient-id 0284
  python scripts/inspect_connectivity.py --path /content/drive/MyDrive/icare_project/intermediate/windows/0284_connectivity.npy
"""

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from configs.config import (
    BANDPASS_HIGH,
    BANDPASS_LOW,
    COMMON_CHANNELS_PATH,
    EEG_RAW_ROOT,
    MAX_EEG_SEGMENTS,
    TEMP_DIR,
    WINDOWS_OUTPUT_DIR,
    WINDOW_SECONDS,
)
from src.data_loading.patient_list import _normalize_patient_id
from src.preprocessing.patient_processor import process_patient


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run preprocessing on a single patient with connectivity validation."
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        required=True,
        help="Patient ID (e.g. 0284 or 284); will be normalized to 4 digits.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: WINDOWS_OUTPUT_DIR).",
    )
    args = parser.parse_args()

    patient_id = _normalize_patient_id(args.patient_id)
    output_dir = args.output_dir or WINDOWS_OUTPUT_DIR

    if not os.path.isfile(COMMON_CHANNELS_PATH):
        print(f"Error: Common channels file not found: {COMMON_CHANNELS_PATH}")
        return 1
    with open(COMMON_CHANNELS_PATH, "r", encoding="utf-8") as f:
        common_channels = json.load(f)
    if not common_channels:
        print("Error: common_eeg_channels.json is empty.")
        return 1

    print(f"Patient ID: {patient_id}")
    print(f"Common channels: {len(common_channels)}")
    print(f"EEG root: {EEG_RAW_ROOT}")
    print(f"Output dir: {output_dir}")
    print("Running with validate_connectivity=True ...")

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
        validate_connectivity=True,
    )

    if result["skipped"] and result.get("reason") == "output_exists":
        print("Output already exists; loading for inspection.")
        path = result["output_path"]
        if path and os.path.isfile(path):
            arr = np.load(path)
            print(f"  shape: {arr.shape}")
            print(f"  dtype: {arr.dtype}")
            print(f"  connectivity matrices: {arr.shape[0]}")
            print(f"  output path: {path}")
        return 0
    if result["skipped"]:
        print(f"Skipped: {result.get('reason', 'unknown')}")
        return 1
    if not result["processed"]:
        print(f"Failed: {result.get('error', result.get('reason', 'unknown'))}")
        return 1

    path = result["output_path"]
    arr = np.load(path)
    print("")
    print("--- Connectivity output summary ---")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    print(f"  connectivity matrices saved: {result['n_connectivity_matrices']}")
    print(f"  output path: {path}")
    print("")
    print("Next: run inspect_connectivity.py on this file to verify values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
