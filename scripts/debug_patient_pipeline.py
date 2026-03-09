#!/usr/bin/env python3
"""
Debug the preprocessing pipeline step-by-step for one patient's first segment.

Runs each stage in order and prints shapes so the failure point is visible:
  load -> filter -> average reference -> windowing -> connectivity

Usage:
  python scripts/debug_patient_pipeline.py --patient-id 0284

In Colab: mount drive, clone repo, pip install -r requirements.txt, then run the above.
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
    WINDOW_SECONDS,
)
from src.data_loading.patient_list import _normalize_patient_id
from src.preprocessing.eeg_loader import load_eeg_segment
from src.preprocessing.patient_processor import _list_segment_paths
from src.preprocessing.signal_filter import bandpass_filter
from src.preprocessing.windowing import segment_into_windows_list
from src.connectivity.pearson import compute_connectivity_batch


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pipeline stages for the first segment of one patient and print shapes."
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        required=True,
        help="Patient ID (e.g. 0284).",
    )
    args = parser.parse_args()

    patient_id = _normalize_patient_id(args.patient_id)
    patient_dir = os.path.join(EEG_RAW_ROOT, patient_id)
    segment_paths = _list_segment_paths(patient_dir, max_segments=1)
    if not segment_paths:
        print(f"No segments found in {patient_dir}")
        return 1

    record_path = segment_paths[0]
    print(f"Patient: {patient_id}")
    print(f"First segment: {record_path}")
    print("")

    if not os.path.isfile(COMMON_CHANNELS_PATH):
        print(f"Error: {COMMON_CHANNELS_PATH} not found")
        return 1
    with open(COMMON_CHANNELS_PATH, "r", encoding="utf-8") as f:
        common_channels = json.load(f)
    print(f"Common channels ({len(common_channels)}): {common_channels}")
    print("")

    # 1. Load
    print("--- Step 1: Load ---")
    try:
        data, fs = load_eeg_segment(record_path, common_channels)
    except Exception as e:
        print(f"LOAD FAILED: {e!r}")
        return 1
    print(f"Loaded signal shape: {data.shape}")
    print(f"Sampling rate fs: {fs}")
    n_samples, n_channels = data.shape[0], data.shape[1]
    if data.shape[0] == len(common_channels) and data.shape[1] != len(common_channels):
        print("WARNING: shape looks like (n_channels, n_samples); pipeline expects (n_samples, n_channels)")
    print("")

    # 2. Filter
    print("--- Step 2: Bandpass filter ---")
    try:
        filtered = bandpass_filter(data, fs, low_hz=BANDPASS_LOW, high_hz=BANDPASS_HIGH)
    except Exception as e:
        print(f"FILTER FAILED: {e!r}")
        return 1
    print(f"After filter: {filtered.shape}")
    print("")

    # 3. Average reference
    print("--- Step 3: Average reference ---")
    referenced = filtered - filtered.mean(axis=1, keepdims=True)
    print(f"After referencing: {referenced.shape}")
    print("")

    # 4. Windowing
    print("--- Step 4: Windowing ---")
    windows_list = segment_into_windows_list(referenced, fs, WINDOW_SECONDS)
    n_windows = len(windows_list)
    print(f"Windows created: {n_windows}")
    if n_windows == 0:
        print("WARNING: Zero windows! Check signal length and fs.")
        window_samples = int(round(WINDOW_SECONDS * fs))
        print(f"  window_samples = {WINDOW_SECONDS} * {fs} = {window_samples}")
        print(f"  n_full_windows = n_samples // window_samples = {referenced.shape[0]} // {window_samples} = {referenced.shape[0] // window_samples}")
        return 1
    window_shape = windows_list[0].shape
    print(f"Window shape: {window_shape}")
    print("")

    # 5. Connectivity
    print("--- Step 5: Connectivity ---")
    windows_array = np.stack(windows_list, axis=0)
    print(f"Stacked windows shape: {windows_array.shape}")
    try:
        conn = compute_connectivity_batch(windows_array)
    except Exception as e:
        print(f"CONNECTIVITY FAILED: {e!r}")
        return 1
    print(f"Connectivity matrices: {conn.shape}")
    print("")

    print("--- All steps completed successfully ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
