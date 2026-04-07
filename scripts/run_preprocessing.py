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
    PREPROCESS_ENABLE_WINDOW_QC,
    PREPROCESS_QC_FLAT_STD_MIN_ABS,
    PREPROCESS_QC_FLAT_STD_MIN_RATIO,
    PREPROCESS_QC_HIGH_AMP_ROBUST_Z,
    PREPROCESS_QC_MAINS_BAND_HZ,
    PREPROCESS_QC_MAINS_HZ,
    PREPROCESS_QC_MAX_FLAT_CHANNEL_FRAC,
    PREPROCESS_QC_MAX_HIGH_AMP_FRAC,
    PREPROCESS_QC_MAX_LOW_UNIQUE_CHANNEL_FRAC,
    PREPROCESS_QC_MAX_MAINS_RATIO,
    PREPROCESS_QC_MIN_UNIQUE_VALUE_RATIO,
    PREPROCESS_ENABLE_SHORT_SEGMENT_SALVAGE,
    PREPROCESS_ENABLE_RESAMPLING,
    PREPROCESS_MIN_SALVAGE_DURATION_SEC,
    PREPROCESS_TARGET_FS,
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
    parser.add_argument(
        "--enable-window-qc",
        action="store_true",
        help="Enable window-level artifact QC gate before connectivity computation.",
    )
    parser.add_argument(
        "--disable-window-qc",
        action="store_true",
        help="Force-disable window-level artifact QC gate.",
    )
    parser.add_argument(
        "--disable-short-segment-salvage",
        action="store_true",
        help="Disable converting short (<window) segments into one salvage window.",
    )
    parser.add_argument(
        "--min-salvage-duration-sec",
        type=float,
        default=None,
        help=f"Minimum segment duration in seconds to allow salvage (default: {PREPROCESS_MIN_SALVAGE_DURATION_SEC}).",
    )
    parser.add_argument(
        "--enable-resampling",
        action="store_true",
        help="Enable sampling-rate harmonization via polyphase resampling to target fs.",
    )
    parser.add_argument(
        "--disable-resampling",
        action="store_true",
        help="Force-disable sampling-rate harmonization.",
    )
    parser.add_argument(
        "--target-fs",
        type=float,
        default=None,
        help=f"Target sampling frequency in Hz when resampling is enabled (default: {PREPROCESS_TARGET_FS}).",
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
    if args.disable_window_qc:
        enable_window_qc = False
    elif args.enable_window_qc:
        enable_window_qc = True
    else:
        enable_window_qc = PREPROCESS_ENABLE_WINDOW_QC

    qc_params = {
        "flat_std_min_abs": PREPROCESS_QC_FLAT_STD_MIN_ABS,
        "flat_std_min_ratio": PREPROCESS_QC_FLAT_STD_MIN_RATIO,
        "max_flat_channel_frac": PREPROCESS_QC_MAX_FLAT_CHANNEL_FRAC,
        "min_unique_value_ratio": PREPROCESS_QC_MIN_UNIQUE_VALUE_RATIO,
        "max_low_unique_channel_frac": PREPROCESS_QC_MAX_LOW_UNIQUE_CHANNEL_FRAC,
        "high_amp_robust_z": PREPROCESS_QC_HIGH_AMP_ROBUST_Z,
        "max_high_amp_frac": PREPROCESS_QC_MAX_HIGH_AMP_FRAC,
        "mains_hz": PREPROCESS_QC_MAINS_HZ,
        "mains_band_hz": PREPROCESS_QC_MAINS_BAND_HZ,
        "max_mains_ratio": PREPROCESS_QC_MAX_MAINS_RATIO,
    }
    if args.disable_short_segment_salvage:
        enable_short_segment_salvage = False
    else:
        enable_short_segment_salvage = PREPROCESS_ENABLE_SHORT_SEGMENT_SALVAGE
    min_salvage_duration_sec = (
        args.min_salvage_duration_sec
        if args.min_salvage_duration_sec is not None
        else PREPROCESS_MIN_SALVAGE_DURATION_SEC
    )
    if args.disable_resampling:
        enable_resampling = False
    elif args.enable_resampling:
        enable_resampling = True
    else:
        enable_resampling = PREPROCESS_ENABLE_RESAMPLING
    target_fs = args.target_fs if args.target_fs is not None else PREPROCESS_TARGET_FS

    print(f"Loaded {len(patient_ids)} patients from {split_path}")
    print(f"Common channels: {len(common_channels)}")
    print(f"EEG root: {EEG_RAW_ROOT}")
    print(f"Output dir: {output_dir}")
    print(f"Window QC enabled: {enable_window_qc}")
    print(f"Short-segment salvage enabled: {enable_short_segment_salvage}")
    if enable_short_segment_salvage:
        print(f"Min salvage duration (sec): {min_salvage_duration_sec}")
    print(f"Resampling enabled: {enable_resampling}")
    if enable_resampling:
        print(f"Target fs (Hz): {target_fs}")

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
            enable_window_qc=enable_window_qc,
            qc_params=qc_params,
            enable_short_segment_salvage=enable_short_segment_salvage,
            min_salvage_duration_sec=min_salvage_duration_sec,
            enable_resampling=enable_resampling,
            target_fs=target_fs,
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
