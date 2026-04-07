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
import pandas as pd

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
    PREPROCESS_ENABLE_NOTCH,
    PREPROCESS_MIN_SALVAGE_DURATION_SEC,
    PREPROCESS_NOTCH_FREQS,
    PREPROCESS_NOTCH_Q,
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
        "--audit-csv",
        type=str,
        default=None,
        help="Optional path to save per-patient preprocessing audit CSV.",
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
    parser.add_argument(
        "--enable-notch",
        action="store_true",
        help="Enable utility-frequency notch filtering.",
    )
    parser.add_argument(
        "--disable-notch",
        action="store_true",
        help="Force-disable utility-frequency notch filtering.",
    )
    parser.add_argument(
        "--notch-freqs",
        type=str,
        default=None,
        help=f"Comma-separated notch frequencies in Hz (default: {PREPROCESS_NOTCH_FREQS}).",
    )
    parser.add_argument(
        "--notch-q",
        type=float,
        default=None,
        help=f"Notch Q factor (default: {PREPROCESS_NOTCH_Q}).",
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
    if args.disable_notch:
        enable_notch = False
    elif args.enable_notch:
        enable_notch = True
    else:
        enable_notch = PREPROCESS_ENABLE_NOTCH
    notch_q = args.notch_q if args.notch_q is not None else PREPROCESS_NOTCH_Q
    notch_freq_str = args.notch_freqs if args.notch_freqs is not None else PREPROCESS_NOTCH_FREQS
    notch_freqs = []
    for tok in str(notch_freq_str).split(","):
        t = tok.strip()
        if not t:
            continue
        try:
            notch_freqs.append(float(t))
        except ValueError:
            print(f"Warning: invalid notch frequency token ignored: {t!r}")

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
    print(f"Notch enabled: {enable_notch}")
    if enable_notch:
        print(f"Notch freqs (Hz): {notch_freqs}")
        print(f"Notch Q: {notch_q}")

    audit_rows = []
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
            enable_notch=enable_notch,
            notch_freqs=notch_freqs,
            notch_q=notch_q,
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

        fr = result.get("failure_reasons", {}) or {}
        data_source = "fresh"
        if bool(result.get("skipped", False)) and result.get("reason") == "output_exists":
            data_source = "cached"
        audit_rows.append(
            {
                "patient_id": patient_id,
                "data_source": data_source,
                "processed": bool(result.get("processed", False)),
                "skipped": bool(result.get("skipped", False)),
                "reason": result.get("reason"),
                "n_segments_total": int(result.get("n_segments_total", 0) or 0),
                "n_segments_processed": int(result.get("n_segments_processed", 0) or 0),
                "n_segments_failed": int(result.get("n_segments_failed", 0) or 0),
                "n_windows_pre_qc": (
                    int(result.get("n_windows_pre_qc", 0) or 0) if data_source == "fresh" else None
                ),
                "n_windows_after_qc": (
                    int(result.get("n_windows", 0) or 0) if data_source == "fresh" else None
                ),
                "n_windows_rejected_qc": (
                    int(result.get("n_windows_rejected_qc", 0) or 0) if data_source == "fresh" else None
                ),
                "retention_ratio": (
                    float(result.get("n_windows", 0) or 0)
                    / max(1, int(result.get("n_windows_pre_qc", 0) or 0))
                ) if data_source == "fresh" else None,
                "n_connectivity_matrices": (
                    int(result.get("n_connectivity_matrices", 0) or 0) if data_source == "fresh" else None
                ),
                "fail_load_exception": int(fr.get("load_exception", 0)),
                "fail_missing_channels": int(fr.get("missing_channels", 0)),
                "fail_resample_failure": int(fr.get("resample_failure", 0)),
                "fail_filter_failure": int(fr.get("filter_failure", 0)),
                "fail_windowing_returned_0_windows": int(fr.get("windowing_returned_0_windows", 0)),
                "fail_all_windows_rejected_by_qc": int(fr.get("all_windows_rejected_by_qc", 0)),
                "fail_connectivity_failure": int(fr.get("connectivity_failure", 0)),
                "fail_validation_failure": int(fr.get("validation_failure", 0)),
            }
        )

    print(f"Done. Processed: {n_processed}, Skipped: {n_skipped}, Failed: {n_failed}")
    if args.audit_csv:
        audit_path = args.audit_csv
    else:
        split_name = os.path.splitext(os.path.basename(split_path))[0]
        audit_path = os.path.join(output_dir, f"preprocessing_audit_{split_name}.csv")
    os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    print(f"Audit CSV: {audit_path}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
