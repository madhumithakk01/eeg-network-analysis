#!/usr/bin/env python3
"""
Entry point for EEG channel consistency analysis.

Loads the canonical patient list, scans WFDB .hea headers per patient,
filters non-EEG channels (I-CARE whitelist), computes channel frequency
and the intersection of EEG channels across all patients, and writes:

  - channel_inventory.csv   (per-patient channel list)
  - channel_frequency.csv   (channel -> patient count)
  - common_eeg_channels.json (intersection: channels in every patient)

Outputs are written to ANALYSIS_OUTPUT_PATH (configs.config).

CLI arguments:
  --patient-list  Path to patient ID CSV (default: BATCH_FOLDER + CANONICAL_PATIENT_LIST_FILENAME)
  --output-dir    Output directory (default: ANALYSIS_OUTPUT_PATH)
"""

import argparse
import os
import sys

# Ensure project root is on path when running as script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from configs.config import (
    ANALYSIS_OUTPUT_PATH,
    BATCH_FOLDER,
    CANONICAL_PATIENT_LIST_FILENAME,
    EEG_RAW_ROOT,
)
from src.data_loading.channel_inventory import run_channel_inventory
from src.data_loading.patient_list import load_patient_ids


def main() -> int:
    """Orchestrate channel inventory and write artifacts to analysis/."""
    parser = argparse.ArgumentParser(
        description="Run EEG channel consistency analysis (inventory, frequency, common channels)."
    )
    parser.add_argument(
        "--patient-list",
        type=str,
        default=None,
        help="Path to patient ID CSV with 'patient_id' column (default: BATCH_FOLDER + CANONICAL_PATIENT_LIST_FILENAME)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for CSVs and JSON (default: ANALYSIS_OUTPUT_PATH)",
    )
    args = parser.parse_args()

    patient_list_path = args.patient_list
    if patient_list_path is None:
        patient_list_path = os.path.join(BATCH_FOLDER, CANONICAL_PATIENT_LIST_FILENAME)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ANALYSIS_OUTPUT_PATH

    if not os.path.isfile(patient_list_path):
        print(f"Error: Patient list not found: {patient_list_path}")
        return 1

    patient_ids = load_patient_ids(patient_list_path)
    print(f"Loaded {len(patient_ids)} patient IDs from {patient_list_path}")
    print(f"EEG root: {EEG_RAW_ROOT}")
    print(f"Output dir: {output_dir}")

    summary = run_channel_inventory(
        patient_ids=patient_ids,
        eeg_raw_root=EEG_RAW_ROOT,
        output_dir=output_dir,
    )

    print(f"Processed: {summary['n_patients_processed']}")
    print(f"Skipped: {summary['n_patients_skipped']}")
    print(f"Common EEG channels: {summary['n_common_channels']}")
    if summary.get("skipped_patient_ids"):
        print(f"Skipped IDs (first 10): {summary['skipped_patient_ids'][:10]}")
    print(f"Wrote: {output_dir}/channel_inventory.csv")
    print(f"Wrote: {output_dir}/channel_frequency.csv")
    print(f"Wrote: {output_dir}/common_eeg_channels.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
