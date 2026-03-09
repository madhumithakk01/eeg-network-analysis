#!/usr/bin/env python3
"""
Inspect a connectivity .npy file: shape, dtype, first matrix, diagonal, min/max.

Use after test_single_patient.py to visually verify output correctness.
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load a connectivity .npy file and print shape, dtype, and sample values."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to patient_id_connectivity.npy file.",
    )
    args = parser.parse_args()

    path = os.path.abspath(args.path)
    if not os.path.isfile(path):
        print(f"Error: File not found: {path}")
        return 1

    arr = np.load(path)
    print(f"Path: {path}")
    print(f"Shape: {arr.shape}")
    print(f"dtype: {arr.dtype}")
    print("")

    if arr.ndim != 3:
        print("Expected 3D array (n_windows, n_channels, n_channels).")
        return 0

    n_w, n_ch, _ = arr.shape
    print(f"First matrix (window 0), shape ({n_ch}, {n_ch}):")
    print(arr[0])
    print("")
    print("Diagonal of first matrix (should be ~1):")
    print(np.diag(arr[0]))
    print("")
    print("Min connectivity value:", float(np.min(arr)))
    print("Max connectivity value:", float(np.max(arr)))
    print("Any NaN:", np.any(np.isnan(arr)))
    print("Symmetric (first matrix):", np.allclose(arr[0], arr[0].T))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
