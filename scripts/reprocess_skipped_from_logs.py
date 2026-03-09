#!/usr/bin/env python3
"""
One-time recovery script: extract patients with skipped segments from Colab logs.

Run this in a Google Colab cell after pasting the full preprocessing log output.
It parses [segment_debug] SKIP segment=... lines, collects affected patient IDs,
writes patient_split_reprocess.csv, and optionally removes existing connectivity
outputs for those patients so that run_preprocessing recomputes only them.

Usage:
  1. Run the preprocessing notebook and copy the full output from Colab.
  2. Paste that output into the LOGS string below (see the marker).
  3. Run this script in a Colab cell.
  4. Run the printed command to reprocess only the affected patients.
"""

import os
import re
import shutil

# ---------------------------------------------------------------------------
# CONFIGURATION (adjust for your Colab setup)
# ---------------------------------------------------------------------------

# Folder where batch/split CSVs live (same as run_preprocessing --patient-split base)
BATCH_FOLDER = os.environ.get(
    "BATCH_FOLDER",
    "/content/drive/MyDrive/icare_project/batches",
)

# Directory containing patient_id_connectivity.npy and patient_id/ subdirs
WINDOWS_OUTPUT_DIR = os.environ.get(
    "WINDOWS_OUTPUT_DIR",
    "/content/drive/MyDrive/icare_project/intermediate/windows",
)

# Set to True to delete existing connectivity outputs for affected patients
# so that run_preprocessing will fully reprocess them (not just incremental).
# Set to False to only generate the CSV (e.g. for incremental reprocessing).
REMOVE_EXISTING_OUTPUTS = True

# ---------------------------------------------------------------------------
# PASTE YOUR FULL COLAB OUTPUT LOGS HERE
# ---------------------------------------------------------------------------
# Replace the triple-quoted block below with your pasted log text.
# Look for lines like:
#   [segment_debug] SKIP segment=/content/.../0532/... | reason=fs_mismatch
#   [segment_debug] SKIP segment=/content/.../0505/... | reason=windowing_returned_0_windows
# ---------------------------------------------------------------------------

LOGS = """
PASTE YOUR FULL COLAB OUTPUT LOGS HERE

Copy the entire output from the Colab cell that ran:
  !python scripts/run_preprocessing.py --patient-split patient_split_1.csv

Paste it between the triple quotes above, replacing this placeholder text.
"""

# ---------------------------------------------------------------------------
# Parser and recovery logic
# ---------------------------------------------------------------------------

SKIP_MARKER = "[segment_debug] SKIP segment="
# Current pipeline format: [segment_debug] patient_id=0532 ... | reason=...
PATIENT_ID_RE = re.compile(r"patient_id=([^\s\|]+)")


def _normalize_patient_id(value: str) -> str:
    """Normalize to 4-digit zero-padded string (e.g. 532 -> 0532)."""
    s = str(value).strip()
    if not s:
        return s
    try:
        return str(int(s)).zfill(4)
    except ValueError:
        return s.zfill(4) if len(s) <= 4 else s


def extract_patient_id_from_segment_path(segment_path: str) -> str | None:
    """
    Extract patient ID from a segment path.
    Assumes path like /content/.../0532/segment_001.edf -> 0532 (parent dir of file).
    """
    path = segment_path.replace("\\", "/").strip()
    parts = [p for p in path.split("/") if p]
    if len(parts) >= 2:
        return _normalize_patient_id(parts[-2])
    if len(parts) == 1:
        return _normalize_patient_id(parts[0])
    return None


def parse_skip_lines(log_text: str) -> list[tuple[str, str]]:
    """
    Parse log text and return list of (segment_path_or_id, reason) for each skip line.
    Supports two formats:
      1. [segment_debug] SKIP segment=/path/to/0532/... | reason=fs_mismatch
      2. [segment_debug] patient_id=0532 seg_idx=... path=... | reason=...
    """
    results = []
    lines = log_text.splitlines()
    for i, line in enumerate(lines):
        if "[segment_debug]" not in line:
            continue
        # Format 1: SKIP segment=<path>
        if SKIP_MARKER in line:
            rest = line.split(SKIP_MARKER, 1)[-1]
            if " |" in rest:
                segment_path, _ = rest.split(" |", 1)
            else:
                segment_path = rest
            segment_path = segment_path.strip()
            reason = "unknown"
            if "reason=" in line:
                match = re.search(r"reason=(\w+)", line)
                if match:
                    reason = match.group(1)
            results.append((segment_path, reason))
            continue
        # Format 2: patient_id=XXXX ... reason= (current pipeline; reason may be on next line)
        if "reason=" in line:
            match = PATIENT_ID_RE.search(line)
            if match:
                pid = _normalize_patient_id(match.group(1))
                m2 = re.search(r"reason=(\w+)", line)
                results.append((pid, m2.group(1) if m2 else "from_reason_line"))
        else:
            # Multi-line: this line is [segment_debug] ...; next line has reason=
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if "reason=" in next_line and "[segment_debug]" not in next_line:
                match = PATIENT_ID_RE.search(line)
                if match:
                    pid = _normalize_patient_id(match.group(1))
                    m2 = re.search(r"reason=(\w+)", next_line)
                    results.append((pid, m2.group(1) if m2 else "unknown"))
    return results


def get_affected_patient_ids(log_text: str) -> list[str]:
    """Return unique sorted list of patient IDs from SKIP segment lines."""
    skip_entries = parse_skip_lines(log_text)
    seen: set[str] = set()
    for path_or_id, _ in skip_entries:
        # path_or_id is either a path (contains /) or already a patient ID
        if "/" in path_or_id or "\\" in path_or_id:
            pid = extract_patient_id_from_segment_path(path_or_id)
        else:
            pid = _normalize_patient_id(path_or_id) if path_or_id else None
        if pid:
            seen.add(pid)
    return sorted(seen)


def remove_existing_outputs(patient_ids: list[str], output_dir: str) -> list[str]:
    """
    Remove patient_id_connectivity.npy and patient_id/ for each patient.
    Returns list of patient IDs for which something was removed.
    """
    removed: list[str] = []
    for patient_id in patient_ids:
        final_file = os.path.join(output_dir, f"{patient_id}_connectivity.npy")
        segment_dir = os.path.join(output_dir, patient_id)
        did_remove = False
        if os.path.isfile(final_file):
            try:
                os.remove(final_file)
                did_remove = True
            except OSError as e:
                print(f"  Warning: could not remove {final_file}: {e}")
        if os.path.isdir(segment_dir):
            try:
                shutil.rmtree(segment_dir)
                did_remove = True
            except OSError as e:
                print(f"  Warning: could not remove dir {segment_dir}: {e}")
        if did_remove:
            removed.append(patient_id)
    return removed


def main() -> None:
    log_text = LOGS.strip()
    if not log_text or "PASTE YOUR FULL COLAB OUTPUT LOGS HERE" in log_text:
        print("ERROR: You have not pasted your Colab output logs.")
        print("Replace the LOGS string above with the full output from your preprocessing run.")
        return

    skip_entries = parse_skip_lines(log_text)
    patient_ids = get_affected_patient_ids(log_text)

    print("=" * 60)
    print("Recovery script summary")
    print("=" * 60)
    print(f"  Skip lines found:     {len(skip_entries)}")
    print(f"  Unique patients:      {len(patient_ids)}")
    if patient_ids:
        print(f"  Patient IDs:          {patient_ids}")
    else:
        print("  No affected patient IDs found. Check that log lines contain:")
        print('    [segment_debug] SKIP segment=<path> | reason=...')
        print("=" * 60)
        return

    os.makedirs(BATCH_FOLDER, exist_ok=True)
    csv_path = os.path.join(BATCH_FOLDER, "patient_split_reprocess.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("patient_id\n")
        for pid in patient_ids:
            f.write(f"{pid}\n")
    print(f"  Generated CSV:       {csv_path}")
    print("=" * 60)

    if REMOVE_EXISTING_OUTPUTS and patient_ids:
        if not os.path.isdir(WINDOWS_OUTPUT_DIR):
            print(f"  Output dir not found (skipping removal): {WINDOWS_OUTPUT_DIR}")
        else:
            removed = remove_existing_outputs(patient_ids, WINDOWS_OUTPUT_DIR)
            if removed:
                print(f"  Removed existing outputs for: {removed}")
            else:
                print("  No existing outputs to remove for these patients.")
    elif not REMOVE_EXISTING_OUTPUTS:
        print("  REMOVE_EXISTING_OUTPUTS=False: existing files left as-is.")

    print()
    print("Next step: run the following command in Colab to reprocess only these patients:")
    print()
    print("  !python scripts/run_preprocessing.py --patient-split patient_split_reprocess.csv")
    print()


if __name__ == "__main__":
    main()
