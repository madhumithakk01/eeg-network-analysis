"""
Load patient IDs from CSV files (canonical list or balanced splits).

Used by channel consistency analysis and by downstream pipelines to determine
which patients to process. Patient IDs are always treated as strings and
normalized to four-digit zero-padded form (e.g. 284 → "0284") to match
EEG directory naming in the I-CARE dataset.
"""

from typing import List

import pandas as pd


def _normalize_patient_id(value: str) -> str:
    """
    Normalize a single patient ID to a four-digit zero-padded string.

    Examples: 284 → "0284", 31 → "0031", 7 → "0007", "0284" → "0284".
    Numeric IDs are parsed and zero-padded; non-numeric strings are
    padded to 4 characters if they are 4 or fewer characters long.
    """
    s = str(value).strip()
    if not s:
        return s
    try:
        return str(int(s)).zfill(4)
    except ValueError:
        return s.zfill(4) if len(s) <= 4 else s


def load_patient_ids(csv_path: str) -> List[str]:
    """
    Load patient IDs from a CSV file with header 'patient_id'.

    All returned IDs are normalized to four-digit zero-padded strings
    (e.g. 284 → "0284") so they match EEG directory names on disk.
    Empty rows are skipped.

    Args:
        csv_path: Path to the CSV (e.g. all_downloaded_patients_294.csv
                  or patient_split_1.csv).

    Returns:
        List of patient ID strings, each four-digit zero-padded.
        Empty rows skipped; whitespace stripped before normalization.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If the CSV does not have a 'patient_id' column.
    """
    df = pd.read_csv(csv_path)
    if "patient_id" not in df.columns:
        raise ValueError(f"CSV must have 'patient_id' column; found: {list(df.columns)}")
    ids = df["patient_id"].astype(str).str.strip()
    ids = ids[ids != ""]
    return [_normalize_patient_id(pid) for pid in ids.tolist()]
