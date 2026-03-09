"""
Load patient IDs from CSV files (canonical list or balanced splits).

Used by channel consistency analysis and by downstream pipelines to determine
which patients to process. Patient IDs are always treated as strings.
"""

from typing import List


def load_patient_ids(csv_path: str) -> List[str]:
    """
    Load patient IDs from a CSV file with header 'patient_id'.

    Args:
        csv_path: Path to the CSV (e.g. all_downloaded_patients_294.csv
                  or patient_split_1.csv).

    Returns:
        List of patient ID strings. Empty rows skipped; whitespace stripped.
        IDs are not converted to integers.

    Raises:
        FileNotFoundError: If csv_path does not exist.
    """
    raise NotImplementedError("Patient list loading not yet implemented.")
