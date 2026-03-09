"""
Parse WFDB .hea header files to extract channel (signal) names.

Uses the wfdb library. Only header metadata is read; no .mat signal data
is loaded in this module.
"""

import os
from typing import List

import wfdb


def get_channel_names_from_hea(hea_path: str) -> List[str]:
    """
    Read a WFDB header file and return the list of channel/signal names.

    Args:
        hea_path: Path to the .hea file, or path to the record (without
                  extension) so that WFDB can locate the header.

    Returns:
        List of channel name strings as they appear in the header
        (e.g. ["Fp1", "Fp2", "F3", ...]).

    Raises:
        FileNotFoundError: If the header file is missing.
        Exception: For invalid or unreadable WFDB headers.
    """
    record_name = os.path.abspath(hea_path)
    if record_name.lower().endswith(".hea"):
        record_name = record_name[:-4]
    record = wfdb.rdheader(record_name=record_name)
    if record.sig_name is None:
        return []
    return list(record.sig_name)
