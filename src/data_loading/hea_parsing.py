"""
Parse WFDB .hea header files to extract channel (signal) names.

Uses the wfdb library. Only header metadata is read; no .mat signal data
is loaded in this module.
"""

from typing import List


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
    raise NotImplementedError("HEA parsing not yet implemented.")
