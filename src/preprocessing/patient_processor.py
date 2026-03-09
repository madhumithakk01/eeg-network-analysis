"""
Process one patient's EEG: load segments, filter, average reference, window,
compute connectivity matrices, and save.

Fault-tolerant and resumable: skips patient if output already exists;
writes to temp then moves to final path. Segment limit is by count (first 48).
Output is one .npy per patient with shape (n_windows, n_channels, n_channels).
"""

import os
import shutil
from typing import Any, Dict, List, Optional

import numpy as np

from src.connectivity.pearson import compute_connectivity_batch
from .eeg_loader import load_eeg_segment
from .signal_filter import bandpass_filter
from .windowing import segment_into_windows_list


def _list_segment_paths(patient_dir: str, max_segments: int) -> List[str]:
    """
    List EEG segment record paths in chronological order, limited to max_segments.

    Segments are identified by .hea files; record path is dir + base name without .hea.
    """
    try:
        entries = os.listdir(patient_dir)
    except OSError:
        return []
    hea_files = sorted(f for f in entries if f.lower().endswith(".hea"))
    segment_paths = []
    for f in hea_files[:max_segments]:
        base = f[:-4] if f.lower().endswith(".hea") else f
        segment_paths.append(os.path.join(patient_dir, base))
    return segment_paths


def process_patient(
    patient_id: str,
    eeg_raw_root: str,
    output_dir: str,
    common_channel_names: List[str],
    window_seconds: float = 30.0,
    bandpass_low: float = 0.5,
    bandpass_high: float = 40.0,
    max_segments: int = 48,
    temp_dir: Optional[str] = None,
    validate_connectivity: bool = False,
) -> Dict[str, Any]:
    """
    Process one patient: load up to max_segments, filter, average reference,
    window, compute connectivity per segment batch, and save.

    If the final output file (patient_id_connectivity.npy) already exists in
    output_dir, the patient is skipped (resumable pipeline). Writes to
    temp_dir first, then moves to output_dir for atomicity.

    Pipeline per segment: load -> bandpass filter -> average reference ->
    window segmentation -> compute_connectivity_batch -> append matrices.

    Args:
        patient_id: Four-digit zero-padded ID (e.g. "0284").
        eeg_raw_root: Root directory containing patient subdirs.
        output_dir: Directory for final output (e.g. WINDOWS_OUTPUT_DIR).
        common_channel_names: List of channel names to load (from common_eeg_channels.json).
        window_seconds: Window length in seconds.
        bandpass_low: Bandpass lower cutoff in Hz.
        bandpass_high: Bandpass upper cutoff in Hz.
        max_segments: Maximum number of segments to process per patient (default 48).
        temp_dir: Directory for temporary file before move (default uses config or /content/tmp).
        validate_connectivity: If True, run validate_connectivity_batch() after each segment's
            connectivity computation; raises on failure. Default False to avoid slowing full pipeline.

    Returns:
        Summary dict with keys: processed, skipped, reason, n_segments_processed,
        n_windows, n_connectivity_matrices, output_path, error.
    """
    if temp_dir is None:
        temp_dir = os.environ.get("TEMP_DIR", "/content/tmp")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    final_name = f"{patient_id}_connectivity.npy"
    final_path = os.path.join(output_dir, final_name)
    if os.path.isfile(final_path):
        return {
            "processed": False,
            "skipped": True,
            "reason": "output_exists",
            "n_segments_processed": 0,
            "n_windows": 0,
            "n_connectivity_matrices": 0,
            "output_path": final_path,
            "error": None,
        }

    patient_dir = os.path.join(eeg_raw_root, patient_id.strip())
    segment_paths = _list_segment_paths(patient_dir, max_segments)
    if not segment_paths:
        return {
            "processed": False,
            "skipped": True,
            "reason": "no_segments",
            "n_segments_processed": 0,
            "n_windows": 0,
            "n_connectivity_matrices": 0,
            "output_path": None,
            "error": None,
        }

    all_connectivity: List[np.ndarray] = []
    n_segments_processed = 0
    n_windows_total = 0
    n_channels_expected = len(common_channel_names)

    # Minimum segment duration (seconds) to allow a single full-segment window when 30s windowing yields 0
    MIN_SEGMENT_DURATION_SEC = 1.0

    for seg_idx, record_path in enumerate(segment_paths):
        try:
            data, fs = load_eeg_segment(record_path, common_channel_names)
        except Exception as e:
            print(
                f"  [segment_debug] patient_id={patient_id} seg_idx={seg_idx} path={record_path} | "
                f"reason=load_exception | exception={e!r}"
            )
            continue

        # Ensure (n_samples, n_channels); WFDB may return (n_channels, n_samples) in some setups
        if data.shape[0] == n_channels_expected and data.shape[1] != n_channels_expected:
            data = data.T
            print(
                f"  [segment_debug] patient_id={patient_id} seg_idx={seg_idx} | "
                f"signal_orientation=transposed to (n_samples, n_channels)"
            )
        signal_shape = data.shape
        n_samples = signal_shape[0]
        if signal_shape[1] != n_channels_expected:
            print(
                f"  [segment_debug] patient_id={patient_id} seg_idx={seg_idx} path={record_path} | "
                f"reason=missing_channels | expected={n_channels_expected} got={signal_shape[1]}"
            )
            continue

        # Process each segment with its own sampling rate; no fs_mismatch skip (reduces data loss)
        try:
            filtered = bandpass_filter(
                data, fs, low_hz=bandpass_low, high_hz=bandpass_high
            )
        except Exception as e:
            print(
                f"  [segment_debug] patient_id={patient_id} seg_idx={seg_idx} path={record_path} | "
                f"reason=filter_failure | fs={fs} | exception={e!r}"
            )
            continue
        filtered = filtered - filtered.mean(axis=1, keepdims=True)

        windows_list = segment_into_windows_list(filtered, fs, window_seconds)
        n_windows_seg = len(windows_list)
        if n_windows_seg == 0:
            segment_duration_sec = n_samples / fs if fs > 0 else 0.0
            required_samples_30s = int(round(window_seconds * fs)) if fs > 0 else 0
            if segment_duration_sec >= MIN_SEGMENT_DURATION_SEC:
                # Salvage short segment as one full-segment window (one connectivity matrix)
                windows_list = segment_into_windows_list(
                    filtered, fs, window_seconds=segment_duration_sec
                )
                n_windows_seg = len(windows_list)
            if n_windows_seg == 0:
                print(
                    f"  [segment_debug] patient_id={patient_id} seg_idx={seg_idx} path={record_path} | "
                    f"reason=windowing_returned_0_windows | signal_shape={signal_shape} fs={fs} "
                    f"window_seconds={window_seconds} required_samples_30s={required_samples_30s} "
                    f"segment_duration_sec={segment_duration_sec:.2f} n_full_30s_windows={n_samples // max(1, required_samples_30s)}"
                )
                continue
        window_shape = windows_list[0].shape if windows_list else None
        print(
            f"  [segment_debug] patient_id={patient_id} seg_idx={seg_idx} | signal_shape={signal_shape} fs={fs} | "
            f"channels={signal_shape[1]} windows={n_windows_seg} window_shape={window_shape}"
        )

        try:
            windows_array = np.stack(windows_list, axis=0)
            conn = compute_connectivity_batch(windows_array)
        except Exception as e:
            print(
                f"  [segment_debug] patient_id={patient_id} seg_idx={seg_idx} path={record_path} | "
                f"reason=connectivity_failure | exception={e!r}"
            )
            continue
        n_conn = conn.shape[0]
        print(
            f"  [segment_debug] patient_id={patient_id} seg_idx={seg_idx} | connectivity_matrices={n_conn}"
        )

        if validate_connectivity:
            try:
                from src.utils.connectivity_checks import validate_connectivity_batch as _validate
                _validate(conn)
            except Exception as e:
                print(
                    f"  [segment_debug] patient_id={patient_id} seg_idx={seg_idx} path={record_path} | "
                    f"reason=validation_failure | exception={e!r}"
                )
                continue
        all_connectivity.append(conn)
        n_segments_processed += 1
        n_windows_total += conn.shape[0]

    if not all_connectivity:
        return {
            "processed": False,
            "skipped": True,
            "reason": "no_windows",
            "n_segments_processed": 0,
            "n_windows": 0,
            "n_connectivity_matrices": 0,
            "output_path": None,
            "error": None,
        }

    connectivity_array = np.concatenate(all_connectivity, axis=0).astype(np.float32)
    tmp_name = f"{patient_id}_connectivity.tmp.npy"
    tmp_path = os.path.join(temp_dir, tmp_name)
    try:
        np.save(tmp_path, connectivity_array)
        try:
            os.replace(tmp_path, final_path)
        except OSError:
            shutil.move(tmp_path, final_path)
    except Exception as e:
        if os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return {
            "processed": False,
            "skipped": False,
            "reason": "write_failed",
            "n_segments_processed": n_segments_processed,
            "n_windows": n_windows_total,
            "n_connectivity_matrices": connectivity_array.shape[0],
            "output_path": None,
            "error": str(e),
        }

    n_matrices = connectivity_array.shape[0]
    print(
        f"  [patient_processor] patient_id={patient_id} | "
        f"segments_processed={n_segments_processed} | "
        f"windows_produced={n_windows_total} | "
        f"connectivity_matrices_saved={n_matrices}"
    )
    return {
        "processed": True,
        "skipped": False,
        "reason": None,
        "n_segments_processed": n_segments_processed,
        "n_windows": n_windows_total,
        "n_connectivity_matrices": n_matrices,
        "output_path": final_path,
        "error": None,
    }
