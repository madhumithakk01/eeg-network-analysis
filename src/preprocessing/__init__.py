"""EEG preprocessing: load, filter, window, and patient-level processing."""

from .eeg_loader import load_eeg_segment
from .artifact_qc import evaluate_window_quality, filter_windows_by_quality
from .patient_processor import process_patient
from .resample import resample_signal_poly
from .signal_filter import bandpass_filter
from .windowing import segment_into_windows, segment_into_windows_list

__all__ = [
    "load_eeg_segment",
    "evaluate_window_quality",
    "filter_windows_by_quality",
    "resample_signal_poly",
    "bandpass_filter",
    "segment_into_windows",
    "segment_into_windows_list",
    "process_patient",
]
