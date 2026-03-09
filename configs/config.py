"""
Centralized configuration for I-CARE EEG network analysis.

All paths default to Google Drive locations used in Colab.
Override any value via environment variables (e.g. DATA_ROOT, METADATA_PATH).
"""

import os

# ---------------------------------------------------------------------------
# Data paths (override with environment variables)
# ---------------------------------------------------------------------------

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    "/content/drive/MyDrive/icare_project",
)

METADATA_PATH = os.environ.get(
    "METADATA_PATH",
    "/content/drive/MyDrive/icare_project/metadata_clean.csv",
)

BATCH_FOLDER = os.environ.get(
    "BATCH_FOLDER",
    "/content/drive/MyDrive/icare_project/batches",
)

AUDIT_PATH = os.environ.get(
    "AUDIT_PATH",
    "/content/drive/MyDrive/icare_project/analysis/selected_patient_metadata_audit.csv",
)

# Raw EEG root: data/raw/eeg/ under DATA_ROOT
EEG_RAW_ROOT = os.environ.get(
    "EEG_RAW_ROOT",
    os.path.join(DATA_ROOT, "data", "raw", "eeg"),
)

# ---------------------------------------------------------------------------
# Output paths (all under Google Drive; never in repository)
# ---------------------------------------------------------------------------

FEATURE_OUTPUT_PATH = os.environ.get(
    "FEATURE_OUTPUT_PATH",
    "/content/drive/MyDrive/icare_project/features",
)

INTERMEDIATE_OUTPUT_PATH = os.environ.get(
    "INTERMEDIATE_OUTPUT_PATH",
    "/content/drive/MyDrive/icare_project/intermediate",
)

MODEL_OUTPUT_PATH = os.environ.get(
    "MODEL_OUTPUT_PATH",
    "/content/drive/MyDrive/icare_project/models",
)

LOG_PATH = os.environ.get(
    "LOG_PATH",
    "/content/drive/MyDrive/icare_project/logs",
)

# Analysis outputs (channel inventory, frequency, common channels)
ANALYSIS_OUTPUT_PATH = os.environ.get(
    "ANALYSIS_OUTPUT_PATH",
    "/content/drive/MyDrive/icare_project/analysis",
)

# Preprocessing outputs: one file per patient (windows or connectivity)
WINDOWS_OUTPUT_DIR = os.environ.get(
    "WINDOWS_OUTPUT_DIR",
    os.path.join(INTERMEDIATE_OUTPUT_PATH, "windows"),
)

# Path to common EEG channel list (JSON array) from channel consistency stage
COMMON_CHANNELS_PATH = os.environ.get(
    "COMMON_CHANNELS_PATH",
    os.path.join(ANALYSIS_OUTPUT_PATH, "common_eeg_channels.json"),
)

# Local temp directory for intermediate writes (Colab: /content/tmp); moved to Drive when done
TEMP_DIR = os.environ.get("TEMP_DIR", "/content/tmp")

# ---------------------------------------------------------------------------
# Processing parameters
# ---------------------------------------------------------------------------

MAX_EEG_HOURS = 48
WINDOW_SECONDS = 30
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 40.0

# ---------------------------------------------------------------------------
# Batch and patient identification
# ---------------------------------------------------------------------------

# Batch files: batch_1.csv, batch_2.csv, batch_3.csv, batch_4.csv
# batch_0 uses: subset_100_ids_updated.csv
BATCH_ZERO_FILENAME = "subset_100_ids_updated.csv"
BATCH_FILENAME_PATTERN = "batch_{}.csv"

# Patient IDs are strings (e.g. "0284", "0286"); no header in batch CSVs.
TOTAL_PATIENTS = 294

# Maximum segments to process per patient (~1 hour each → ~48 hours)
MAX_EEG_SEGMENTS = 48

# Canonical patient list (all 294 with downloaded EEG); has header "patient_id"
CANONICAL_PATIENT_LIST_FILENAME = "all_downloaded_patients_294.csv"

# Balanced splits for parallel workers (~98 each); header "patient_id"
# Balanced by Hospital, Outcome, Sex
PATIENT_SPLIT_FILENAME_PATTERN = "patient_split_{}.csv"
PATIENT_SPLIT_INDICES = (1, 2, 3)
