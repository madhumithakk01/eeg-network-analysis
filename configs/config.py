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
