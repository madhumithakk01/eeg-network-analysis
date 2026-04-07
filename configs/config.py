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

# Sparsification stage: dense connectivity -> sparse connectivity (one file per patient)
SPARSE_CONNECTIVITY_DIR = os.environ.get(
    "SPARSE_CONNECTIVITY_DIR",
    os.path.join(INTERMEDIATE_OUTPUT_PATH, "sparse_connectivity"),
)

# Fraction of edges to retain in sparse connectivity (density-based thresholding)
SPARSE_DENSITY = float(os.environ.get("SPARSE_DENSITY", "0.15"))

# Graph feature extraction: sparse connectivity -> per-window feature vectors
GRAPH_FEATURES_DIR = os.environ.get(
    "GRAPH_FEATURES_DIR",
    os.path.join(INTERMEDIATE_OUTPUT_PATH, "graph_features"),
)

# Temporal aggregation: patient-level dataset (parquet)
PATIENT_TEMPORAL_DATASET_PATH = os.environ.get(
    "PATIENT_TEMPORAL_DATASET_PATH",
    os.path.join(ANALYSIS_OUTPUT_PATH, "patient_temporal_dataset.parquet"),
)

# ML pipeline: model artifacts and results (analysis/model_results/)
MODEL_RESULTS_PATH = os.environ.get(
    "MODEL_RESULTS_PATH",
    os.path.join(ANALYSIS_OUTPUT_PATH, "model_results"),
)

# Path to common EEG channel list (JSON array) from channel consistency stage
COMMON_CHANNELS_PATH = os.environ.get(
    "COMMON_CHANNELS_PATH",
    os.path.join(ANALYSIS_OUTPUT_PATH, "common_eeg_channels.json"),
)

# Network dynamics / trajectory analysis outputs
NETWORK_DYNAMICS_OUTPUT_PATH = os.environ.get(
    "NETWORK_DYNAMICS_OUTPUT_PATH",
    os.path.join(ANALYSIS_OUTPUT_PATH, "network_dynamics"),
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
# Window-level EEG artifact QC (Step 2.1)
# ---------------------------------------------------------------------------

# Keep disabled by default to preserve baseline behavior; enable via CLI or env.
PREPROCESS_ENABLE_WINDOW_QC = os.environ.get("PREPROCESS_ENABLE_WINDOW_QC", "0") == "1"

# Relative flat-channel rule: channel std < max(abs_floor, ratio * median_channel_std)
PREPROCESS_QC_FLAT_STD_MIN_ABS = float(os.environ.get("PREPROCESS_QC_FLAT_STD_MIN_ABS", "1e-8"))
PREPROCESS_QC_FLAT_STD_MIN_RATIO = float(os.environ.get("PREPROCESS_QC_FLAT_STD_MIN_RATIO", "1e-3"))
PREPROCESS_QC_MAX_FLAT_CHANNEL_FRAC = float(os.environ.get("PREPROCESS_QC_MAX_FLAT_CHANNEL_FRAC", "0.2"))

# Low-uniqueness rule (clipping/quantization proxy)
PREPROCESS_QC_MIN_UNIQUE_VALUE_RATIO = float(os.environ.get("PREPROCESS_QC_MIN_UNIQUE_VALUE_RATIO", "0.02"))
PREPROCESS_QC_MAX_LOW_UNIQUE_CHANNEL_FRAC = float(
    os.environ.get("PREPROCESS_QC_MAX_LOW_UNIQUE_CHANNEL_FRAC", "0.3")
)

# Robust amplitude outlier rule
PREPROCESS_QC_HIGH_AMP_ROBUST_Z = float(os.environ.get("PREPROCESS_QC_HIGH_AMP_ROBUST_Z", "12.0"))
PREPROCESS_QC_MAX_HIGH_AMP_FRAC = float(os.environ.get("PREPROCESS_QC_MAX_HIGH_AMP_FRAC", "0.02"))

# Mains power ratio rule (power around mains / power in 1-40 Hz)
PREPROCESS_QC_MAINS_HZ = float(os.environ.get("PREPROCESS_QC_MAINS_HZ", "50.0"))
PREPROCESS_QC_MAINS_BAND_HZ = float(os.environ.get("PREPROCESS_QC_MAINS_BAND_HZ", "1.0"))
PREPROCESS_QC_MAX_MAINS_RATIO = float(os.environ.get("PREPROCESS_QC_MAX_MAINS_RATIO", "0.35"))

# ---------------------------------------------------------------------------
# Short-segment salvage policy (Step 2.2)
# ---------------------------------------------------------------------------

# Backward-compatible default keeps prior behavior.
PREPROCESS_ENABLE_SHORT_SEGMENT_SALVAGE = os.environ.get(
    "PREPROCESS_ENABLE_SHORT_SEGMENT_SALVAGE", "1"
) == "1"

# Minimum segment duration (seconds) required before salvage into a single window.
# Set to 30.0 to effectively disable salvage for short (<30s) segments.
PREPROCESS_MIN_SALVAGE_DURATION_SEC = float(
    os.environ.get("PREPROCESS_MIN_SALVAGE_DURATION_SEC", "1.0")
)

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

# ---------------------------------------------------------------------------
# Temporal deep learning pipeline (window-level graph features -> outcome)
# ---------------------------------------------------------------------------

TEMPORAL_DL_OUTPUT_PATH = os.environ.get(
    "TEMPORAL_DL_OUTPUT_PATH",
    os.path.join(ANALYSIS_OUTPUT_PATH, "model_results_temporal_dl"),
)

TEMPORAL_DL_BATCH_SIZE = int(os.environ.get("TEMPORAL_DL_BATCH_SIZE", "16"))

TEMPORAL_DL_EPOCHS = int(os.environ.get("TEMPORAL_DL_EPOCHS", "40"))

# ---------------------------------------------------------------------------
# Connectivity graph deep learning (raw adjacency sequences -> outcome)
# ---------------------------------------------------------------------------

CONNECTIVITY_DL_OUTPUT_PATH = os.environ.get(
    "CONNECTIVITY_DL_OUTPUT_PATH",
    os.path.join(ANALYSIS_OUTPUT_PATH, "model_results_connectivity_dl"),
)

CONNECTIVITY_DL_STRIDE = int(os.environ.get("CONNECTIVITY_DL_STRIDE", "8"))

# Stride-4 experiment: same model/dataset/CV, separate output directory
CONNECTIVITY_DL_STRIDE4_OUTPUT_PATH = os.environ.get(
    "CONNECTIVITY_DL_STRIDE4_OUTPUT_PATH",
    os.path.join(ANALYSIS_OUTPUT_PATH, "model_results_connectivity_dl_stride4"),
)

CONNECTIVITY_DL_BATCH_SIZE = int(os.environ.get("CONNECTIVITY_DL_BATCH_SIZE", "8"))

CONNECTIVITY_DL_EPOCHS = int(os.environ.get("CONNECTIVITY_DL_EPOCHS", "40"))
