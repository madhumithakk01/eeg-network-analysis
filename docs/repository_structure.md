# Repository Structure and File Descriptions

This document describes the purpose of each top-level directory and key file. No implementation details—only structure and roles.

---

## Root

| Item | Purpose |
|------|---------|
| **README.md** | Project overview, setup, data access, pipeline summary, links to docs. |
| **requirements.txt** | Pinned Python dependencies for Colab reproducibility. Do not change versions without re-testing. |
| **.gitignore** | Excludes data, CSVs, outputs, Python/IDE artifacts. Ensures repo contains only code, config, and docs. |

---

## configs/

| Item | Purpose |
|------|---------|
| **config.py** | Central configuration: data paths (DATA_ROOT, METADATA_PATH, BATCH_FOLDER, EEG_RAW_ROOT, AUDIT_PATH), output paths (FEATURE_OUTPUT_PATH, INTERMEDIATE_OUTPUT_PATH, MODEL_OUTPUT_PATH, LOG_PATH, ANALYSIS_OUTPUT_PATH), processing parameters (MAX_EEG_HOURS, WINDOW_SECONDS, BANDPASS_LOW, BANDPASS_HIGH), and patient list filenames (CANONICAL_PATIENT_LIST_FILENAME, PATIENT_SPLIT_FILENAME_PATTERN). All overridable via environment variables. |

---

## docs/

| Item | Purpose |
|------|---------|
| **dataset_description.md** | Dataset name, locations on Drive, metadata and audit files, batch system, EEG segment rules, patient ID handling, time-window distribution, storage rules. |
| **pipeline_design.md** | Planned pipeline stages (channel consistency, preprocessing, connectivity, graph construction, graph features, ML), data flow, module mapping, experiment tracking, parallel compute. |
| **colab_worker_setup.md** | Step-by-step Colab setup: mount Drive, clone repo, install deps, set worker/range, load patient subset, run processing, write only to Drive. Checklist and multi-account notes. |
| **repository_structure.md** | This file—file and directory descriptions. |
| **channel_consistency_design.md** | Design for EEG channel consistency stage: .hea parsing, patient list loading, non-EEG filter, channel inventory, frequency and intersection, output artifacts (channel_inventory.csv, channel_frequency.csv, common_eeg_channels.json). |

---

## notebooks/

| Subdir | Purpose |
|--------|---------|
| **exploration/** | Exploratory analysis notebooks; no data or large outputs committed. |
| **debugging/** | Debugging and validation notebooks; no data or large outputs committed. |

---

## src/

Source packages; implementation to be added later.

| Package | Purpose |
|---------|---------|
| **data_loading** | Load EEG (e.g. WFDB) and metadata; channel consistency analysis. Submodules: **patient_list** (load patient IDs from canonical/split CSV), **hea_parsing** (extract channel names from .hea), **channel_filter** (exclude non-EEG channels), **channel_inventory** (orchestrate scan, frequency, intersection, write to analysis/). |
| **preprocessing** | Bandpass filtering, segmentation (30 s windows). |
| **connectivity** | Functional connectivity (e.g. Pearson correlation between electrodes). |
| **graph_features** | Graph-theory metrics: clustering, efficiency, density, path length, centrality. |
| **modeling** | ML training and evaluation (Random Forest, XGBoost, LightGBM). |
| **utils** | Shared helpers used across pipeline stages. |

---

## scripts/

| Script | Purpose |
|--------|---------|
| **run_channel_inventory.py** | Entry point for channel consistency analysis: load canonical patient list, scan .hea per patient, filter non-EEG, write channel_inventory.csv, channel_frequency.csv, common_eeg_channels.json to ANALYSIS_OUTPUT_PATH. |
| **run_feature_extraction.py** | Entry point for feature pipeline: load patient subset, preprocessing → connectivity → graph features, write to Drive. Worker index and patient range via CLI or env. |
| **run_model_training.py** | Entry point for model training: load features and labels, train models, write models and logs to Drive. |

---

## experiments/

| Subdir | Purpose |
|--------|---------|
| **feature_runs/** | Logs, config snapshots, or run IDs for feature extraction; no large artifacts. |
| **model_runs/** | Logs, config snapshots, or run IDs for model training; no large artifacts. |

---

## tests/

| Subdir | Purpose |
|--------|---------|
| **data_loading/** | Unit/integration tests for data_loading. |
| **preprocessing/** | Tests for preprocessing. |
| **connectivity/** | Tests for connectivity. |

Additional test directories (e.g. `graph_features`, `modeling`) can be added when those modules are implemented.

---

## Data and outputs (not in repo)

All of the following live on **Google Drive** only:

- **data/raw/eeg/** — Raw EEG (.mat, .hea) per patient folder.
- **batches/** — batch_1.csv … batch_4.csv; subset_100_ids_updated.csv; **all_downloaded_patients_294.csv** (canonical list); **patient_split_1/2/3.csv** (balanced splits).
- **metadata_clean.csv**, **analysis/selected_patient_metadata_audit.csv**
- **analysis/** — Channel consistency outputs: channel_inventory.csv, channel_frequency.csv, common_eeg_channels.json.
- **features/** — Extracted graph/connectivity features.
- **intermediate/** — Preprocessing and connectivity intermediates.
- **models/** — Trained model artifacts.
- **logs/** — Run logs.

The repository never contains these; `.gitignore` enforces exclusion.
