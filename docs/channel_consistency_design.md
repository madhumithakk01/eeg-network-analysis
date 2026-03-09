# Channel Consistency Analysis: Design Document

## Purpose

Before preprocessing or connectivity analysis, the pipeline must determine **which EEG electrodes are consistently available across the dataset**. Different hospitals may record different electrode sets; connectivity analysis requires a **common set of EEG channels** for all patients. This stage also **excludes non-EEG channels** (EKG, ECG, EMG, RESP, SpO2, etc.) so that brain networks are not corrupted by physiological sensors.

This document defines the architecture, data flow, module responsibilities, and output artifacts for the **EEG Channel Consistency Analysis** stage. It does not specify preprocessing, connectivity, or ML.

---

## Goals

1. **Dataset inspection** — Scan WFDB `.hea` headers across all 294 patients and collect the set of channel names per patient.
2. **Non-EEG exclusion** — Identify and exclude non-brain channels so only valid EEG electrodes are considered.
3. **Channel inventory** — Produce a per-patient channel list and channel frequency across patients.
4. **Common EEG set** — Compute the **intersection** of valid EEG channels across all patients; this set will drive downstream preprocessing and connectivity.

---

## Critical Data Issue: Non-EEG Channels

WFDB `.hea` files may list **non-EEG channels** from clinical recordings, for example:

- EKG, ECG  
- EMG  
- RESP  
- SpO2  
- Pulse  
- Temp  

If these are not filtered out, connectivity matrices will mix brain and non-brain signals and **silently corrupt** network analysis. Therefore this stage must produce a **whitelist of valid EEG channels** (or equivalently, a blacklist of known non-EEG patterns to exclude). The **common_eeg_channels** set is defined only over channels that pass this filter.

---

## Inputs

| Input | Location | Description |
|------|----------|-------------|
| Canonical patient list | `BATCH_FOLDER` / `all_downloaded_patients_294.csv` | Single column `patient_id`; one row per patient (294 total). Patient IDs are strings. |
| Raw EEG directory | `EEG_RAW_ROOT` | Per-patient subdirs named by patient ID, e.g. `0284/`, `0286/`. |
| WFDB header files | Inside each patient dir | `.hea` files (e.g. `0284_001_000_EEG.hea`). Channel names and metadata are in the header. |

**Patient list loading:** The script loads patient IDs from the canonical CSV. It does **not** use the legacy batch_0–4 CSVs for this stage; the single consolidated list is the source of truth. For parallel workers, the same codebase can later load `patient_split_1.csv`, `patient_split_2.csv`, or `patient_split_3.csv` and run inventory only on that subset (e.g. for validation); the **full** channel consistency run uses the canonical 294 list.

**Directory scanning:** For each patient ID, the pipeline resolves `EEG_RAW_ROOT / patient_id` and discovers all `.hea` files in that directory. It is sufficient to read **one** `.hea` per patient to get the channel list, because channel layout is typically consistent across segments within a patient. The design uses the **first** segment’s header (e.g. the alphabetically first `.hea`) to avoid reading every segment.

---

## Module Responsibilities

### 1. Patient list loading — `src/data_loading/patient_list.py`

- **Responsibility:** Load patient IDs from a CSV file.
- **Inputs:** Path to CSV (e.g. canonical list or a split file). CSV has header `patient_id`.
- **Output:** List (or sequence) of patient ID strings.
- **Details:** Strip whitespace; treat IDs as strings; do not convert to int. Skip empty rows. Path is configurable (default: `BATCH_FOLDER` + `CANONICAL_PATIENT_LIST_FILENAME`).

### 2. WFDB header parsing — `src/data_loading/hea_parsing.py`

- **Responsibility:** Parse WFDB `.hea` files to obtain the list of channel names (labels).
- **Inputs:** Path to a single `.hea` file, or path to a record (without extension) so that WFDB can find the header.
- **Output:** List of channel name strings as they appear in the header (e.g. `["Fp1", "Fp2", "F3", ...]`).
- **Details:** Use the `wfdb` library (already in requirements) to read the header. WFDB provides channel/signal metadata; channel names are typically in the header line for each signal. Extract the signal names/labels and return them. No signal data (.mat) is read in this stage.

### 3. Non-EEG channel filter — `src/data_loading/channel_filter.py`

- **Responsibility:** Define which channels are valid EEG vs non-EEG, and filter a list of channel names to only valid EEG.
- **Inputs:** List of channel names from a header; optionally a configurable whitelist or blacklist.
- **Output:** List of channel names that are considered valid EEG (whitelist).
- **Details:** Non-EEG channels are identified by name patterns (e.g. EKG, ECG, EMG, RESP, SpO2, Pulse, Temp). The module maintains a set of **excluded patterns** (or a **whitelist** of known EEG electrode names, e.g. 10–20 system). Design choice: either (a) **blacklist** known non-EEG substrings and keep all others, or (b) **whitelist** known EEG channel names and keep only those. Whitelist is safer to avoid accidentally including unknown non-EEG channels. Both approaches must be documented; the output artifact `common_eeg_channels.json` is the set of channels that pass the filter and are present in every patient.

### 4. Channel inventory orchestration — `src/data_loading/channel_inventory.py`

- **Responsibility:** Orchestrate the full channel consistency run: iterate patients, read one header per patient, collect channel lists, apply non-EEG filter, compute frequency statistics, compute intersection, and write outputs.
- **Inputs:** List of patient IDs; paths to EEG root and output directory (from config).
- **Outputs:** Writes three artifacts to `ANALYSIS_OUTPUT_PATH` (see below).
- **Details:**
  - For each patient ID: resolve patient dir `EEG_RAW_ROOT / patient_id`, find at least one `.hea` file (e.g. first by name). If none found, log and skip patient (or record “no channels”).
  - Use `hea_parsing` to get raw channel list from that header.
  - Use `channel_filter` to retain only valid EEG channel names.
  - Store per-patient result: patient_id → list of valid EEG channel names (or full list before filter for inventory; see below).
  - **channel_inventory.csv:** One row per patient. Columns: `patient_id`, and either a column with the full comma-separated list of detected channels, or one column per channel (sparse). Design: one row per patient, one column `channels` with the list of channel names as a string (e.g. comma-separated) so the file is compact and readable. Alternatively: `patient_id`, `channel_count`, `channel_list` (comma-separated).
  - **channel_frequency.csv:** For each channel name that appears in at least one patient, count how many patients have that channel. Columns: `channel`, `patient_count`. Sorted by `patient_count` descending. Includes only channels that pass the non-EEG filter if the inventory is built after filtering; or include all and mark in documentation that downstream “common set” uses filtered only.
  - **common_eeg_channels.json:** List (array) of channel names that (1) pass the non-EEG filter and (2) appear in **every** patient in the canonical list. This is the intersection of (filtered) channel sets across all 294 patients. Format: JSON array of strings, e.g. `["Fp1", "Fp2", "F3", ...]`.

---

## Data Flow Summary

```
all_downloaded_patients_294.csv
        → patient_list.load() → [patient_id, ...]
        → for each patient_id:
              EEG_RAW_ROOT / patient_id / *.hea
                    → pick first .hea
                    → hea_parsing.get_channel_names(hea_path)
                    → channel_filter.filter_eeg_only(channel_names)
                    → store (patient_id, eeg_channel_list)
        → aggregate:
              channel_inventory.csv (patient_id, channel list per patient)
              channel_frequency.csv (channel, patient_count)
              common_eeg_channels = intersection of all per-patient eeg sets
              common_eeg_channels.json
```

---

## Output Artifacts

All written to **`ANALYSIS_OUTPUT_PATH`** (default: `/content/drive/MyDrive/icare_project/analysis/`).

| Artifact | Description |
|----------|-------------|
| **channel_inventory.csv** | One row per patient. Columns: `patient_id`, and a column with the full list of detected (and filtered) EEG channels for that patient (e.g. comma-separated or a structured column). |
| **channel_frequency.csv** | One row per channel. Columns: `channel`, `patient_count`. How many patients have that channel (after non-EEG filter). Sorted by `patient_count` descending. |
| **common_eeg_channels.json** | JSON array of channel names that are (1) valid EEG (pass filter) and (2) present in every patient. This is the **whitelist** used by preprocessing and connectivity. |

---

## Script Entry Point: `scripts/run_channel_inventory.py`

- **Responsibility:** CLI/orchestration for the channel consistency stage.
- **Steps:**
  1. Load config (paths, filenames).
  2. Load patient IDs from canonical list (or from a split file if an optional argument is provided).
  3. Ensure `ANALYSIS_OUTPUT_PATH` exists (create if necessary).
  4. Call the channel inventory orchestration (e.g. `channel_inventory.run(...)`) with patient list and paths.
  5. Write `channel_inventory.csv`, `channel_frequency.csv`, and `common_eeg_channels.json` to `ANALYSIS_OUTPUT_PATH`.
  6. Log summary (e.g. number of patients processed, number of common channels, any patients skipped).
- **Arguments (planned):** Optional `--patient-list` to override the default canonical list path; optional `--output-dir` to override `ANALYSIS_OUTPUT_PATH`. No implementation required in this design phase.

---

## Configuration

Existing config already provides:

- `DATA_ROOT`, `BATCH_FOLDER`, `EEG_RAW_ROOT`, `METADATA_PATH`, `LOG_PATH`, etc.
- `MAX_EEG_HOURS`, `WINDOW_SECONDS`, `BANDPASS_LOW`, `BANDPASS_HIGH` (for downstream use).

**Added for this stage:**

- **`ANALYSIS_OUTPUT_PATH`** — Directory where channel inventory and related analysis outputs are written (default: `.../icare_project/analysis/`).
- **`CANONICAL_PATIENT_LIST_FILENAME`** — `all_downloaded_patients_294.csv`.
- **`PATIENT_SPLIT_FILENAME_PATTERN`** — `patient_split_{}.csv` for optional use with split index 1, 2, or 3.

All paths remain overridable via environment variables for local and Colab runs.

---

## Dependencies

No new dependencies. Use existing stack:

- **wfdb** — For reading `.hea` headers and extracting channel/signal names.
- **pandas** — For building and writing `channel_inventory.csv` and `channel_frequency.csv`.
- **json** (stdlib) — For writing `common_eeg_channels.json`.

---

## Testing (Scope for Later)

- **patient_list:** Load a small CSV with known patient IDs; assert count and string type.
- **hea_parsing:** Given a sample `.hea` path (fixture), assert returned channel list matches expected labels.
- **channel_filter:** Given a list including "EKG", "Fp1", "Fp2", assert only EEG names remain.
- **channel_inventory:** With a tiny patient list and mock header reader, assert output CSV and JSON structure and intersection logic.

Tests live under `tests/data_loading/`. No implementation in this design phase.

---

## Summary

| Concern | Handled by |
|--------|------------|
| Load patient IDs from canonical list | `patient_list.py` |
| Parse .hea to get channel names | `hea_parsing.py` (using wfdb) |
| Exclude non-EEG channels | `channel_filter.py` |
| Scan patient dirs, aggregate, frequency, intersection | `channel_inventory.py` |
| Write channel_inventory.csv, channel_frequency.csv, common_eeg_channels.json | `channel_inventory.py` → `ANALYSIS_OUTPUT_PATH` |
| Orchestration and CLI | `scripts/run_channel_inventory.py` |

This design supports reproducible channel consistency analysis, keeps non-EEG channels out of connectivity, and produces a single common EEG channel set for the rest of the pipeline.
