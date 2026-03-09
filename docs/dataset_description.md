# Dataset Description: I-CARE EEG

## Overview

**Dataset name:** I-CARE EEG Dataset (International Cardiac Arrest Research Consortium)

**Working title:** Predicting Neurological Outcomes After Cardiac Arrest Using EEG Network Collapse Analysis

**Research goal:** Analyze EEG recordings from comatose cardiac arrest patients and determine whether brain connectivity networks derived from EEG signals can predict neurological outcomes. The study models the brain as a functional network of interacting regions using EEG electrode connectivity; connectivity graphs are constructed from EEG signals and graph-theory metrics used to quantify network structure. These network features feed machine learning models for outcome prediction.

---

## Dataset Contents

- Continuous EEG recordings
- Clinical metadata
- Neurological outcome labels

**Subset used in this project:** 294 patients

**Outcome labels:**
- **Good outcome:** CPC 1–2
- **Poor outcome:** CPC 3–5

---

## Metadata

**Metadata file location (Google Drive):**  
`/content/drive/MyDrive/icare_project/metadata_clean.csv`

**Fields available:**  
`patient_id`, `Age`, `CPC`, `Hospital`, `OHCA`, `Outcome`, `Patient`, `ROSC`, `Sex`, `Shockable Rhythm`, `TTM`

---

## Raw EEG Data Location

**Root directory:**  
`/content/drive/MyDrive/icare_project/`

**Raw EEG recordings:**  
`/content/drive/MyDrive/icare_project/data/raw/eeg/`

**Structure:**
- Each patient has a folder named by **patient ID** (e.g. `0284`, `0286`).
- EEG files **never** appear directly under the `eeg` root; they always lie inside patient folders.

**Example:**

```
data/raw/eeg/
├── 0284/
│     ├── 0284_001_000_EEG.mat
│     ├── 0284_001_000_EEG.hea
│     ├── 0284_002_001_EEG.mat
│     ├── 0284_002_001_EEG.hea
│     ...
├── 0286/
│     ├── 0286_001_000_EEG.mat
│     ├── 0286_001_000_EEG.hea
│     ...
```

**File types:**
- **`.mat`** — EEG signal data
- **`.hea`** — WFDB header (metadata and channel information)

**Patient IDs** are numeric strings (e.g. `"0284"`, `"0286"`). They must be treated as **strings**, not integers, and correspond to directory names under `data/raw/eeg/`.

---

## EEG Segment Information

- Each segment corresponds to approximately **1 hour** of recording.
- Patients may have different numbers of segments.
- **Analysis cap:** 48 hours of EEG per patient. Segments beyond 48 hours are ignored.

**Example:**  
`segments_downloaded = 85` → `segments_used_for_analysis = 48`

**Rule:**  
`segments_used_for_analysis = min(total_segments_available, 48)`

**Segment file naming:**  
Pairs of files per segment, e.g.  
`0284_001_000_EEG.mat` / `0284_001_000_EEG.hea`  
`0284_002_001_EEG.mat` / `0284_002_001_EEG.hea`  
Segments are sequential.

---

## Batch Processing System

The EEG dataset was downloaded in batches (bandwidth limits). Batch definitions live in Google Drive.

**Batch folder:**  
`/content/drive/MyDrive/icare_project/batches/`

**Files:**
- `batch_1.csv`, `batch_2.csv`, `batch_3.csv`, `batch_4.csv`
- **Batch 0** uses: `subset_100_ids_updated.csv`  
  Location: `/content/drive/MyDrive/icare_project/subset_100_ids_updated.csv`

**Batch CSV structure:**
- **No header row.**
- Single column: one patient ID per row.

**Example:**

```
0284
0286
0296
0299
0303
...
```

**Mapping:**  
Patient ID `0284` → EEG files in  
`/content/drive/MyDrive/icare_project/data/raw/eeg/0284/`

**Patients actually downloaded per batch:**
- batch_0 → 100 patients
- batch_1 → 54 patients
- batch_2 → 72 patients
- batch_3 → 45 patients
- batch_4 → 23 patients  

**Total with EEG data:** 294 (only these should be processed).

---

## Canonical Patient List and Balanced Splits

A **canonical patient list** consolidates all 294 patients with downloaded EEG:

**File:**  
`/content/drive/MyDrive/icare_project/batches/all_downloaded_patients_294.csv`

**Format:** CSV with header row. Single column: **`patient_id`**. One row per patient. Patient IDs are strings (e.g. `"0284"`, `"0286"`). This is the source of truth for the channel consistency analysis and for full-dataset runs.

**Balanced compute splits** (for parallel Colab workers):

- `.../batches/patient_split_1.csv`
- `.../batches/patient_split_2.csv`
- `.../batches/patient_split_3.csv`

Each split contains approximately **98 patients**, with header **`patient_id`**. Splits are **balanced** by:

- Hospital distribution  
- Outcome distribution (Good vs Poor)  
- Sex distribution  

These splits are used later for parallel workers; the channel consistency stage uses the **canonical** list to compute the global channel intersection across all 294 patients.

---

## Audit Dataset

**Location:**  
`/content/drive/MyDrive/icare_project/analysis/selected_patient_metadata_audit.csv`

**Columns:**  
`patient_id`, `batch`, `Age`, `Sex`, `Hospital`, `CPC`, `Outcome`, `Shockable_Rhythm`, `TTM`, `segments_downloaded_total`, `segments_used_for_analysis`, `usable_eeg_hours`, `has_eeg_data`, `eeg_time_window`

---

## EEG Time Window Distribution

Patients by EEG window length:

| Window       | Count |
|-------------|-------|
| 0–6 hours   | 10    |
| 6–12 hours  | 5     |
| 12–24 hours | 35    |
| 24–48 hours | 244   |

Early windows are small; analysis uses **0–48 hours** EEG per patient.

---

## Data Storage Rules

- The **repository** must never contain: EEG files, metadata CSVs, batch CSVs, or any patient-identifiable data.
- All data remains in **Google Drive**.
- The repository holds only: code, configuration, and documentation.
