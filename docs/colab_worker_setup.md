# Google Colab Worker Setup for Parallel EEG Processing

## Overview

The I-CARE EEG pipeline is designed to run on **multiple parallel Google Colab sessions**. Each session (worker) processes a **non-overlapping subset of patients** and reads/writes data only on Google Drive. The repository is cloned once per session; no data is stored in the GitHub repo.

---

## Prerequisites

- Google account(s); multiple accounts allow more parallel workers.
- Google Drive with the I-CARE project data mounted (see paths below).
- GitHub repository clone URL for this project.

---

## Data and Paths on Google Drive

Ensure the following exist under your project root (e.g. `/content/drive/MyDrive/icare_project/`):

| Path / role        | Location |
|--------------------|----------|
| Project root       | `/content/drive/MyDrive/icare_project/` |
| Raw EEG            | `.../data/raw/eeg/<patient_id>/` |
| Batches            | `.../batches/` (batch_1.csv, etc.) |
| Batch 0 IDs        | `.../subset_100_ids_updated.csv` |
| Metadata           | `.../metadata_clean.csv` |
| Audit              | `.../analysis/selected_patient_metadata_audit.csv` |
| Feature output     | `.../features/` |
| Intermediate       | `.../intermediate/` |
| Model output       | `.../models/` |
| Logs               | `.../logs/` |

Create `features/`, `intermediate/`, `models/`, and `logs/` if they do not exist.

---

## Worker Assignment Example

**Total patients:** 294 (string IDs, e.g. `"0284"`, `"0286"`).

Example split across three workers:

| Worker   | Patient index range | Description    |
|----------|---------------------|----------------|
| Worker 1 | 1–100               | First 100      |
| Worker 2 | 101–200             | Next 100       |
| Worker 3 | 201–294             | Remaining 94   |

Each worker must process **only** its assigned range so there is no overlap and no duplicate writes (unless explicitly designed for redundancy).

---

## Step-by-Step Setup in Each Colab Notebook

Run these steps in order in a new Colab notebook. Adjust the **worker index** and **patient range** per session.

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Confirm that `/content/drive/MyDrive/icare_project/` (and subdirs) are visible.

### 2. Clone the Repository

Clone the project repo into a fixed path (e.g. project root or a subfolder). Use the same branch/tag for all workers.

```bash
# Example (replace with your repo URL and desired path)
!git clone https://github.com/<org>/<repo>.git /content/icare_eeg_network
%cd /content/icare_eeg_network
```

Or clone into the Drive project folder if you prefer:

```bash
!git clone https://github.com/<org>/<repo>.git /content/drive/MyDrive/icare_project/icare_eeg_network
%cd /content/drive/MyDrive/icare_project/icare_eeg_network
```

### 3. Install Dependencies

Use the pinned versions from the repository for reproducibility:

```bash
!pip install -r requirements.txt
```

Do not upgrade packages unless you have tested and documented new versions.

### 4. Set Worker-Specific Options (Optional)

If the pipeline supports worker index and patient range via environment variables, set them before importing project code:

```python
import os
os.environ["WORKER_INDEX"] = "1"        # e.g. 1, 2, 3
os.environ["PATIENT_START_INDEX"] = "1"
os.environ["PATIENT_END_INDEX"] = "100"
```

Or pass these as arguments to the entry-point scripts (see repository scripts and config).

### 5. Import Project Modules

Ensure the project root is on `sys.path`, then import:

```python
import sys
sys.path.insert(0, "/content/icare_eeg_network")  # or your clone path

from configs.config import DATA_ROOT, EEG_RAW_ROOT, FEATURE_OUTPUT_PATH
# Import from src.data_loading, src.preprocessing, etc. as needed
```

### 6. Load Patient Subset

- Read the full list of 294 patient IDs from the audit file, batch files, or a single master list.
- Slice the list by this worker’s range (e.g. indices 0–99 for worker 1).
- Use only these patient IDs for loading EEG and writing outputs.

Patient IDs must be treated as **strings** and must match directory names under `data/raw/eeg/<patient_id>/`.

### 7. Run Processing

- **Feature extraction:** Run the feature pipeline (preprocessing → connectivity → graph features) for this worker’s patient subset. Write results under `FEATURE_OUTPUT_PATH` (and `INTERMEDIATE_OUTPUT_PATH` if used), with filenames or subfolders that include worker ID or patient ID to avoid overwrites.
- **Model training:** If a worker is used for training, use only the designated output paths (`MODEL_OUTPUT_PATH`, `LOG_PATH`) and avoid overwriting other workers’ outputs.

### 8. Write Outputs to Google Drive Only

All outputs go to Drive paths configured in `configs.config` (or overridden by environment variables). Do **not** write EEG data, features, or models into the cloned repository directory.

---

## Checklist for Each Worker

- [ ] Drive mounted at `/content/drive`
- [ ] Repo cloned; `requirements.txt` installed
- [ ] Worker index and patient range set (env or script args)
- [ ] Patient list loaded and subset to this worker’s range
- [ ] Output paths point to Drive (`features/`, `intermediate/`, `models/`, `logs/`)
- [ ] Processing script/notebook uses only this worker’s patient IDs
- [ ] No data committed to the GitHub repository

---

## Multiple Google Accounts

To run several workers in parallel:

1. Open Colab in different browsers or incognito windows, each logged into a different Google account.
2. Mount each account’s Drive; ensure each has access to the same I-CARE data (shared folder or copy).
3. Clone the same repository and use the same `requirements.txt` in each session.
4. Assign disjoint patient ranges (e.g. 1–100, 101–200, 201–294) so outputs do not conflict.

---

## Troubleshooting

- **Drive path not found:** Confirm mount path and that `icare_project` (and subdirs) exist under My Drive.
- **Patient folder missing:** Only the 294 patients with downloaded EEG should be processed; skip or log missing IDs.
- **Permission errors:** Ensure the Colab account has edit access to the Drive folders used for outputs.
- **Version conflicts:** Use exactly the versions in `requirements.txt`; avoid `pip install --upgrade` for project packages.

For dataset and pipeline details, see `docs/dataset_description.md` and `docs/pipeline_design.md`.
