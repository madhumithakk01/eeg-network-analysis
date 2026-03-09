# Pipeline Design: EEG Network Analysis for Neurological Outcome Prediction

## Purpose

This document describes the **planned** computational pipeline for the I-CARE EEG project. It supports reproducible research, parallel compute on Google Colab, and version-controlled pipeline stages. No implementation code is specified here—only architecture and planning.

---

## Pipeline Stages (Planned)

### 1. EEG Channel Consistency Analysis Across Hospitals

**Purpose:** Determine which EEG electrodes exist consistently across patients and hospitals.

**Inputs:** Raw EEG headers/metadata; hospital or recording metadata.  
**Outputs:** Channel list or mask defining electrodes to use in downstream steps.  
**Location:** Outputs may be written to `intermediate/` (see configuration).

---

### 2. EEG Preprocessing

**Purpose:** Standardize signals for connectivity and graph analysis.

**Parameters (from config):**
- **Bandpass filter:** 0.5 Hz – 40 Hz (`BANDPASS_LOW`, `BANDPASS_HIGH`)
- **Segment size:** 30 seconds (`WINDOW_SECONDS`)

**Inputs:** Raw EEG (e.g. WFDB `.mat`/`.hea`); channel consistency result.  
**Outputs:** Filtered, segmented time series per patient/window.  
**Location:** `intermediate/`.

---

### 3. Functional Connectivity Computation

**Purpose:** Build connectivity matrices from EEG signals.

**Metric (planned):** Pearson correlation between electrodes.

**Inputs:** Preprocessed EEG segments.  
**Outputs:** Connectivity matrix per window (and optionally per patient aggregate).  
**Location:** `intermediate/` or `features/`.

---

### 4. Brain Network Construction

**Purpose:** Represent the brain as a graph for graph-theory analysis.

- **Nodes:** EEG electrodes  
- **Edges:** Connectivity strength (from step 3)

**Inputs:** Connectivity matrices.  
**Outputs:** Graph representations (e.g. adjacency or weight matrices).  
**Location:** `intermediate/` or `features/`.

---

### 5. Graph-Theory Feature Extraction

**Purpose:** Quantify network structure for ML.

**Metrics (planned):**
- Clustering coefficient
- Global efficiency
- Network density
- Average path length
- Node centrality (and related measures)

**Inputs:** Brain networks from step 4.  
**Outputs:** Feature vectors/tables per patient or per window.  
**Location:** `features/`.

---

### 6. Machine Learning Prediction

**Purpose:** Predict neurological outcome (Good: CPC 1–2 vs Poor: CPC 3–5).

**Models (planned):**
- Random Forest
- XGBoost
- LightGBM

**Inputs:** Graph features; outcome labels (from metadata/audit).  
**Outputs:** Trained models; evaluation metrics; optional predictions.  
**Location:** `models/` and `logs/`.

---

## Data Flow Summary

```
Raw EEG (Drive) → Preprocessing → Connectivity → Graph construction
       → Graph features → ML models → Predictions / evaluation
```

All inputs read from Google Drive; all generated outputs written to Google Drive (features, intermediate, models, logs). Repository contains only code and config.

---

## Repository Modules and Pipeline Mapping

| Stage                      | Primary module(s)   | Script / entry point        |
|---------------------------|---------------------|-----------------------------|
| Channel consistency       | `src/data_loading`  | (TBD)                      |
| Preprocessing             | `src/preprocessing` | (TBD)                      |
| Connectivity              | `src/connectivity`  | (TBD)                      |
| Graph construction/feat.  | `src/graph_features`| `scripts/run_feature_extraction.py` |
| ML training / evaluation  | `src/modeling`      | `scripts/run_model_training.py`     |

Utility and configuration are in `src/utils` and `configs/`.

---

## Experiment Tracking

- **Feature runs:** `experiments/feature_runs/` — logs, config snapshots, or run IDs for feature extraction.
- **Model runs:** `experiments/model_runs/` — logs, config snapshots, or run IDs for training/evaluation.

No data or large artifacts are stored in the repository; only references and metadata.

---

## Parallel Compute

The pipeline is designed so that:

1. Patient lists can be split into non-overlapping ranges (e.g. 1–100, 101–200, 201–294).
2. Each Colab worker runs the same codebase (clone from GitHub) and reads/writes only on Google Drive.
3. Workers use a shared config (paths, parameters) with optional overrides (e.g. worker ID, patient range).
4. Outputs are written to shared Drive paths so that a single downstream process can aggregate features or models if needed.

See **Colab Worker Setup** (`docs/colab_worker_setup.md`) for step-by-step worker configuration.
