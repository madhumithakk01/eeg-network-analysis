"""
Temporal deep learning models for neurological outcome prediction from window-level graph features.

Consumes existing *_features.npy (n_windows, 40) per patient. Patient-level splitting is enforced;
no windows from the same patient appear in both train and test.
"""

from __future__ import annotations

from .dataset import WindowLevelDataset
from .models import TemporalCNN
from .training import train_epoch, evaluate, run_patient_cv

__all__ = [
    "WindowLevelDataset",
    "TemporalCNN",
    "train_epoch",
    "evaluate",
    "run_patient_cv",
]
