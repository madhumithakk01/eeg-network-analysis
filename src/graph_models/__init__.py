"""
Deep learning on connectivity graphs: graph representation learning + temporal sequence modeling.

Learns directly from sequences of (19, 19) connectivity matrices. Does not use the 40 graph
feature vectors as primary input; preserves network structure and temporal evolution.
"""

from __future__ import annotations

from .dataset import ConnectivitySequenceDataset, collate_connectivity_batch
from .models import DynamicGraphTemporalModel
from .training import run_patient_cv

__all__ = [
    "ConnectivitySequenceDataset",
    "collate_connectivity_batch",
    "DynamicGraphTemporalModel",
    "run_patient_cv",
]
