"""
PyTorch Dataset for window-level graph feature sequences per patient.

Loads (n_windows, 40) from *_features.npy with no truncation; sequences are variable length.
Padding is applied only at batch time in collate_patient_batch so that no temporal information is discarded.
Returns (seq [T, F], mask [T], label); mask 1 = valid timestep, 0 = padding (used only after collate).
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# Graph feature dimension (must match feature_extractor output).
N_GRAPH_FEATURES = 40

# Minimum windows to include a patient (align with temporal_analysis).
MIN_WINDOWS = 10


def validate_feature_file(path: str, min_windows: int = MIN_WINDOWS, n_features: int = N_GRAPH_FEATURES) -> Tuple[bool, str]:
    """
    Verify feature file shape and minimum length. Returns (valid, message).
    """
    if not os.path.isfile(path):
        return False, "file_not_found"
    try:
        arr = np.load(path)
        if arr.ndim != 2:
            return False, f"ndim={arr.ndim}"
        if arr.shape[1] != n_features:
            return False, f"n_features={arr.shape[1]}"
        if arr.shape[0] < min_windows:
            return False, f"n_windows={arr.shape[0]}"
        return True, ""
    except Exception as e:
        return False, str(e)


class WindowLevelDataset(Dataset):
    """
    Dataset of patient sequences: each sample is (seq [T, F], mask [T], label).
    T is variable per patient; no truncation. Padding is applied in collate to max T in batch.
    """

    def __init__(
        self,
        patient_ids: List[str],
        labels: np.ndarray,
        graph_features_dir: str,
        min_windows: int = MIN_WINDOWS,
        n_features: int = N_GRAPH_FEATURES,
    ):
        self.patient_ids = list(patient_ids)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.graph_features_dir = graph_features_dir.rstrip("/")
        self.min_windows = min_windows
        self.n_features = n_features
        assert len(self.patient_ids) == len(self.labels)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        pid = self.patient_ids[idx]
        label = int(self.labels[idx])
        path = os.path.join(self.graph_features_dir, f"{pid}_features.npy")
        valid, _ = validate_feature_file(path, self.min_windows, self.n_features)
        if not valid:
            # Return minimal valid sequence so collate still works; will be skipped by mask
            features = np.zeros((self.min_windows, self.n_features), dtype=np.float32)
            mask = np.ones(self.min_windows, dtype=np.float32)
            return torch.from_numpy(features), torch.from_numpy(mask), label
        features = np.load(path)
        features = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        T = features.shape[0]
        mask = np.ones(T, dtype=np.float32)
        return (
            torch.from_numpy(features),
            torch.from_numpy(mask),
            label,
        )


def collate_patient_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stack (seq, mask, label) into batched tensors.
    Pads sequences to the maximum length within the batch; mask is 1 for valid, 0 for padding.
    Padded values must not influence the model (handled by masked pooling in the model).
    """
    max_len = max(b[0].size(0) for b in batch)
    n_features = batch[0][0].size(1)
    device_cpu = batch[0][0].device
    seqs_list = []
    masks_list = []
    for seq, mask, _ in batch:
        T = seq.size(0)
        if T < max_len:
            pad_seq = torch.zeros(max_len - T, n_features, dtype=seq.dtype, device=device_cpu)
            seq = torch.cat([seq, pad_seq], dim=0)
            pad_mask = torch.zeros(max_len - T, dtype=mask.dtype, device=device_cpu)
            mask = torch.cat([mask, pad_mask], dim=0)
        seqs_list.append(seq)
        masks_list.append(mask)
    seqs = torch.stack(seqs_list)
    masks = torch.stack(masks_list)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return seqs, masks, labels
