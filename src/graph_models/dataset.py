"""
Dataset for patient-level connectivity graph sequences.

Loads {patient_id}_sparse.npy from SPARSE_CONNECTIVITY_DIR, shape (n_windows, 19, 19).
Returns (connectivity_sequence, temporal_mask, label). Full sequence is loaded;
optional temporal stride reduces resolution while preserving full recording duration.
No truncation is applied.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Connectivity matrix dimensions (19 EEG channels).
N_CHANNELS = 19
MIN_WINDOWS = 10


def validate_connectivity_file(
    path: str,
    min_windows: int = MIN_WINDOWS,
    n_channels: int = N_CHANNELS,
) -> Tuple[bool, str]:
    """Verify connectivity file shape. Returns (valid, message)."""
    if not os.path.isfile(path):
        return False, "file_not_found"
    try:
        arr = np.load(path)
        if arr.ndim != 3:
            return False, f"ndim={arr.ndim}"
        if arr.shape[1] != n_channels or arr.shape[2] != n_channels:
            return False, f"shape={arr.shape}"
        if arr.shape[0] < min_windows:
            return False, f"n_windows={arr.shape[0]}"
        return True, ""
    except Exception as e:
        return False, str(e)


class ConnectivitySequenceDataset(Dataset):
    """
    Each sample: (connectivity_sequence [T, 19, 19], temporal_mask [T], label).
    Full sequence is loaded; optional stride subsamples in time (preserves full duration).
    """

    def __init__(
        self,
        patient_ids: List[str],
        labels: np.ndarray,
        sparse_connectivity_dir: str,
        stride: int = 1,
        min_windows: int = MIN_WINDOWS,
        n_channels: int = N_CHANNELS,
    ):
        self.patient_ids = list(patient_ids)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.sparse_connectivity_dir = sparse_connectivity_dir.rstrip("/")
        self.stride = max(1, stride)
        self.min_windows = min_windows
        self.n_channels = n_channels
        assert len(self.patient_ids) == len(self.labels)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        pid = self.patient_ids[idx]
        label = int(self.labels[idx])
        path = os.path.join(self.sparse_connectivity_dir, f"{pid}_sparse.npy")
        valid, _ = validate_connectivity_file(path, self.min_windows, self.n_channels)
        if not valid:
            conn = np.zeros((self.min_windows, self.n_channels, self.n_channels), dtype=np.float32)
            mask = np.zeros(self.min_windows, dtype=np.float32)
            return torch.from_numpy(conn), torch.from_numpy(mask), label
        data = np.load(path)
        data = np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if self.stride > 1:
            data = data[:: self.stride]
        T = data.shape[0]
        mask = np.ones(T, dtype=np.float32)
        return (
            torch.from_numpy(data),
            torch.from_numpy(mask),
            label,
        )


def collate_connectivity_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad connectivity sequences to max T in batch. Returns (conn [B, T, 19, 19], mask [B, T], labels).
    """
    max_len = max(b[0].size(0) for b in batch)
    n_ch = batch[0][0].size(1)
    device_cpu = batch[0][0].device
    conn_list = []
    mask_list = []
    for conn, mask, _ in batch:
        T = conn.size(0)
        if T < max_len:
            pad = torch.zeros(max_len - T, n_ch, n_ch, dtype=conn.dtype, device=device_cpu)
            conn = torch.cat([conn, pad], dim=0)
            pad_mask = torch.zeros(max_len - T, dtype=mask.dtype, device=device_cpu)
            mask = torch.cat([mask, pad_mask], dim=0)
        conn_list.append(conn)
        mask_list.append(mask)
    conns = torch.stack(conn_list)
    masks = torch.stack(mask_list)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return conns, masks, labels
