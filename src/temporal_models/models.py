"""
Temporal Convolutional Network over window-level graph features for patient-level outcome prediction.

Operates on (batch, time_steps, 40). Uses 1D convolutions along the time dimension with
multi-scale receptive fields. Masked pooling ensures padded timesteps never influence the output.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .dataset import N_GRAPH_FEATURES


def masked_global_avg_pool(
    x: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Global average pooling over the time dimension, ignoring padded positions.
    x: (batch, channels, time)
    mask: (batch, time), 1 = valid, 0 = padding
    Returns: (batch, channels)
    """
    # x (B, C, T), mask (B, T) -> (B, 1, T)
    mask = mask.unsqueeze(1)
    x_masked = x * mask
    sum_t = x_masked.sum(dim=2)
    count = mask.sum(dim=2).clamp(min=1e-6)
    return sum_t / count


class TemporalCNN(nn.Module):
    """
    Temporal CNN: (batch, time_steps, n_features) -> (batch, 2) logits.
    Convolutions operate on time; padded timesteps are excluded via masked pooling.
    """

    def __init__(
        self,
        input_size: int = N_GRAPH_FEATURES,
        channels: tuple = (64, 64, 64, 32),
        kernel_sizes: tuple = (7, 5, 5, 3),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        assert len(channels) == len(kernel_sizes)
        layers = []
        in_c = input_size
        for i, (out_c, k) in enumerate(zip(channels, kernel_sizes)):
            pad = k // 2
            layers.append(nn.Conv1d(in_c, out_c, kernel_size=k, padding=pad))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_c = out_c
        self.conv = nn.Sequential(*layers)
        self.out_channels = in_c
        self.fc = nn.Sequential(
            nn.Linear(self.out_channels, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        mask: (batch, seq_len), 1 = valid, 0 = pad. Required for correct masked pooling.
        Returns: (batch, 2) logits.
        """
        # (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        # x (B, C, T). Pool only over valid steps.
        if mask is not None:
            out = masked_global_avg_pool(x, mask)
        else:
            out = x.mean(dim=2)
        return self.fc(out)
