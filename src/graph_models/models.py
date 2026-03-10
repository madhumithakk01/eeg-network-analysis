"""
Dynamic graph temporal model: graph encoder per window + LSTM over time.

Operates on raw connectivity matrices (batch, T, 19, 19). Each window is encoded by a
graph convolution module that preserves topology; then a temporal module models evolution.
Supports variable length via masking; padded timesteps do not affect the output.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .dataset import N_CHANNELS


def _normalized_adjacency(A: torch.Tensor) -> torch.Tensor:
    """A: (batch, n, n). Add self-loops and symmetric normalize. Returns (batch, n, n)."""
    batch, n, _ = A.shape
    I = torch.eye(n, device=A.device, dtype=A.dtype).unsqueeze(0).expand(batch, -1, -1)
    A = A + I
    deg = A.sum(dim=2).clamp(min=1e-6)
    d_inv_sqrt = deg.pow(-0.5)
    return A * d_inv_sqrt.unsqueeze(2) * d_inv_sqrt.unsqueeze(1)


class GraphEncoder(nn.Module):
    """
    Encodes a single connectivity matrix (batch, 19, 19) into a graph embedding (batch, hidden).
    Uses two layers of normalized adjacency propagation; node features = rows of adjacency.
    """

    def __init__(
        self,
        in_dim: int = N_CHANNELS,
        hidden_dim: int = 64,
        out_dim: int = 64,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        A: (batch, n, n). Node features X = A (each node's row). Output: (batch, out_dim).
        """
        batch, n, _ = A.shape
        A_norm = _normalized_adjacency(A)
        X = A
        X = torch.relu(A_norm @ self.w1(X))
        X = torch.relu(A_norm @ self.w2(X))
        return X.mean(dim=1)


class DynamicGraphTemporalModel(nn.Module):
    """
    Graph encoder per time step -> sequence of embeddings -> LSTM -> classifier.
    Input: (batch, T, 19, 19), mask (batch, T). Output: (batch, 2) logits.
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        graph_hidden: int = 64,
        graph_embed: int = 64,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.graph_encoder = GraphEncoder(
            in_dim=n_channels,
            hidden_dim=graph_hidden,
            out_dim=graph_embed,
        )
        self.lstm = nn.LSTM(
            input_size=graph_embed,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(
        self,
        conn: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        conn: (batch, T, 19, 19)
        mask: (batch, T), 1 = valid, 0 = pad
        Returns: (batch, 2) logits.
        """
        batch, T, n, _ = conn.shape
        conn_flat = conn.view(batch * T, n, n)
        emb = self.graph_encoder(conn_flat)
        emb = emb.view(batch, T, -1)
        if mask is not None:
            emb = emb * mask.unsqueeze(2)
        out, (h_n, _) = self.lstm(emb)
        if mask is not None and mask.any():
            lengths = mask.sum(dim=1).long().clamp(min=1)
            batch_idx = torch.arange(batch, device=conn.device)
            last_idx = (lengths - 1).clamp(min=0)
            last_h = out[batch_idx, last_idx]
        else:
            last_h = out[:, -1]
        return self.fc(last_h)
