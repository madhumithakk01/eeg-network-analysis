"""
Training and patient-level cross-validation for connectivity graph DL models.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .dataset import (
    ConnectivitySequenceDataset,
    collate_connectivity_batch,
)
from .models import DynamicGraphTemporalModel


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for conns, masks, labels in loader:
        conns = conns.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(conns, masks)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * conns.size(0)
        n += conns.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []
    for conns, masks, labels in loader:
        conns = conns.to(device)
        masks = masks.to(device)
        logits = model(conns, masks)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
    y_true = np.concatenate(all_labels)
    y_proba = np.concatenate(all_probs)
    y_pred = (y_proba >= 0.5).astype(np.int64)
    return y_true, y_proba, y_pred


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    try:
        roc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc = 0.5
    return {
        "roc_auc": float(roc),
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(sens),
        "specificity": float(spec),
    }


def run_patient_cv(
    patient_ids: List[str],
    y: np.ndarray,
    sparse_connectivity_dir: str,
    stride: int = 8,
    n_splits: int = 5,
    batch_size: int = 8,
    epochs: int = 40,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    output_dir: Optional[str] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Stratified K-Fold at patient level. Train connectivity graph model per fold.
    Full sequences are used; no truncation. Stride optionally reduces temporal resolution.
    """
    set_seeds(random_state)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patient_ids = np.array(patient_ids)
    y = np.asarray(y, dtype=np.int64)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics: List[Dict[str, float]] = []
    all_fold_results: List[Dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(patient_ids, y)):
        train_ids = patient_ids[train_idx].tolist()
        val_ids = patient_ids[val_idx].tolist()
        y_train = y[train_idx]
        y_val = y[val_idx]
        n_pos = max(1, int(y_train.sum()))
        n_neg = max(1, len(y_train) - n_pos)
        weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32).to(device)

        train_ds = ConnectivitySequenceDataset(
            train_ids,
            y_train,
            sparse_connectivity_dir,
            stride=stride,
        )
        val_ds = ConnectivitySequenceDataset(
            val_ids,
            y_val,
            sparse_connectivity_dir,
            stride=stride,
        )
        g = torch.Generator()
        g.manual_seed(random_state + fold)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_connectivity_batch,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            generator=g,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_connectivity_batch,
            num_workers=0,
        )

        model = DynamicGraphTemporalModel(
            n_channels=19,
            graph_hidden=64,
            graph_embed=64,
            lstm_hidden=64,
            lstm_layers=2,
            dropout=0.3,
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_roc = 0.0
        best_metrics: Optional[Dict[str, float]] = None
        best_y_true: Optional[np.ndarray] = None
        best_y_proba: Optional[np.ndarray] = None
        for ep in range(epochs):
            train_epoch(model, train_loader, criterion, optimizer, device)
            y_v, p_v, pred_v = evaluate(model, val_loader, device)
            m = compute_metrics(y_v, pred_v, p_v)
            if m["roc_auc"] > best_roc:
                best_roc = m["roc_auc"]
                best_metrics = m
                best_y_true = y_v
                best_y_proba = p_v
        if best_metrics is None:
            y_v, p_v, pred_v = evaluate(model, val_loader, device)
            best_metrics = compute_metrics(y_v, pred_v, p_v)
            best_y_true = y_v
            best_y_proba = p_v
        best_metrics["fold"] = fold
        fold_metrics.append(best_metrics)
        all_fold_results.append({
            "fold": fold,
            "metrics": best_metrics,
            "y_true": best_y_true.tolist() if best_y_true is not None else [],
            "y_proba": best_y_proba.tolist() if best_y_proba is not None else [],
        })
        print(
            f"  Fold {fold + 1}/{n_splits} ROC-AUC={best_metrics['roc_auc']:.4f} "
            f"F1={best_metrics['f1']:.4f} Sens={best_metrics['sensitivity']:.4f} "
            f"Spec={best_metrics['specificity']:.4f}"
        )

    mean_metrics = {
        "mean_roc_auc": float(np.mean([m["roc_auc"] for m in fold_metrics])),
        "std_roc_auc": float(np.std([m["roc_auc"] for m in fold_metrics])),
        "mean_f1": float(np.mean([m["f1"] for m in fold_metrics])),
        "mean_sensitivity": float(np.mean([m["sensitivity"] for m in fold_metrics])),
        "mean_specificity": float(np.mean([m["specificity"] for m in fold_metrics])),
        "mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out = {
            "fold_metrics": fold_metrics,
            "summary": mean_metrics,
            "fold_predictions": [
                {"fold": r["fold"], "y_true": r["y_true"], "y_proba": r["y_proba"]}
                for r in all_fold_results
            ],
        }
        with open(os.path.join(output_dir, "connectivity_dl_cv_metrics.json"), "w") as f:
            json.dump(out, f, indent=2)
    return {"fold_metrics": fold_metrics, "summary": mean_metrics, "all_folds": all_fold_results}
