"""Metrics helpers for binary classifiers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    binary_preds = (preds > 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, binary_preds)),
        "precision": float(precision_score(labels, binary_preds, zero_division=0)),
        "recall": float(recall_score(labels, binary_preds, zero_division=0)),
        "f1": float(f1_score(labels, binary_preds, zero_division=0)),
    }
