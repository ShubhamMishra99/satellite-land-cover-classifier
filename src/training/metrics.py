"""Evaluation metrics and scoring functions for EuroSAT classification.

Primary Model-Selection Metric: Macro F1
$$\text{Macro } F_1 = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
Where $C=10$. Macro F1 treats all 10 land-cover classes equally, ensuring that
overall performance is not distorted by majority class dominance.
"""

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import torch

from src.constants import EUROSAT_CLASSES, NUM_CLASSES


def _to_numpy_array(data: Union[np.ndarray, torch.Tensor, Sequence[int]]) -> np.ndarray:
    """Convert input tensor or sequence safely to 1D integer numpy array."""
    if isinstance(data, torch.Tensor):
        if data.requires_grad:
            data = data.detach()
        return data.cpu().numpy().ravel().astype(np.int64)
    elif isinstance(data, np.ndarray):
        return data.ravel().astype(np.int64)
    else:
        return np.array(data, dtype=np.int64).ravel()


def calculate_metrics(
    y_true: Union[np.ndarray, torch.Tensor, Sequence[int]],
    y_pred: Union[np.ndarray, torch.Tensor, Sequence[int]],
    labels: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics for classification.

    Computes:
        - accuracy: Overall classification accuracy.
        - macro_precision: Unweighted mean of class-wise precision.
        - macro_recall: Unweighted mean of class-wise recall.
        - macro_f1: Unweighted mean of class-wise F1 score (PRIMARY METRIC).

    Args:
        y_true: Ground-truth class labels.
        y_pred: Predicted class labels.
        labels: Optional explicit list of class indices (default: 0..num_classes-1).

    Returns:
        Dictionary mapping metric names to float scores in range [0.0, 1.0].
    """
    targets = _to_numpy_array(y_true)
    predictions = _to_numpy_array(y_pred)

    if len(targets) != len(predictions):
        raise ValueError(
            f"Length mismatch: y_true has {len(targets)} items, y_pred has {len(predictions)} items."
        )

    if len(targets) == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }

    all_labels = labels if labels is not None else list(range(NUM_CLASSES))

    acc = float(accuracy_score(targets, predictions))
    macro_prec = float(
        precision_score(targets, predictions, labels=all_labels, average="macro", zero_division=0)
    )
    macro_rec = float(
        recall_score(targets, predictions, labels=all_labels, average="macro", zero_division=0)
    )
    macro_f1 = float(
        f1_score(targets, predictions, labels=all_labels, average="macro", zero_division=0)
    )

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
    }


def get_per_class_f1(
    y_true: Union[np.ndarray, torch.Tensor, Sequence[int]],
    y_pred: Union[np.ndarray, torch.Tensor, Sequence[int]],
    class_names: Sequence[str] = EUROSAT_CLASSES,
) -> Dict[str, float]:
    """Calculate per-class F1 scores for in-depth error diagnosis."""
    targets = _to_numpy_array(y_true)
    predictions = _to_numpy_array(y_pred)
    all_labels = list(range(len(class_names)))

    f1_per_class = f1_score(
        targets, predictions, labels=all_labels, average=None, zero_division=0
    )

    return {
        class_names[idx]: round(float(f1_per_class[idx]), 4)
        for idx in range(len(class_names))
    }
