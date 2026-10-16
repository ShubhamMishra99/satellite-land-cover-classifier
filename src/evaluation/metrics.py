"""Comprehensive metric computation and statistical evaluation for EuroSAT classifiers.

This module calculates overall and per-class classification metrics using
scikit-learn with explicit class ordering according to `src.constants.EUROSAT_CLASSES`.

Metrics calculated:
- Accuracy: Overall correct predictions / total samples.
- Macro Precision: Unweighted mean of per-class precisions.
- Macro Recall: Unweighted mean of per-class recalls.
- Macro F1-score: Unweighted mean of per-class F1-scores (primary model-selection metric).
- Weighted F1-score: Support-weighted mean of per-class F1-scores.
- Per-class Precision, Recall, F1-score, and Support.
- Confusion Matrices: Raw counts and normalized distributions.
- Top Misclassification Pairs: Most frequent confusion pairs.

Zero Division Policy:
- If a class has zero predicted samples (precision denominator = 0) or zero true samples
  (recall denominator = 0), `zero_division=0.0` is used to return 0.0 rather than raising
  an error or generating NaN values.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch

from src.constants import EUROSAT_CLASSES, NUM_CLASSES


def to_numpy_1d(data: Union[np.ndarray, torch.Tensor, Sequence[int]]) -> np.ndarray:
    """Safely convert tensor or sequence to a 1D int64 numpy array."""
    if isinstance(data, torch.Tensor):
        if data.requires_grad:
            data = data.detach()
        return data.cpu().numpy().ravel().astype(np.int64)
    elif isinstance(data, np.ndarray):
        return data.ravel().astype(np.int64)
    else:
        return np.array(data, dtype=np.int64).ravel()


def calculate_overall_metrics(
    y_true: Union[np.ndarray, torch.Tensor, Sequence[int]],
    y_pred: Union[np.ndarray, torch.Tensor, Sequence[int]],
    labels: Optional[Sequence[int]] = None,
    zero_division: float = 0.0,
) -> Dict[str, float]:
    """Calculate overall classification metrics with explicit label ordering.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        labels: Explicit sequence of class indices (default: 0..NUM_CLASSES-1).
        zero_division: Value returned when division by zero occurs (default: 0.0).

    Returns:
        Dictionary containing accuracy, macro_precision, macro_recall,
        macro_f1, and weighted_f1.
    """
    targets = to_numpy_1d(y_true)
    predictions = to_numpy_1d(y_pred)

    if len(targets) != len(predictions):
        raise ValueError(
            f"Length mismatch: y_true has {len(targets)} samples, y_pred has {len(predictions)} samples."
        )

    if len(targets) == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
        }

    label_list = list(labels) if labels is not None else list(range(NUM_CLASSES))

    acc = float(accuracy_score(targets, predictions))
    macro_prec = float(
        precision_score(targets, predictions, labels=label_list, average="macro", zero_division=zero_division)
    )
    macro_rec = float(
        recall_score(targets, predictions, labels=label_list, average="macro", zero_division=zero_division)
    )
    macro_f1 = float(
        f1_score(targets, predictions, labels=label_list, average="macro", zero_division=zero_division)
    )
    weighted_f1 = float(
        f1_score(targets, predictions, labels=label_list, average="weighted", zero_division=zero_division)
    )

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
    }


def calculate_per_class_metrics(
    y_true: Union[np.ndarray, torch.Tensor, Sequence[int]],
    y_pred: Union[np.ndarray, torch.Tensor, Sequence[int]],
    class_names: Sequence[str] = EUROSAT_CLASSES,
    zero_division: float = 0.0,
) -> Dict[str, Dict[str, Union[float, int]]]:
    """Calculate precision, recall, F1-score, and support for each class.

    Preserves exact class order as specified by `class_names`.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Sequence of class names defining order and length.
        zero_division: Value returned when division by zero occurs (default: 0.0).

    Returns:
        Dictionary mapping class_name -> {"precision": float, "recall": float, "f1_score": float, "support": int}.
    """
    targets = to_numpy_1d(y_true)
    predictions = to_numpy_1d(y_pred)

    if len(targets) != len(predictions):
        raise ValueError(
            f"Length mismatch: y_true has {len(targets)} samples, y_pred has {len(predictions)} samples."
        )

    labels = list(range(len(class_names)))

    precisions = precision_score(
        targets, predictions, labels=labels, average=None, zero_division=zero_division
    )
    recalls = recall_score(
        targets, predictions, labels=labels, average=None, zero_division=zero_division
    )
    f1s = f1_score(
        targets, predictions, labels=labels, average=None, zero_division=zero_division
    )

    # Calculate support per class
    supports = [int(np.sum(targets == idx)) for idx in labels]

    per_class: Dict[str, Dict[str, Union[float, int]]] = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(precisions[idx]), 4),
            "recall": round(float(recalls[idx]), 4),
            "f1_score": round(float(f1s[idx]), 4),
            "support": supports[idx],
        }

    return per_class


def calculate_confusion_matrix(
    y_true: Union[np.ndarray, torch.Tensor, Sequence[int]],
    y_pred: Union[np.ndarray, torch.Tensor, Sequence[int]],
    class_names: Sequence[str] = EUROSAT_CLASSES,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Calculate confusion matrix for predictions.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Sequence of class names defining order and length.
        normalize: Normalization mode ('pred', 'true', 'all', or None).
                   'true' normalizes over ground truth (rows sum to 1.0).

    Returns:
        2D numpy array of shape [len(class_names), len(class_names)].
    """
    targets = to_numpy_1d(y_true)
    predictions = to_numpy_1d(y_pred)

    labels = list(range(len(class_names)))
    cm = confusion_matrix(targets, predictions, labels=labels, normalize=normalize)
    return cm


def find_top_confusion_pairs(
    y_true: Union[np.ndarray, torch.Tensor, Sequence[int]],
    y_pred: Union[np.ndarray, torch.Tensor, Sequence[int]],
    class_names: Sequence[str] = EUROSAT_CLASSES,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Identify the most frequent misclassification pairs (True != Predicted).

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Sequence of canonical class names.
        top_n: Maximum number of confusion pairs to return.

    Returns:
        List of dicts sorted descending by count:
        [{"true_class": str, "predicted_class": str, "count": int, "pct_of_errors": float}]
    """
    targets = to_numpy_1d(y_true)
    predictions = to_numpy_1d(y_pred)

    mask = targets != predictions
    total_errors = int(np.sum(mask))

    if total_errors == 0:
        return []

    err_true = targets[mask]
    err_pred = predictions[mask]

    pair_counts: Dict[Tuple[int, int], int] = {}
    for t, p in zip(err_true, err_pred):
        pair = (int(t), int(p))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    sorted_pairs = sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)

    results: List[Dict[str, Any]] = []
    for (t_idx, p_idx), count in sorted_pairs[:top_n]:
        results.append({
            "true_class": class_names[t_idx],
            "predicted_class": class_names[p_idx],
            "count": count,
            "pct_of_errors": round(float(count / total_errors * 100), 2),
        })

    return results


def calculate_all_metrics(
    y_true: Union[np.ndarray, torch.Tensor, Sequence[int]],
    y_pred: Union[np.ndarray, torch.Tensor, Sequence[int]],
    class_names: Sequence[str] = EUROSAT_CLASSES,
    zero_division: float = 0.0,
) -> Dict[str, Any]:
    """Calculate complete metric suite (overall, per-class, confusion matrices, error pairs).

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        class_names: Sequence of class names.
        zero_division: Zero division fill value.

    Returns:
        Nested dictionary with overall, per_class, raw_confusion_matrix,
        normalized_confusion_matrix, and top_confusion_pairs.
    """
    targets = to_numpy_1d(y_true)
    predictions = to_numpy_1d(y_pred)

    overall = calculate_overall_metrics(targets, predictions, zero_division=zero_division)
    per_class = calculate_per_class_metrics(targets, predictions, class_names=class_names, zero_division=zero_division)
    cm_raw = calculate_confusion_matrix(targets, predictions, class_names=class_names, normalize=None)
    cm_norm = calculate_confusion_matrix(targets, predictions, class_names=class_names, normalize="true")
    top_pairs = find_top_confusion_pairs(targets, predictions, class_names=class_names)

    # Determine highest and lowest F1 classes
    f1_sorted = sorted(per_class.items(), key=lambda item: item[1]["f1_score"], reverse=True)
    highest_f1_classes = [{"class": k, "f1_score": v["f1_score"]} for k, v in f1_sorted[:3]]
    lowest_f1_classes = [{"class": k, "f1_score": v["f1_score"]} for k, v in f1_sorted[-3:]]

    return {
        "overall": overall,
        "per_class": per_class,
        "raw_confusion_matrix": cm_raw.tolist(),
        "normalized_confusion_matrix": np.round(cm_norm, 4).tolist(),
        "top_confusion_pairs": top_pairs,
        "highest_f1_classes": highest_f1_classes,
        "lowest_f1_classes": lowest_f1_classes,
        "total_samples": len(targets),
        "total_errors": int(np.sum(targets != predictions)),
        "error_rate": round(float(np.mean(targets != predictions)), 4),
    }
