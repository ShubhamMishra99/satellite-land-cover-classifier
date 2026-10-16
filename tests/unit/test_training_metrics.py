"""Unit tests for training metrics, loss functions, and evaluation scoring."""

import numpy as np
import pytest
import torch

from src.constants import EUROSAT_CLASSES, NUM_CLASSES
from src.training.losses import get_loss_fn
from src.training.metrics import calculate_metrics, get_per_class_f1


def test_calculate_metrics_perfect_predictions():
    """Verify metrics calculation when predictions exactly match ground truth."""
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_calculate_metrics_known_example():
    """Verify metrics calculation on a controlled synthetic toy scenario."""
    # 4 samples: 2 correct, 2 incorrect across 10 classes
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 3, 2]

    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.5
    assert 0.0 <= metrics["macro_precision"] <= 1.0
    assert 0.0 <= metrics["macro_recall"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_calculate_metrics_tensor_inputs():
    """Verify metrics calculation accepts PyTorch Tensor inputs directly."""
    y_true = torch.tensor([1, 2, 3, 4, 5])
    y_pred = torch.tensor([1, 2, 0, 4, 5])

    metrics = calculate_metrics(y_true, y_pred)
    assert isinstance(metrics["macro_f1"], float)
    assert metrics["accuracy"] == 0.8


def test_calculate_metrics_handles_missing_class_predictions():
    """Verify zero_division handling when certain classes have 0 predictions."""
    # Only class 0 is predicted for all samples
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.1
    assert 0.0 <= metrics["macro_f1"] <= 1.0


def test_calculate_metrics_length_mismatch():
    """Verify ValueError is raised if prediction and ground-truth lengths differ."""
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_metrics([0, 1, 2], [0, 1])


def test_calculate_metrics_empty():
    """Verify safe fallback for empty arrays."""
    metrics = calculate_metrics([], [])
    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0


def test_get_per_class_f1():
    """Verify per-class F1 dictionary contains all 10 canonical EuroSAT classes."""
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    class_f1 = get_per_class_f1(y_true, y_pred)
    assert len(class_f1) == 10
    for cls_name in EUROSAT_CLASSES:
        assert cls_name in class_f1
        assert class_f1[cls_name] == 1.0


def test_get_loss_fn_cross_entropy():
    """Verify get_loss_fn creates CrossEntropyLoss."""
    criterion = get_loss_fn("cross_entropy")
    assert isinstance(criterion, torch.nn.CrossEntropyLoss)

    with pytest.raises(ValueError, match="Unsupported loss function"):
        get_loss_fn("unsupported_loss")
