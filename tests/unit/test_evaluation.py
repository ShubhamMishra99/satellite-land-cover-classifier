"""Unit tests for Milestone 5: Final Model Evaluation, Metrics, and Error Analysis."""

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.constants import EUROSAT_CLASSES, NUM_CLASSES
from src.evaluation.evaluator import (
    EvaluationResult,
    ModelEvaluator,
    compute_model_parameter_hash,
)
from src.evaluation.metrics import (
    calculate_all_metrics,
    calculate_confusion_matrix,
    calculate_overall_metrics,
    calculate_per_class_metrics,
    find_top_confusion_pairs,
    to_numpy_1d,
)
from src.evaluation.report import (
    build_model_comparison_dict,
    generate_error_analysis_csv,
    generate_markdown_report,
    generate_per_class_csv,
    save_confusion_matrix_csv,
)
from src.models.custom_cnn import CustomCNN


class ToyDataset(torch.utils.data.Dataset):
    """Small synthetic dataset yielding (tensor, label, metadata)."""

    def __init__(self, num_samples: int = 12):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, 3, 64, 64)
        self.labels = torch.tensor([i % NUM_CLASSES for i in range(num_samples)], dtype=torch.long)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        metadata = {
            "path": f"/path/to/mock_{idx}.jpg",
            "relative_path": f"mock/mock_{idx}.jpg",
            "class_name": EUROSAT_CLASSES[self.labels[idx].item()],
            "index": idx,
            "split": "test",
        }
        return self.data[idx], self.labels[idx], metadata


# =========================================================================
# 1. Metric Calculations
# =========================================================================


def test_metric_calculations_known_values():
    """Verify metrics calculation against mathematically known synthetic values."""
    # 4 samples: 2 of class 0, 2 of class 1
    # class 0: one predicted 0, one predicted 1
    # class 1: both predicted 1
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    overall = calculate_overall_metrics(y_true, y_pred, labels=[0, 1])

    # Accuracy: 3 / 4 = 0.75
    assert overall["accuracy"] == 0.75
    # Class 0: Precision = 1 / 1 = 1.0, Recall = 1 / 2 = 0.5, F1 = 2*(1*0.5)/(1+0.5) = 2/3 = 0.6667
    # Class 1: Precision = 2 / 3 = 0.6667, Recall = 2 / 2 = 1.0, F1 = 2*(0.6667*1)/(1.6667) = 0.8000
    # Macro Precision: (1.0 + 0.6667)/2 = 0.8333
    # Macro Recall: (0.5 + 1.0)/2 = 0.7500
    # Macro F1: (0.6667 + 0.8000)/2 = 0.7333
    assert np.isclose(overall["macro_precision"], 0.8333, atol=1e-3)
    assert np.isclose(overall["macro_recall"], 0.7500, atol=1e-3)
    assert np.isclose(overall["macro_f1"], 0.7333, atol=1e-3)


def test_per_class_metric_ordering():
    """Verify per-class metrics dictionary preserves exact EUROSAT_CLASSES canonical ordering."""
    # Create targets across all 10 classes
    y_true = list(range(NUM_CLASSES))
    y_pred = list(range(NUM_CLASSES))

    per_class = calculate_per_class_metrics(y_true, y_pred, class_names=EUROSAT_CLASSES)

    # Must contain exactly the 10 canonical classes
    assert len(per_class) == 10
    assert list(per_class.keys()) == list(EUROSAT_CLASSES)

    for idx, cls_name in enumerate(EUROSAT_CLASSES):
        assert per_class[cls_name]["support"] == 1
        assert per_class[cls_name]["precision"] == 1.0
        assert per_class[cls_name]["recall"] == 1.0
        assert per_class[cls_name]["f1_score"] == 1.0


def test_confusion_matrix_dimensions_and_class_ordering():
    """Verify confusion matrix shape is [10, 10] and sums to total sample count."""
    np.random.seed(42)
    n_samples = 50
    y_true = np.random.randint(0, NUM_CLASSES, size=n_samples)
    y_pred = np.random.randint(0, NUM_CLASSES, size=n_samples)

    cm_raw = calculate_confusion_matrix(y_true, y_pred, class_names=EUROSAT_CLASSES, normalize=None)
    assert cm_raw.shape == (NUM_CLASSES, NUM_CLASSES)
    assert int(np.sum(cm_raw)) == n_samples

    cm_norm = calculate_confusion_matrix(y_true, y_pred, class_names=EUROSAT_CLASSES, normalize="true")
    assert cm_norm.shape == (NUM_CLASSES, NUM_CLASSES)
    # Each row with support > 0 should sum to 1.0
    for row_idx in range(NUM_CLASSES):
        if np.sum(cm_raw[row_idx, :]) > 0:
            assert np.isclose(np.sum(cm_norm[row_idx, :]), 1.0, atol=1e-5)


def test_zero_division_safety():
    """Verify zero_division handling when certain classes have 0 samples or 0 predictions."""
    # Only class 0 is ever predicted, across all samples
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 0, 0, 0, 0]

    # Must execute safely without throwing ZeroDivisionError or generating NaNs
    metrics = calculate_overall_metrics(y_true, y_pred, zero_division=0.0)
    assert not np.isnan(metrics["macro_precision"])
    assert not np.isnan(metrics["macro_recall"])
    assert not np.isnan(metrics["macro_f1"])

    per_class = calculate_per_class_metrics(y_true, y_pred, zero_division=0.0)
    # Class 1 had 0 predictions, so precision is 0.0
    assert per_class[EUROSAT_CLASSES[1]]["precision"] == 0.0
    assert per_class[EUROSAT_CLASSES[1]]["f1_score"] == 0.0


def test_find_top_confusion_pairs():
    """Verify identification of most frequent misclassification pairs."""
    # True 0 predicted as 1 (3 times), True 2 predicted as 3 (1 time), others correct
    y_true = [0, 0, 0, 2, 4, 5]
    y_pred = [1, 1, 1, 3, 4, 5]

    pairs = find_top_confusion_pairs(y_true, y_pred, class_names=EUROSAT_CLASSES)
    assert len(pairs) == 2
    assert pairs[0]["true_class"] == EUROSAT_CLASSES[0]
    assert pairs[0]["predicted_class"] == EUROSAT_CLASSES[1]
    assert pairs[0]["count"] == 3
    assert pairs[0]["pct_of_errors"] == 75.0


# =========================================================================
# 2. Input Validation & Error Handling
# =========================================================================


def test_invalid_lengths_raise_value_error():
    """Verify ValueError is raised if y_true and y_pred have differing lengths."""
    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_overall_metrics([0, 1], [0])

    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_per_class_metrics([0, 1], [0, 1, 2])


def test_empty_inputs_handled_safely():
    """Verify empty inputs yield 0.0 metrics rather than crashing."""
    overall = calculate_overall_metrics([], [])
    assert overall["accuracy"] == 0.0
    assert overall["macro_f1"] == 0.0


# =========================================================================
# 3. Model Evaluation Mode & Gradient Disabling
# =========================================================================


def test_model_evaluation_mode_and_no_gradients():
    """Verify ModelEvaluator enforces eval() mode and computes no gradients."""
    model = CustomCNN(in_channels=3, num_classes=10)
    model.train()  # Intentionally put in train mode

    evaluator = ModelEvaluator(model=model, model_name="custom_cnn", device=torch.device("cpu"))
    assert not model.training, "Evaluator __init__ must set model to eval() mode"

    dataset = ToyDataset(num_samples=10)
    loader = DataLoader(dataset, batch_size=4)

    pre_hash = compute_model_parameter_hash(model)
    result = evaluator.evaluate(loader, warmup_batches=1)
    post_hash = compute_model_parameter_hash(model)

    # Weights must be identical before and after evaluation
    assert pre_hash == post_hash, "Model weights were modified during evaluation!"

    # Ensure no gradients were attached
    for param in model.parameters():
        assert param.grad is None, "Model parameters must not contain gradients after evaluation"

    assert result.num_samples == 10
    assert len(result.y_true) == 10
    assert len(result.y_pred) == 10


def test_prediction_and_label_count_consistency():
    """Verify prediction arrays are consistent with probabilities and ground truth."""
    model = CustomCNN(in_channels=3, num_classes=10)
    evaluator = ModelEvaluator(model=model, model_name="custom_cnn", device=torch.device("cpu"))

    dataset = ToyDataset(num_samples=15)
    loader = DataLoader(dataset, batch_size=5)

    res = evaluator.evaluate(loader, warmup_batches=0)

    assert len(res.y_true) == 15
    assert len(res.y_pred) == 15
    assert res.probabilities.shape == (15, 10)
    assert res.logits.shape == (15, 10)

    # Probabilities must sum to 1.0 along class axis
    row_sums = np.sum(res.probabilities, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)

    # y_pred must equal argmax(probabilities)
    expected_preds = np.argmax(res.probabilities, axis=1)
    assert np.array_equal(res.y_pred, expected_preds)


# =========================================================================
# 4. Report Serialization
# =========================================================================


def test_report_serialization(tmp_path: Path):
    """Verify all serialization functions write valid files without errors."""
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_pred_1 = np.array([0, 1, 2, 3, 4, 0, 6, 7, 8, 9])  # 9/10 correct
    y_pred_2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 10/10 correct

    metrics_1 = calculate_all_metrics(y_true, y_pred_1)
    metrics_2 = calculate_all_metrics(y_true, y_pred_2)

    eval_res_1 = EvaluationResult(
        model_name="custom_cnn",
        checkpoint_path="models/custom_cnn_best.pt",
        device="cpu",
        num_samples=10,
        batch_size=4,
        total_parameters=289000,
        y_true=y_true,
        y_pred=y_pred_1,
        probabilities=np.ones((10, 10)) / 10.0,
        logits=np.zeros((10, 10)),
        image_paths=[f"mock_{i}.jpg" for i in range(10)],
        relative_paths=[f"mock/mock_{i}.jpg" for i in range(10)],
        inference_time_total_sec=0.05,
        inference_time_per_image_ms=5.0,
        timestamp="2026-10-15T00:00:00",
    )
    eval_res_2 = EvaluationResult(
        model_name="resnet18",
        checkpoint_path="models/resnet18_best.pt",
        device="cpu",
        num_samples=10,
        batch_size=4,
        total_parameters=11180000,
        y_true=y_true,
        y_pred=y_pred_2,
        probabilities=np.ones((10, 10)) / 10.0,
        logits=np.zeros((10, 10)),
        image_paths=[f"mock_{i}.jpg" for i in range(10)],
        relative_paths=[f"mock/mock_{i}.jpg" for i in range(10)],
        inference_time_total_sec=0.15,
        inference_time_per_image_ms=15.0,
        timestamp="2026-10-15T00:00:00",
    )

    # 1. Confusion Matrix CSV
    cm_path = save_confusion_matrix_csv(
        np.array(metrics_1["raw_confusion_matrix"]),
        tmp_path / "custom_cnn_confusion_matrix.csv"
    )
    assert cm_path.exists()
    cm_df = pd.read_csv(cm_path, index_col="true_class")
    assert cm_df.shape == (10, 10)

    # 2. Per-class CSV
    pc_path = generate_per_class_csv(
        metrics_1, metrics_2, tmp_path / "per_class_metrics.csv"
    )
    assert pc_path.exists()
    pc_df = pd.read_csv(pc_path)
    assert len(pc_df) == 10
    assert "custom_cnn_f1" in pc_df.columns
    assert "resnet18_f1" in pc_df.columns

    # 3. Model comparison dict & CSV
    comp_dict = build_model_comparison_dict(eval_res_1, eval_res_2, metrics_1, metrics_2)
    assert len(comp_dict["comparison_table"]) >= 7

    # 4. Error Analysis CSV
    err_path = generate_error_analysis_csv(
        eval_res_1, eval_res_2, tmp_path / "error_analysis.csv"
    )
    assert err_path.exists()
    err_df = pd.read_csv(err_path)
    assert len(err_df) == 10
    assert "custom_cnn_predicted_class" in err_df.columns
    assert "resnet18_predicted_class" in err_df.columns

    # 5. Markdown Report
    md_path = generate_markdown_report(
        eval_res_1, eval_res_2, metrics_1, metrics_2, tmp_path / "evaluation_report.md"
    )
    assert md_path.exists()
    md_content = md_path.read_text(encoding="utf-8")
    assert "EuroSAT" in md_content
    assert "Macro F1" in md_content
