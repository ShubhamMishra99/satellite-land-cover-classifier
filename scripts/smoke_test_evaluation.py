"""Smoke test script for Milestone 5: Final Model Evaluation & Comparison Pipeline.

Verifies end-to-end evaluation functionality on a small subset of test images:
1. Verifies checkpoint and dataset existence.
2. Creates subset test DataLoader.
3. Loads Custom CNN and verifies weight immutability during evaluation.
4. Loads ResNet-18 and verifies weight immutability during evaluation.
5. Verifies metric calculations (Accuracy, Macro F1, Weighted F1, Per-class).
6. Verifies confusion matrix generation and serialization.
7. Verifies comparison report generation in a temporary directory.
"""

from datetime import datetime
import hashlib
import logging
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.constants import EUROSAT_CLASSES, MODELS_DIR, NUM_CLASSES, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.dataloader import create_dataloader
from src.evaluation.evaluator import (
    EvaluationResult,
    ModelEvaluator,
    compute_model_parameter_hash,
    load_model_for_evaluation,
)
from src.evaluation.metrics import calculate_all_metrics
from src.evaluation.report import (
    build_model_comparison_dict,
    generate_error_analysis_csv,
    generate_markdown_report,
    generate_per_class_csv,
    save_confusion_matrix_csv,
)
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_per_class_f1_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smoke_test_evaluation")


def run_smoke_test() -> int:
    print("=" * 75)
    print("   MILESTONE 5: FINAL MODEL EVALUATION SMOKE TEST")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] Compute Device: {device}")

    # Step 1: Check Checkpoints and Manifest
    print("\n[Step 1/7] Verifying Checkpoints and Split Manifest...")
    splits_file = PROCESSED_DATA_DIR / "splits.json"
    cnn_ckpt = MODELS_DIR / "custom_cnn_best.pt"
    res_ckpt = MODELS_DIR / "resnet18_best.pt"

    assert splits_file.exists(), f"Missing splits manifest: {splits_file}"
    assert cnn_ckpt.exists(), f"Missing Custom CNN checkpoint: {cnn_ckpt}"
    assert res_ckpt.exists(), f"Missing ResNet-18 checkpoint: {res_ckpt}"
    print(f"  [PASS] All prerequisites present on disk.")

    # Step 2: Create a Small Subset Test DataLoader (e.g. 30 samples)
    print("\n[Step 2/7] Creating 30-Sample Test Subset DataLoader...")
    full_loader, full_dataset = create_dataloader(
        split="test",
        batch_size=10,
        return_metadata=True,
    )
    subset_indices = list(range(30))
    subset_dataset = Subset(full_dataset, subset_indices)
    subset_loader = DataLoader(subset_dataset, batch_size=10, shuffle=False)
    print(f"  [PASS] Created test subset of {len(subset_dataset)} samples across 3 batches.")

    # Step 3: Evaluate Custom CNN & Verify Weight Immutability
    print("\n[Step 3/7] Evaluating Custom CNN & Verifying Weight Immutability...")
    cnn_model, _ = load_model_for_evaluation("custom_cnn", checkpoint_path=cnn_ckpt, device=device)
    cnn_evaluator = ModelEvaluator(cnn_model, "custom_cnn", device=device, checkpoint_path=cnn_ckpt)

    pre_hash_cnn = compute_model_parameter_hash(cnn_model)
    cnn_result = cnn_evaluator.evaluate(subset_loader, warmup_batches=1)
    post_hash_cnn = compute_model_parameter_hash(cnn_model)

    assert pre_hash_cnn == post_hash_cnn, "Custom CNN weights mutated during evaluation!"
    assert cnn_result.num_samples == 30
    assert len(cnn_result.y_true) == 30
    assert len(cnn_result.y_pred) == 30
    print(f"  [PASS] Custom CNN evaluated: 30 samples, 0 weight mutations. Hash: {pre_hash_cnn[:12]}...")

    # Step 4: Evaluate ResNet-18 & Verify Weight Immutability
    print("\n[Step 4/7] Evaluating ResNet-18 & Verifying Weight Immutability...")
    res_model, _ = load_model_for_evaluation("resnet18", checkpoint_path=res_ckpt, device=device)
    res_evaluator = ModelEvaluator(res_model, "resnet18", device=device, checkpoint_path=res_ckpt)

    pre_hash_res = compute_model_parameter_hash(res_model)
    res_result = res_evaluator.evaluate(subset_loader, warmup_batches=1)
    post_hash_res = compute_model_parameter_hash(res_model)

    assert pre_hash_res == post_hash_res, "ResNet-18 weights mutated during evaluation!"
    assert res_result.num_samples == 30
    assert len(res_result.y_true) == 30
    assert len(res_result.y_pred) == 30
    print(f"  [PASS] ResNet-18 evaluated: 30 samples, 0 weight mutations. Hash: {pre_hash_res[:12]}...")

    # Step 5: Calculate Comprehensive Metrics
    print("\n[Step 5/7] Calculating Metrics & Verifying Metric Integrity...")
    cnn_metrics = calculate_all_metrics(cnn_result.y_true, cnn_result.y_pred, class_names=EUROSAT_CLASSES)
    res_metrics = calculate_all_metrics(res_result.y_true, res_result.y_pred, class_names=EUROSAT_CLASSES)

    for m in (cnn_metrics, res_metrics):
        assert 0.0 <= m["overall"]["accuracy"] <= 1.0
        assert 0.0 <= m["overall"]["macro_f1"] <= 1.0
        assert 0.0 <= m["overall"]["weighted_f1"] <= 1.0
        assert len(m["per_class"]) == 10
        assert np.array(m["raw_confusion_matrix"]).shape == (10, 10)
    print(f"  [PASS] Metrics calculated safely. Zero division handled gracefully.")

    # Step 6: Test Visualization Generation
    print("\n[Step 6/7] Testing High-Resolution Plot Generation in Temp Directory...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Confusion Matrix Plots
        cm_png_cnn = plot_confusion_matrix(
            cnn_metrics["raw_confusion_matrix"],
            class_names=EUROSAT_CLASSES,
            title="Custom CNN Smoke Test CM",
            output_path=tmp_path / "cnn_cm.png",
        )
        assert cm_png_cnn.exists() and cm_png_cnn.stat().st_size > 0

        # Per-class F1 Comparison Plot
        f1_png = plot_per_class_f1_comparison(
            cnn_metrics["per_class"],
            res_metrics["per_class"],
            name1="Custom CNN",
            name2="ResNet-18",
            class_names=EUROSAT_CLASSES,
            output_path=tmp_path / "f1_comparison.png",
        )
        assert f1_png.exists() and f1_png.stat().st_size > 0
    print(f"  [PASS] Confusion matrix and F1 comparison plots generated successfully.")

    # Step 7: Test Report & CSV Serialization
    print("\n[Step 7/7] Testing Report & CSV Serialization in Temp Directory...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        save_confusion_matrix_csv(cnn_metrics["raw_confusion_matrix"], tmp_path / "cm.csv")
        generate_per_class_csv(cnn_metrics, res_metrics, tmp_path / "per_class.csv")
        generate_error_analysis_csv(cnn_result, res_result, tmp_path / "error_analysis.csv")
        generate_markdown_report(cnn_result, res_result, cnn_metrics, res_metrics, tmp_path / "report.md")

        assert (tmp_path / "cm.csv").exists()
        assert (tmp_path / "per_class.csv").exists()
        assert (tmp_path / "error_analysis.csv").exists()
        assert (tmp_path / "report.md").exists()
    print(f"  [PASS] All reports and CSV manifests successfully serialized.")

    print("\n" + "=" * 75)
    print("   RESULT: ALL EVALUATION SMOKE TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
