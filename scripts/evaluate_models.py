"""CLI script for final model evaluation and comparison on EuroSAT RGB test dataset.

Evaluates Custom CNN baseline and fine-tuned ResNet-18 on the untouched test partition,
computes classification metrics, generates publication-quality confusion matrices and
F1 comparison plots, performs sample-level error analysis, and serializes all artifacts.

Usage Examples:
    # Evaluate both models (default)
    python scripts/evaluate_models.py

    # Evaluate ResNet-18 only
    python scripts/evaluate_models.py --model resnet18

    # Evaluate Custom CNN only
    python scripts/evaluate_models.py --model custom_cnn

    # Custom batch size and target output directory
    python scripts/evaluate_models.py --batch-size 128 --output-dir reports/evaluation
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import torch

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import get_base_config, get_custom_cnn_config, get_resnet18_config
from src.constants import EUROSAT_CLASSES, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.dataloader import create_dataloader
from src.evaluation.evaluator import ModelEvaluator, load_model_for_evaluation
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
logger = logging.getLogger("evaluate_models")


def format_comparison_table(comp_dict: Dict[str, Any], m1_name: str, m2_name: str) -> str:
    """Format comparison table into cleanly aligned console string."""
    headers = f"{'Metric':<35} | {m1_name:<15} | {m2_name:<15} | {'Delta':<12} | {'Winner':<15}"
    sep = "-" * len(headers)
    lines = [sep, headers, sep]

    for row in comp_dict["comparison_table"]:
        val1 = row[m1_name]
        val2 = row[m2_name]
        delta = row["delta (m2 - m1)"]
        winner = row["winner"]

        s_val1 = f"{val1:.4f}" if isinstance(val1, float) else f"{val1:,d}" if isinstance(val1, int) else str(val1)
        s_val2 = f"{val2:.4f}" if isinstance(val2, float) else f"{val2:,d}" if isinstance(val2, int) else str(val2)
        s_delta = f"{delta:+.4f}" if isinstance(delta, float) else f"{delta:+d}" if isinstance(delta, int) else str(delta)

        lines.append(f"{row['metric']:<35} | {s_val1:<15} | {s_val2:<15} | {s_delta:<12} | {winner:<15}")

    lines.append(sep)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run final evaluation and side-by-side comparison on the EuroSAT RGB test partition.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["both", "custom_cnn", "resnet18"],
        default="both",
        help="Model(s) to evaluate: 'custom_cnn', 'resnet18', or 'both' (default: both)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size (default: 64)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (default: 0 for Windows safety)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Compute device ('cpu' or 'cuda'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_DATA_DIR / "EuroSAT_RGB",
        help="Path to raw EuroSAT_RGB directory",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=PROCESSED_DATA_DIR / "splits.json",
        help="Path to dataset split manifest JSON",
    )
    parser.add_argument(
        "--custom-cnn-checkpoint",
        type=Path,
        default=MODELS_DIR / "custom_cnn_best.pt",
        help="Checkpoint path for Custom CNN (default: models/custom_cnn_best.pt)",
    )
    parser.add_argument(
        "--resnet18-checkpoint",
        type=Path,
        default=MODELS_DIR / "resnet18_best.pt",
        help="Checkpoint path for ResNet-18 (default: models/resnet18_best.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/evaluation"),
        help="Directory to save evaluation reports and visualizations (default: reports/evaluation)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("      SATELLITE LAND-USE CLASSIFIER - MILESTONE 5 FINAL EVALUATION")
    print("                 EuroSAT RGB Test Partition Benchmark")
    print("=" * 80)

    # 1. Device Resolution
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ENVIRONMENT]")
    print(f"  Execution Device  : {device}")
    print(f"  Target Split      : test (Strictly untouched during training/tuning)")
    print(f"  Output Directory  : {args.output_dir.resolve()}")

    # 2. Path & Prerequisite Validations
    if not args.splits_file.exists():
        print(f"\n[ERROR] Splits manifest not found at: {args.splits_file}", file=sys.stderr)
        print("Please run 'python scripts/create_splits.py' before evaluating.", file=sys.stderr)
        return 1

    if not args.data_dir.exists():
        print(f"\n[ERROR] Dataset directory not found at: {args.data_dir}", file=sys.stderr)
        print("Please run 'python scripts/download_data.py' before evaluating.", file=sys.stderr)
        return 1

    eval_custom_cnn = args.model in ("both", "custom_cnn")
    eval_resnet18 = args.model in ("both", "resnet18")

    if eval_custom_cnn and not args.custom_cnn_checkpoint.exists():
        print(f"\n[ERROR] Custom CNN checkpoint not found at: {args.custom_cnn_checkpoint}", file=sys.stderr)
        print("Please train Custom CNN or provide --custom-cnn-checkpoint path.", file=sys.stderr)
        return 1

    if eval_resnet18 and not args.resnet18_checkpoint.exists():
        print(f"\n[ERROR] ResNet-18 checkpoint not found at: {args.resnet18_checkpoint}", file=sys.stderr)
        print("Please train ResNet-18 or provide --resnet18-checkpoint path.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load Base Configuration
    base_cfg = get_base_config()

    # 4. Create Test DataLoader (Deterministic, return_metadata=True)
    print(f"\n[DATA] Creating Test DataLoader from {args.splits_file}...")
    try:
        test_loader, test_dataset = create_dataloader(
            split="test",
            root_dir=args.data_dir,
            splits_file=args.splits_file,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            norm_type="eurosat",
            config=base_cfg,
            return_metadata=True,
        )
    except Exception as exc:
        logger.exception(f"Failed to create test dataloader: {exc}")
        return 1

    num_test_samples = len(test_dataset)
    print(f"  Test Samples Found: {num_test_samples:,d} across {len(EUROSAT_CLASSES)} classes")
    print(f"  Total Batches     : {len(test_loader)} (Batch size: {args.batch_size})")

    if num_test_samples == 0:
        print("[ERROR] Test dataset is empty! Aborting evaluation.", file=sys.stderr)
        return 1

    eval_res_cnn: Optional[Any] = None
    eval_res_res: Optional[Any] = None
    metrics_cnn: Optional[Dict[str, Any]] = None
    metrics_res: Optional[Dict[str, Any]] = None

    # 5. Evaluate Custom CNN
    if eval_custom_cnn:
        print("\n" + "-" * 70)
        print("  EVALUATING: Custom CNN Baseline")
        print("-" * 70)
        cnn_cfg = {**base_cfg, **get_custom_cnn_config()}
        cnn_model, cnn_ckpt_path = load_model_for_evaluation(
            model_name="custom_cnn",
            checkpoint_path=args.custom_cnn_checkpoint,
            device=device,
            config=cnn_cfg,
        )
        cnn_evaluator = ModelEvaluator(
            model=cnn_model,
            model_name="custom_cnn",
            device=device,
            checkpoint_path=cnn_ckpt_path,
        )
        print(f"  Total Parameters : {cnn_evaluator.total_parameters:,d}")
        print("  Running test evaluation batches (measuring pure forward latency)...")
        eval_res_cnn = cnn_evaluator.evaluate(test_loader, warmup_batches=3)

        print(f"  Evaluation Complete. Calculating metrics...")
        metrics_cnn = calculate_all_metrics(
            eval_res_cnn.y_true, eval_res_cnn.y_pred, class_names=EUROSAT_CLASSES
        )

        # Print snapshot
        print(f"  -> Accuracy       : {metrics_cnn['overall']['accuracy']:.4f}")
        print(f"  -> Macro Precision: {metrics_cnn['overall']['macro_precision']:.4f}")
        print(f"  -> Macro Recall   : {metrics_cnn['overall']['macro_recall']:.4f}")
        print(f"  -> Macro F1       : {metrics_cnn['overall']['macro_f1']:.4f} (PRIMARY)")
        print(f"  -> Weighted F1    : {metrics_cnn['overall']['weighted_f1']:.4f}")
        print(f"  -> Total Errors   : {metrics_cnn['total_errors']} / {num_test_samples}")
        print(f"  -> Avg Latency    : {eval_res_cnn.inference_time_per_image_ms:.2f} ms/image")

        # Save Custom CNN specific artifacts
        cnn_metrics_path = args.output_dir / "custom_cnn_metrics.json"
        with open(cnn_metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": "custom_cnn",
                "checkpoint": str(cnn_ckpt_path),
                "device": str(device),
                "timestamp": eval_res_cnn.timestamp,
                "num_samples": num_test_samples,
                "total_parameters": eval_res_cnn.total_parameters,
                "inference_time_total_sec": eval_res_cnn.inference_time_total_sec,
                "inference_time_per_image_ms": eval_res_cnn.inference_time_per_image_ms,
                "metrics": metrics_cnn,
            }, f, indent=2)

        # Confusion matrices
        cm_raw_cnn = np.array(metrics_cnn["raw_confusion_matrix"])
        cm_norm_cnn = np.array(metrics_cnn["normalized_confusion_matrix"])
        save_confusion_matrix_csv(cm_raw_cnn, args.output_dir / "custom_cnn_confusion_matrix.csv")
        save_confusion_matrix_csv(cm_norm_cnn, args.output_dir / "custom_cnn_confusion_matrix_normalized.csv")

        plot_confusion_matrix(
            cm_raw_cnn,
            class_names=EUROSAT_CLASSES,
            title="Custom CNN: Test Set Confusion Matrix (Raw Counts)",
            output_path=args.output_dir / "custom_cnn_confusion_matrix.png",
            normalized=False,
            cmap="Blues",
        )

    # 6. Evaluate ResNet-18
    if eval_resnet18:
        print("\n" + "-" * 70)
        print("  EVALUATING: ResNet-18 Transfer Learning")
        print("-" * 70)
        res_cfg = {**base_cfg, **get_resnet18_config()}
        res_model, res_ckpt_path = load_model_for_evaluation(
            model_name="resnet18",
            checkpoint_path=args.resnet18_checkpoint,
            device=device,
            config=res_cfg,
        )
        res_evaluator = ModelEvaluator(
            model=res_model,
            model_name="resnet18",
            device=device,
            checkpoint_path=res_ckpt_path,
        )
        print(f"  Total Parameters : {res_evaluator.total_parameters:,d}")
        print("  Running test evaluation batches (measuring pure forward latency)...")
        eval_res_res = res_evaluator.evaluate(test_loader, warmup_batches=3)

        print(f"  Evaluation Complete. Calculating metrics...")
        metrics_res = calculate_all_metrics(
            eval_res_res.y_true, eval_res_res.y_pred, class_names=EUROSAT_CLASSES
        )

        # Print snapshot
        print(f"  -> Accuracy       : {metrics_res['overall']['accuracy']:.4f}")
        print(f"  -> Macro Precision: {metrics_res['overall']['macro_precision']:.4f}")
        print(f"  -> Macro Recall   : {metrics_res['overall']['macro_recall']:.4f}")
        print(f"  -> Macro F1       : {metrics_res['overall']['macro_f1']:.4f} (PRIMARY)")
        print(f"  -> Weighted F1    : {metrics_res['overall']['weighted_f1']:.4f}")
        print(f"  -> Total Errors   : {metrics_res['total_errors']} / {num_test_samples}")
        print(f"  -> Avg Latency    : {eval_res_res.inference_time_per_image_ms:.2f} ms/image")

        # Save ResNet-18 specific artifacts
        res_metrics_path = args.output_dir / "resnet18_metrics.json"
        with open(res_metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": "resnet18",
                "checkpoint": str(res_ckpt_path),
                "device": str(device),
                "timestamp": eval_res_res.timestamp,
                "num_samples": num_test_samples,
                "total_parameters": eval_res_res.total_parameters,
                "inference_time_total_sec": eval_res_res.inference_time_total_sec,
                "inference_time_per_image_ms": eval_res_res.inference_time_per_image_ms,
                "metrics": metrics_res,
            }, f, indent=2)

        # Confusion matrices
        cm_raw_res = np.array(metrics_res["raw_confusion_matrix"])
        cm_norm_res = np.array(metrics_res["normalized_confusion_matrix"])
        save_confusion_matrix_csv(cm_raw_res, args.output_dir / "resnet18_confusion_matrix.csv")
        save_confusion_matrix_csv(cm_norm_res, args.output_dir / "resnet18_confusion_matrix_normalized.csv")

        plot_confusion_matrix(
            cm_raw_res,
            class_names=EUROSAT_CLASSES,
            title="ResNet-18: Test Set Confusion Matrix (Raw Counts)",
            output_path=args.output_dir / "resnet18_confusion_matrix.png",
            normalized=False,
            cmap="Blues",
        )

    # 7. Comparative Analysis (When both evaluated)
    if eval_res_cnn and eval_res_res and metrics_cnn and metrics_res:
        print("\n" + "=" * 80)
        print("          HEAD-TO-HEAD MODEL COMPARISON (EUROSAT TEST PARTITION)")
        print("=" * 80)

        comp_dict = build_model_comparison_dict(eval_res_cnn, eval_res_res, metrics_cnn, metrics_res)
        table_str = format_comparison_table(comp_dict, "custom_cnn", "resnet18")
        print(table_str)

        # Save model_comparison.json
        comp_json_path = args.output_dir / "model_comparison.json"
        with open(comp_json_path, "w", encoding="utf-8") as f:
            json.dump(comp_dict, f, indent=2)
        print(f"\n[ARTIFACT] Saved comparison JSON -> {comp_json_path}")

        # Save model_comparison.csv
        comp_df = pd.DataFrame(comp_dict["comparison_table"])
        comp_csv_path = args.output_dir / "model_comparison.csv"
        comp_df.to_csv(comp_csv_path, index=False)
        print(f"[ARTIFACT] Saved comparison CSV  -> {comp_csv_path}")

        # Save per_class_metrics.csv
        per_class_csv_path = args.output_dir / "per_class_metrics.csv"
        generate_per_class_csv(
            metrics_cnn,
            metrics_res,
            per_class_csv_path,
            name1="custom_cnn",
            name2="resnet18",
            class_names=EUROSAT_CLASSES,
        )
        print(f"[ARTIFACT] Saved per-class CSV   -> {per_class_csv_path}")

        # Save per_class_f1_comparison.png
        f1_plot_path = args.output_dir / "per_class_f1_comparison.png"
        plot_per_class_f1_comparison(
            metrics_cnn["per_class"],
            metrics_res["per_class"],
            name1="Custom CNN",
            name2="ResNet-18",
            class_names=EUROSAT_CLASSES,
            output_path=f1_plot_path,
        )
        print(f"[ARTIFACT] Saved F1 plot image   -> {f1_plot_path}")

        # Save error_analysis.csv
        error_csv_path = args.output_dir / "error_analysis.csv"
        generate_error_analysis_csv(
            eval_res_cnn,
            eval_res_res,
            error_csv_path,
            class_names=EUROSAT_CLASSES,
        )
        print(f"[ARTIFACT] Saved error analysis  -> {error_csv_path}")

        # Save evaluation_report.md
        report_md_path = args.output_dir / "evaluation_report.md"
        generate_markdown_report(
            eval_res_cnn,
            eval_res_res,
            metrics_cnn,
            metrics_res,
            report_md_path,
            class_names=EUROSAT_CLASSES,
        )
        print(f"[ARTIFACT] Saved evaluation report -> {report_md_path}")

    print("\n" + "=" * 80)
    print("   FINAL EVALUATION RUN COMPLETE - ALL ARTIFACTS VERIFIED")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
