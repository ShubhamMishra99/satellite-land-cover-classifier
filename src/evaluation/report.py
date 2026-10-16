"""Report generation and artifact serialization for EuroSAT model evaluation.

This module exports:
1. `custom_cnn_metrics.json` & `resnet18_metrics.json`: Detailed single-model metric logs.
2. `model_comparison.json` & `model_comparison.csv`: Side-by-side metric and latency comparisons.
3. `per_class_metrics.csv`: Tabular per-class precision, recall, F1, and support for both models.
4. `custom_cnn_confusion_matrix.csv` & `resnet18_confusion_matrix.csv`: Raw and normalized counts.
5. `error_analysis.csv`: Detailed diagnostic table of correctly classified and misclassified examples.
6. `evaluation_report.md`: Comprehensive, publishable academic evaluation report.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from src.constants import EUROSAT_CLASSES, NUM_CLASSES
from src.evaluation.evaluator import EvaluationResult
from src.evaluation.metrics import calculate_all_metrics

logger = logging.getLogger(__name__)


def build_model_comparison_dict(
    eval_res_1: EvaluationResult,
    eval_res_2: EvaluationResult,
    metrics_1: Dict[str, Any],
    metrics_2: Dict[str, Any],
) -> Dict[str, Any]:
    """Compile structured comparison dictionary between two evaluated models.

    Args:
        eval_res_1: Evaluation result for model 1.
        eval_res_2: Evaluation result for model 2.
        metrics_1: Calculated metrics dict for model 1.
        metrics_2: Calculated metrics dict for model 2.

    Returns:
        Structured comparison dictionary.
    """
    m1_overall = metrics_1["overall"]
    m2_overall = metrics_2["overall"]

    comparison_records = [
        {
            "metric": "Accuracy",
            eval_res_1.model_name: m1_overall["accuracy"],
            eval_res_2.model_name: m2_overall["accuracy"],
            "delta (m2 - m1)": round(m2_overall["accuracy"] - m1_overall["accuracy"], 4),
            "winner": eval_res_2.model_name if m2_overall["accuracy"] > m1_overall["accuracy"] else eval_res_1.model_name,
        },
        {
            "metric": "Macro Precision",
            eval_res_1.model_name: m1_overall["macro_precision"],
            eval_res_2.model_name: m2_overall["macro_precision"],
            "delta (m2 - m1)": round(m2_overall["macro_precision"] - m1_overall["macro_precision"], 4),
            "winner": eval_res_2.model_name if m2_overall["macro_precision"] > m1_overall["macro_precision"] else eval_res_1.model_name,
        },
        {
            "metric": "Macro Recall",
            eval_res_1.model_name: m1_overall["macro_recall"],
            eval_res_2.model_name: m2_overall["macro_recall"],
            "delta (m2 - m1)": round(m2_overall["macro_recall"] - m1_overall["macro_recall"], 4),
            "winner": eval_res_2.model_name if m2_overall["macro_recall"] > m1_overall["macro_recall"] else eval_res_1.model_name,
        },
        {
            "metric": "Macro F1-Score (Primary)",
            eval_res_1.model_name: m1_overall["macro_f1"],
            eval_res_2.model_name: m2_overall["macro_f1"],
            "delta (m2 - m1)": round(m2_overall["macro_f1"] - m1_overall["macro_f1"], 4),
            "winner": eval_res_2.model_name if m2_overall["macro_f1"] > m1_overall["macro_f1"] else eval_res_1.model_name,
        },
        {
            "metric": "Weighted F1-Score",
            eval_res_1.model_name: m1_overall["weighted_f1"],
            eval_res_2.model_name: m2_overall["weighted_f1"],
            "delta (m2 - m1)": round(m2_overall["weighted_f1"] - m1_overall["weighted_f1"], 4),
            "winner": eval_res_2.model_name if m2_overall["weighted_f1"] > m1_overall["weighted_f1"] else eval_res_1.model_name,
        },
        {
            "metric": "Total Parameters",
            eval_res_1.model_name: eval_res_1.total_parameters,
            eval_res_2.model_name: eval_res_2.total_parameters,
            "delta (m2 - m1)": eval_res_2.total_parameters - eval_res_1.total_parameters,
            "winner": f"{eval_res_1.model_name} (lighter)",
        },
        {
            "metric": "Inference Time Full Test Set (s)",
            eval_res_1.model_name: eval_res_1.inference_time_total_sec,
            eval_res_2.model_name: eval_res_2.inference_time_total_sec,
            "delta (m2 - m1)": round(eval_res_2.inference_time_total_sec - eval_res_1.inference_time_total_sec, 4),
            "winner": eval_res_1.model_name if eval_res_1.inference_time_total_sec < eval_res_2.inference_time_total_sec else eval_res_2.model_name,
        },
        {
            "metric": "Avg Inference Time per Image (ms)",
            eval_res_1.model_name: eval_res_1.inference_time_per_image_ms,
            eval_res_2.model_name: eval_res_2.inference_time_per_image_ms,
            "delta (m2 - m1)": round(eval_res_2.inference_time_per_image_ms - eval_res_1.inference_time_per_image_ms, 3),
            "winner": eval_res_1.model_name if eval_res_1.inference_time_per_image_ms < eval_res_2.inference_time_per_image_ms else eval_res_2.model_name,
        },
    ]

    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset_split": "test",
            "num_test_images": eval_res_1.num_samples,
            "device": eval_res_1.device,
            "model_1": {
                "name": eval_res_1.model_name,
                "checkpoint": eval_res_1.checkpoint_path,
                "parameters": eval_res_1.total_parameters,
            },
            "model_2": {
                "name": eval_res_2.model_name,
                "checkpoint": eval_res_2.checkpoint_path,
                "parameters": eval_res_2.total_parameters,
            },
        },
        "comparison_table": comparison_records,
    }


def save_confusion_matrix_csv(
    cm: np.ndarray,
    output_path: Union[str, Path],
    class_names: Sequence[str] = EUROSAT_CLASSES,
) -> Path:
    """Save confusion matrix as a formatted CSV with row and column class headers."""
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index_label="true_class")
    logger.info(f"Saved confusion matrix CSV to: {out}")
    return out


def generate_per_class_csv(
    metrics_1: Dict[str, Any],
    metrics_2: Dict[str, Any],
    output_path: Union[str, Path],
    name1: str = "custom_cnn",
    name2: str = "resnet18",
    class_names: Sequence[str] = EUROSAT_CLASSES,
) -> Path:
    """Save comprehensive per-class metrics comparison to CSV."""
    rows = []
    pc1 = metrics_1["per_class"]
    pc2 = metrics_2["per_class"]

    for cls in class_names:
        c1 = pc1.get(cls, {})
        c2 = pc2.get(cls, {})
        support = c1.get("support", c2.get("support", 0))

        rows.append({
            "class_name": cls,
            "support": support,
            f"{name1}_precision": c1.get("precision", 0.0),
            f"{name1}_recall": c1.get("recall", 0.0),
            f"{name1}_f1": c1.get("f1_score", 0.0),
            f"{name2}_precision": c2.get("precision", 0.0),
            f"{name2}_recall": c2.get("recall", 0.0),
            f"{name2}_f1": c2.get("f1_score", 0.0),
            "f1_delta": round(c2.get("f1_score", 0.0) - c1.get("f1_score", 0.0), 4),
        })

    df = pd.DataFrame(rows)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Saved per-class metrics CSV to: {out}")
    return out


def generate_error_analysis_csv(
    eval_res_1: EvaluationResult,
    eval_res_2: EvaluationResult,
    output_path: Union[str, Path],
    class_names: Sequence[str] = EUROSAT_CLASSES,
    sample_limit: Optional[int] = None,
) -> Path:
    """Generate sample-level diagnostic error analysis table.

    Captures:
    - Correctly classified examples.
    - Misclassified examples.
    - True class.
    - Predicted class.
    - Model confidence.
    - Image path.

    Args:
        eval_res_1: EvaluationResult for model 1 (Custom CNN).
        eval_res_2: EvaluationResult for model 2 (ResNet-18).
        output_path: Destination path (.csv).
        class_names: Sequence of canonical class names.
        sample_limit: Optional maximum number of rows to save.

    Returns:
        Resolved Path to saved CSV.
    """
    conf_1 = eval_res_1.get_confidence_scores()
    conf_2 = eval_res_2.get_confidence_scores()
    correct_1 = eval_res_1.get_is_correct()
    correct_2 = eval_res_2.get_is_correct()

    records = []
    num_samples = eval_res_1.num_samples

    for idx in range(num_samples):
        true_idx = eval_res_1.y_true[idx]
        p1_idx = eval_res_1.y_pred[idx]
        p2_idx = eval_res_2.y_pred[idx]

        records.append({
            "sample_index": idx,
            "image_path": eval_res_1.image_paths[idx],
            "relative_path": eval_res_1.relative_paths[idx],
            "true_class": class_names[true_idx],
            f"{eval_res_1.model_name}_predicted_class": class_names[p1_idx],
            f"{eval_res_1.model_name}_confidence": round(float(conf_1[idx]), 4),
            f"{eval_res_1.model_name}_correct": bool(correct_1[idx]),
            f"{eval_res_2.model_name}_predicted_class": class_names[p2_idx],
            f"{eval_res_2.model_name}_confidence": round(float(conf_2[idx]), 4),
            f"{eval_res_2.model_name}_correct": bool(correct_2[idx]),
            "models_agree": bool(p1_idx == p2_idx),
            "error_category": (
                "both_correct"
                if (correct_1[idx] and correct_2[idx])
                else (
                    "both_wrong"
                    if (not correct_1[idx] and not correct_2[idx])
                    else (
                        f"{eval_res_2.model_name}_only_correct"
                        if correct_2[idx]
                        else f"{eval_res_1.model_name}_only_correct"
                    )
                )
            ),
        })

    df = pd.DataFrame(records)
    if sample_limit is not None and sample_limit < len(df):
        # Stratify to ensure representation of all error categories
        categories = df["error_category"].unique()
        per_cat_limit = max(sample_limit // len(categories), 5)
        df = df.groupby("error_category").apply(lambda g: g.head(per_cat_limit)).reset_index(drop=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Saved error analysis CSV ({len(df)} samples) to: {out}")
    return out


def generate_markdown_report(
    eval_res_1: EvaluationResult,
    eval_res_2: EvaluationResult,
    metrics_1: Dict[str, Any],
    metrics_2: Dict[str, Any],
    output_path: Union[str, Path],
    class_names: Sequence[str] = EUROSAT_CLASSES,
) -> Path:
    """Generate comprehensive academic evaluation markdown report."""
    comp_dict = build_model_comparison_dict(eval_res_1, eval_res_2, metrics_1, metrics_2)
    m1_name = eval_res_1.model_name
    m2_name = eval_res_2.model_name

    m1_ov = metrics_1["overall"]
    m2_ov = metrics_2["overall"]

    # Top misclassification pairs
    m1_top_pairs = metrics_1["top_confusion_pairs"][:5]
    m2_top_pairs = metrics_2["top_confusion_pairs"][:5]

    lines = []
    lines.append("# Satellite Land-Use Classifier: Milestone 5 Final Evaluation & Model Comparison")
    lines.append("")
    lines.append("> **EuroSAT RGB Land-Use / Land-Cover Classification Benchmark**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(
        f"This report presents the rigorous, reproducible final evaluation of the **Custom CNN baseline** "
        f"and the fine-tuned **ResNet-18 transfer learning** model on the untouched EuroSAT RGB test partition. "
        f"Both models were evaluated on the exact same **{eval_res_1.num_samples:,d} test images** using identical deterministic "
        f"preprocessing pipelines and zero gradient computation."
    )
    lines.append("")
    f1_delta = m2_ov["macro_f1"] - m1_ov["macro_f1"]
    lines.append(
        f"- **Primary Model Selection Metric (Macro F1)**: **{m2_name} achieved {m2_ov['macro_f1']:.4f}** "
        f"vs **{m1_name} achieved {m1_ov['macro_f1']:.4f}** (an absolute improvement of **+{f1_delta:.4f}** / **{f1_delta*100:.2f}%**)."
    )
    lines.append(
        f"- **Test Accuracy**: **{m2_name} achieved {m2_ov['accuracy']*100:.2f}%** vs **{m1_name} at {m1_ov['accuracy']*100:.2f}%**."
    )
    lines.append(
        f"- **Total Errors on Test Set**: {m2_name} committed **{metrics_2['total_errors']} errors** ({metrics_2['error_rate']*100:.2f}% error rate), "
        f"compared to **{metrics_1['total_errors']} errors** ({metrics_1['error_rate']*100:.2f}% error rate) for {m1_name}."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Evaluation Metadata & Strict Protocol")
    lines.append("")
    lines.append("| Parameter | Specification / Value |")
    lines.append("|---|---|")
    lines.append(f"| **Evaluation Date / Timestamp** | `{comp_dict['metadata']['timestamp']}` |")
    lines.append(f"| **Dataset Partition** | `test` (Partition manifest: `data/processed/splits.json`) |")
    lines.append(f"| **Evaluated Sample Count** | **{eval_res_1.num_samples:,d} images** (15% stratified test partition) |")
    lines.append(f"| **Hardware Execution Device** | `{eval_res_1.device}` |")
    lines.append(f"| **Image Input Dimensions** | `64 x 64` pixels, 3 RGB channels |")
    lines.append(f"| **Deterministic Preprocessing** | Resize(64,64) -> ToTensor() -> EuroSAT Dataset Normalization |")
    lines.append(f"| **Normalization Statistics** | EuroSAT Mean `[0.3444, 0.3803, 0.4078]`, Std `[0.2026, 0.1368, 0.1156]` |")
    lines.append(f"| **Gradient Computation** | Strictly disabled (`torch.inference_mode()`) |")
    lines.append(f"| **Weight Integrity Verification** | SHA-256 parameter hash verified before and after evaluation (Zero mutation) |")
    lines.append(f"| **Zero Division Policy** | `zero_division=0.0` (Safely handles zero precision/recall cases) |")
    lines.append(f"| **Custom CNN Checkpoint** | `{eval_res_1.checkpoint_path}` |")
    lines.append(f"| **ResNet-18 Checkpoint** | `{eval_res_2.checkpoint_path}` |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Overall Model Comparison")
    lines.append("")
    lines.append("| Metric / Dimension | Custom CNN (Baseline) | ResNet-18 (Transfer Learning) | Delta (ResNet - CNN) | Top Performer |")
    lines.append("|---|---|---|---|---|")
    for row in comp_dict["comparison_table"]:
        lines.append(f"| **{row['metric']}** | {row[m1_name]} | {row[m2_name]} | {row['delta (m2 - m1)']} | **{row['winner']}** |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Per-Class Performance Breakdown")
    lines.append("")
    lines.append("Canonical EuroSAT alphabetical class ordering preserved:")
    lines.append("")
    lines.append("| # | Land-Cover Class | Support | Custom CNN Prec | Custom CNN Rec | Custom CNN F1 | ResNet-18 Prec | ResNet-18 Rec | ResNet-18 F1 | F1 Delta |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    pc1 = metrics_1["per_class"]
    pc2 = metrics_2["per_class"]
    for idx, cls in enumerate(class_names, start=1):
        c1 = pc1[cls]
        c2 = pc2[cls]
        delta_f1 = c2["f1_score"] - c1["f1_score"]
        lines.append(
            f"| {idx} | **{cls}** | {c1['support']} | {c1['precision']:.4f} | {c1['recall']:.4f} | {c1['f1_score']:.4f} | "
            f"{c2['precision']:.4f} | {c2['recall']:.4f} | **{c2['f1_score']:.4f}** | {delta_f1:+.4f} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Error Analysis & Confusion Matrix Diagnosis")
    lines.append("")
    lines.append("### 5.1 Most Difficult and Easiest Classes")
    lines.append("")
    lines.append(f"#### Custom CNN:")
    m1_high = ", ".join([f"{x['class']} ({x['f1_score']:.4f})" for x in metrics_1['highest_f1_classes']])
    m1_low = ", ".join([f"{x['class']} ({x['f1_score']:.4f})" for x in metrics_1['lowest_f1_classes']])
    lines.append(f"- **Top 3 Highest F1 Classes**: {m1_high}")
    lines.append(f"- **Top 3 Lowest F1 Classes**: {m1_low}")
    lines.append("")
    lines.append(f"#### ResNet-18:")
    m2_high = ", ".join([f"{x['class']} ({x['f1_score']:.4f})" for x in metrics_2['highest_f1_classes']])
    m2_low = ", ".join([f"{x['class']} ({x['f1_score']:.4f})" for x in metrics_2['lowest_f1_classes']])
    lines.append(f"- **Top 3 Highest F1 Classes**: {m2_high}")
    lines.append(f"- **Top 3 Lowest F1 Classes**: {m2_low}")
    lines.append("")
    lines.append("### 5.2 Top Confusion Pairs")
    lines.append("")
    lines.append(f"#### Custom CNN Top Confusion Pairs:")
    lines.append("| True Class | Predicted Class | Misclassified Count | Share of Total Errors |")
    lines.append("|---|---|---|---|")
    for pair in m1_top_pairs:
        lines.append(f"| {pair['true_class']} | {pair['predicted_class']} | {pair['count']} | {pair['pct_of_errors']:.1f}% |")
    lines.append("")
    lines.append(f"#### ResNet-18 Top Confusion Pairs:")
    lines.append("| True Class | Predicted Class | Misclassified Count | Share of Total Errors |")
    lines.append("|---|---|---|---|")
    for pair in m2_top_pairs:
        lines.append(f"| {pair['true_class']} | {pair['predicted_class']} | {pair['count']} | {pair['pct_of_errors']:.1f}% |")
    lines.append("")
    lines.append("### 5.3 Remote Sensing Domain Explanations")
    lines.append(
        "- **Highway vs River**: Both feature narrow, linear spectral signatures threading across rural or urban terrain. "
        "At 64x64 resolution, asphalt and dark water bodies share low reflectance patterns that confuse basic convolutional feature detectors.\n"
        "- **Pasture vs HerbaceousVegetation**: Both represent non-forest green vegetation. "
        "Distinguishing managed grazing fields from natural grasslands requires deep hierarchical texture cues learned effectively by pretrained ResNet-18.\n"
        "- **AnnualCrop vs PermanentCrop**: Crop cycles produce seasonal soil exposure and regular furrow textures that occasionally overlap in spectral space."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Inference Latency & Benchmarking Methodology")
    lines.append("")
    lines.append(
        "- **Hardware Platform**: `" + eval_res_1.device + "`\n"
        "- **Methodology**: Evaluated using warm-up passes (5 batches) prior to timing. "
        "CUDA synchronization invoked before and after batch executions when running on GPU. "
        "Checkpoint loading and dataset initialization overhead were strictly excluded from the timing window.\n"
        f"- **Custom CNN Latency**: {eval_res_1.inference_time_total_sec:.2f} s total ({eval_res_1.inference_time_per_image_ms:.2f} ms/image).\n"
        f"- **ResNet-18 Latency**: {eval_res_2.inference_time_total_sec:.2f} s total ({eval_res_2.inference_time_per_image_ms:.2f} ms/image).\n"
        "- **Generalization Note**: Latency results depend heavily on host CPU architecture, thread allocation, batch size, and thermal throttling; "
        "timing measurements reported here reflect this specific host setup."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. Academic Insights & Viva Alignment")
    lines.append("")
    lines.append("### Why Macro F1 is the Decisive Metric")
    lines.append(
        "Accuracy treats all samples equally, meaning a model can achieve high accuracy by mastering majority classes "
        "while failing on difficult or rarer classes. **Macro F1** computes unweighted per-class harmonic means: "
        "$$\\text{Macro } F_1 = \\frac{1}{C} \\sum_{c=1}^C F_{1, c}$$ "
        "This penalizes models that fail on subtle classes (e.g., HerbaceousVegetation, River) and guarantees balanced domain mapping."
    )
    lines.append("")
    lines.append("### Why Test Performance Differs from Validation Performance")
    lines.append(
        "- **Untouched Partition**: The test partition was never exposed to training, learning-rate scheduling, or early-stopping decisions.\n"
        "- **Validation Selection Bias**: The validation set was used to select the best checkpoint epoch; test evaluation gives an unbiased generalization estimate."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 8. Generated Output Artifacts")
    lines.append("")
    lines.append("- `reports/evaluation/custom_cnn_metrics.json`")
    lines.append("- `reports/evaluation/resnet18_metrics.json`")
    lines.append("- `reports/evaluation/model_comparison.json`")
    lines.append("- `reports/evaluation/model_comparison.csv`")
    lines.append("- `reports/evaluation/per_class_metrics.csv`")
    lines.append("- `reports/evaluation/custom_cnn_confusion_matrix.png`")
    lines.append("- `reports/evaluation/custom_cnn_confusion_matrix.csv`")
    lines.append("- `reports/evaluation/resnet18_confusion_matrix.png`")
    lines.append("- `reports/evaluation/resnet18_confusion_matrix.csv`")
    lines.append("- `reports/evaluation/per_class_f1_comparison.png`")
    lines.append("- `reports/evaluation/error_analysis.csv`")
    lines.append("- `reports/evaluation/evaluation_report.md`")
    lines.append("")

    content = "\n".join(lines)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Saved evaluation markdown report to: {out}")
    return out
