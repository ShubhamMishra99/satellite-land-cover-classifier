"""Evaluation module for Satellite Land-Use Classifier.

Provides:
- Metrics calculation (accuracy, macro/weighted F1, per-class precision/recall/F1/support)
- ModelEvaluator engine for reproducible test partition evaluation and latency timing
- Visualization tools for high-res confusion matrices and per-class comparison charts
- Report generation and structured serialization (JSON, CSV, Markdown)
"""

from src.evaluation.evaluator import (
    EvaluationResult,
    ModelEvaluator,
    compute_model_parameter_hash,
    load_model_for_evaluation,
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
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_per_class_f1_comparison,
)

__all__ = [
    "EvaluationResult",
    "ModelEvaluator",
    "compute_model_parameter_hash",
    "load_model_for_evaluation",
    "calculate_all_metrics",
    "calculate_confusion_matrix",
    "calculate_overall_metrics",
    "calculate_per_class_metrics",
    "find_top_confusion_pairs",
    "to_numpy_1d",
    "build_model_comparison_dict",
    "generate_error_analysis_csv",
    "generate_markdown_report",
    "generate_per_class_csv",
    "save_confusion_matrix_csv",
    "plot_confusion_matrix",
    "plot_per_class_f1_comparison",
]
