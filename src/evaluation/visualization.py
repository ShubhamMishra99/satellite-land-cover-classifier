"""High-resolution visualization generation for model evaluation and comparison.

This module generates publication-quality plots:
1. Annotated confusion matrices (both raw-count and row-normalized).
2. Per-class F1-score comparison bar charts between Custom CNN and ResNet-18.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments and servers
import matplotlib.pyplot as plt
import numpy as np

from src.constants import CLASS_COLORS, EUROSAT_CLASSES

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[Union[int, float]]]],
    class_names: Sequence[str] = EUROSAT_CLASSES,
    title: str = "Confusion Matrix",
    output_path: Optional[Union[str, Path]] = None,
    normalized: bool = False,
    cmap: str = "Blues",
    figsize: tuple = (10, 8),
    dpi: int = 300,
) -> Path:
    """Render and save a high-resolution annotated confusion matrix.

    Args:
        cm: 2D array of shape [num_classes, num_classes].
        class_names: Sequence of canonical class names in label order.
        title: Title string displayed above the plot.
        output_path: File destination path (.png).
        normalized: Whether cm values are normalized (floats in [0, 1]) or raw counts (ints).
        cmap: Matplotlib colormap name.
        figsize: Figure dimensions in inches.
        dpi: Output resolution.

    Returns:
        Resolved Path to the saved image file.
    """
    matrix = np.array(cm, dtype=float)
    num_classes = len(class_names)

    if matrix.shape != (num_classes, num_classes):
        raise ValueError(
            f"Shape mismatch: matrix is {matrix.shape}, expected ({num_classes}, {num_classes})"
        )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized Rate" if normalized else "Sample Count", rotation=-90, va="bottom", fontsize=11)

    # Class ticks
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label (Ground Truth)",
        xlabel="Predicted Label",
    )

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    # Text annotations in each cell
    thresh = matrix.max() / 2.0 if matrix.max() > 0 else 1.0
    for i in range(num_classes):
        for j in range(num_classes):
            val = matrix[i, j]
            if normalized:
                text = f"{val * 100:.1f}%" if val >= 0.01 else ("<1%" if val > 0 else "0%")
            else:
                text = f"{int(round(val)):,d}"

            color = "white" if val > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8, fontweight="medium")

    fig.tight_layout()

    out = Path(output_path) if output_path else Path("confusion_matrix.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved confusion matrix plot to: {out}")
    return out


def plot_per_class_f1_comparison(
    metrics_m1: Dict[str, Dict[str, Union[float, int]]],
    metrics_m2: Dict[str, Dict[str, Union[float, int]]],
    name1: str = "Custom CNN",
    name2: str = "ResNet-18",
    class_names: Sequence[str] = EUROSAT_CLASSES,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 6),
    dpi: int = 300,
) -> Path:
    """Render and save a grouped bar chart comparing per-class F1-scores.

    Args:
        metrics_m1: Per-class metrics dict for model 1.
        metrics_m2: Per-class metrics dict for model 2.
        name1: Display name for model 1.
        name2: Display name for model 2.
        class_names: Ordered list of class names.
        output_path: File destination path (.png).
        figsize: Figure dimensions.
        dpi: Output resolution.

    Returns:
        Resolved Path to saved image file.
    """
    f1_m1 = [metrics_m1.get(cls, {}).get("f1_score", 0.0) for cls in class_names]
    f1_m2 = [metrics_m2.get(cls, {}).get("f1_score", 0.0) for cls in class_names]

    x = np.arange(len(class_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    rects1 = ax.bar(x - width / 2, f1_m1, width, label=name1, color="#377EB8", edgecolor="white", linewidth=0.8)
    rects2 = ax.bar(x + width / 2, f1_m2, width, label=name2, color="#E41A1C", edgecolor="white", linewidth=0.8)

    ax.set_ylabel("F1-Score", fontsize=11, fontweight="bold")
    ax.set_title(f"Per-Class Test F1-Score Comparison: {name1} vs {name2}", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    # Attach score annotations above bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=0,
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    out = Path(output_path) if output_path else Path("per_class_f1_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved per-class F1 comparison plot to: {out}")
    return out
