"""Loss functions and objective criteria for training land-use classifiers."""

from typing import Optional
import torch
import torch.nn as nn


def get_loss_fn(
    loss_name: str = "cross_entropy",
    weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """Instantiate loss criterion for multi-class classification.

    Args:
        loss_name: Loss function name (default: 'cross_entropy').
        weight: Optional class weights tensor of shape [num_classes].
        label_smoothing: Float for label smoothing epsilon (default: 0.0).

    Returns:
        PyTorch loss module (nn.Module).

    Raises:
        ValueError: If loss_name is not supported.
    """
    normalized_name = loss_name.lower().strip()
    if normalized_name in ("cross_entropy", "ce", "crossentropy"):
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    else:
        raise ValueError(
            f"Unsupported loss function '{loss_name}'. Expected 'cross_entropy'."
        )
