"""Neural network architectures and model factory for EuroSAT RGB classification."""

from typing import Any, Dict, Optional
import torch.nn as nn

from src.models.custom_cnn import CustomCNN
from src.models.resnet18 import ResNet18, ResNet18Classifier, ResNet18Transfer

__all__ = ["CustomCNN", "ResNet18", "ResNet18Classifier", "ResNet18Transfer", "build_model"]


def build_model(
    model_name: str = "custom_cnn",
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> nn.Module:
    """Model factory for instantiating neural network architectures.

    Args:
        model_name: Architecture identifier ('custom_cnn' or 'resnet18').
        config: Optional project configuration dictionary.
        **kwargs: Overrides for model hyperparameters.

    Returns:
        Instantiated PyTorch nn.Module.

    Raises:
        ValueError: If model_name is unrecognized.
    """
    normalized_name = model_name.lower().strip().replace("-", "_")

    if normalized_name in ("custom_cnn", "customcnn", "baseline", "cnn"):
        model_cfg = config.get("model", {}) if config else {}
        in_channels = kwargs.get("in_channels", model_cfg.get("in_channels", 3))
        num_classes = kwargs.get("num_classes", model_cfg.get("num_classes", 10))
        conv_channels = kwargs.get("conv_channels", model_cfg.get("conv_channels", [32, 64, 128]))
        dropout_rate = kwargs.get("dropout_rate", model_cfg.get("dropout_rate", 0.3))

        return CustomCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            conv_channels=conv_channels,
            dropout_rate=dropout_rate,
        )

    elif normalized_name in ("resnet18", "resnet_18", "resnet18_transfer", "resnet18transfer", "resnet"):
        model_cfg = config.get("model", {}) if config else {}
        num_classes = kwargs.get("num_classes", model_cfg.get("num_classes", 10))
        pretrained = kwargs.get("pretrained", model_cfg.get("pretrained", True))
        weights = kwargs.get("weights", model_cfg.get("weights", None))
        dropout_rate = kwargs.get("dropout_rate", model_cfg.get("dropout_rate", 0.0))

        return ResNet18Transfer(
            num_classes=num_classes,
            pretrained=pretrained,
            weights=weights,
            dropout_rate=dropout_rate,
        )

    else:
        raise ValueError(
            f"Unsupported model name: '{model_name}'. Supported architectures are: "
            f"['custom_cnn', 'resnet18']"
        )
