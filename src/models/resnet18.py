"""ResNet-18 transfer learning architecture with staged fine-tuning for EuroSAT RGB classification.

Transfer Learning Rationale:
----------------------------
1. Custom CNN Baseline (Milestone 3):
   - Initialized with random weights (He/Kaiming normal initialization).
   - Learns all visual representations (edges, textures, spatial features) from scratch exclusively
     using EuroSAT's 18,900 training images.

2. ResNet-18 Transfer Learning (Milestone 4):
   - Initialized with ImageNet-1K pretrained weights (trained on 1.28 million natural images across 1,000 classes).
   - Early convolutional layers already encode generic, robust visual primitives (Gabor-like edge filters,
     color gradients, textural motifs) that transfer effectively to aerial and satellite imagery.
   - Staged Fine-Tuning Strategy:
     * Phase 1 (Feature Extraction): The entire backbone (conv1, bn1, layer1-layer4) is frozen (requires_grad = False).
       Only the newly initialized 10-class linear classification head is trained with learning rate ~1e-3.
       This aligns the classifier head with EuroSAT class distributions without corrupting pretrained backbone weights.
     * Phase 2 (Selective Fine-Tuning): High-level semantic layers (layer3, layer4) and the classification head
       are unfrozen (requires_grad = True), while early feature extractors (conv1, bn1, layer1, layer2) remain frozen.
       Trained with a smaller learning rate (~5e-5) and Cosine Annealing to gently adapt high-level domain representations
       to satellite land-use patterns without catastrophic forgetting.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from src.constants import NUM_CLASSES

logger = logging.getLogger(__name__)


class ResNet18Transfer(nn.Module):
    """ResNet-18 architecture adapted for EuroSAT RGB 10-class classification.

    Wraps torchvision's ResNet-18 with ImageNet-1K pretrained weights, replaces the
    final 1000-class fully-connected layer with a 10-class classifier, and provides
    methods to manage Phase 1 (feature extraction) and Phase 2 (staged fine-tuning) parameter freezing.

    Attributes:
        num_classes: Number of output land-use categories (default: 10).
        pretrained: Whether ImageNet pretrained weights were requested.
        pretrained_loaded: Whether pretrained weights were successfully initialized.
        dropout_rate: Optional dropout rate applied before linear classification head.
        backbone: Underlying torchvision ResNet-18 model.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        weights: Optional[Union[str, ResNet18_Weights]] = None,
        dropout_rate: float = 0.0,
    ):
        """Initialize ResNet18Transfer model.

        Args:
            num_classes: Number of target land-use classes (default: 10).
            pretrained: If True, load ImageNet-1K pretrained weights.
            weights: Specific torchvision weights enum or string (e.g., 'IMAGENET1K_V1').
                     Defaults to ResNet18_Weights.IMAGENET1K_V1 when pretrained=True.
            dropout_rate: Dropout probability in classification head (default: 0.0).
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.pretrained_loaded = False

        # Resolve torchvision weights specification
        resolved_weights: Optional[ResNet18_Weights] = None
        if pretrained:
            if weights is None or weights == "DEFAULT" or weights == "IMAGENET1K_V1":
                resolved_weights = ResNet18_Weights.IMAGENET1K_V1
            elif isinstance(weights, ResNet18_Weights):
                resolved_weights = weights
            elif isinstance(weights, str):
                try:
                    resolved_weights = ResNet18_Weights[weights]
                except KeyError:
                    logger.warning(
                        f"Unknown weights string '{weights}'. Falling back to ResNet18_Weights.IMAGENET1K_V1."
                    )
                    resolved_weights = ResNet18_Weights.IMAGENET1K_V1

        # Instantiate torchvision ResNet-18 backbone
        try:
            self.backbone = models.resnet18(weights=resolved_weights)
            if resolved_weights is not None:
                self.pretrained_loaded = True
                logger.info(f"Loaded ResNet-18 with ImageNet-1K pretrained weights: {resolved_weights}")
            else:
                logger.info("Instantiated ResNet-18 with random weight initialization (pretrained=False).")
        except Exception as exc:
            logger.error(
                f"Failed to load pretrained weights ({exc}). Instantiating ResNet-18 without pretrained weights."
            )
            self.backbone = models.resnet18(weights=None)
            self.pretrained_loaded = False

        # Replace classification head (fc: in_features=512 -> num_classes)
        in_features = self.backbone.fc.in_features
        if dropout_rate > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

        # Initialize the new classifier head weights using Xavier uniform
        self._initialize_head()

    def _initialize_head(self) -> None:
        """Initialize parameters of the new classification head."""
        for m in self.backbone.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def fc(self) -> nn.Module:
        """Access the classification head module."""
        return self.backbone.fc

    @property
    def classifier(self) -> nn.Module:
        """Alias for classification head."""
        return self.backbone.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResNet-18 backbone and EuroSAT classification head.

        Args:
            x: Input tensor of shape [B, 3, H, W].

        Returns:
            Raw logits tensor of shape [B, num_classes].
            Note: Softmax is NOT applied internally; CrossEntropyLoss expects raw logits.
        """
        return self.backbone(x)

    def freeze_for_phase1(self) -> None:
        """Configure model for Phase 1: Feature Extraction.

        Freezes all convolutional backbone parameters (conv1, bn1, layer1, layer2, layer3, layer4)
        and leaves only the final classification head (fc) trainable.
        """
        for name, param in self.backbone.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        total, trainable = self.count_parameters()
        logger.info(
            f"[Phase 1: Feature Extraction] Backbone frozen. "
            f"Trainable params: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)"
        )

    def unfreeze_for_phase2(
        self,
        unfreeze_layers: Sequence[str] = ("layer3", "layer4", "fc"),
    ) -> None:
        """Configure model for Phase 2: Staged Fine-Tuning.

        Unfreezes specified high-level residual blocks (default: layer3, layer4, fc) while
        keeping early feature extractors (conv1, bn1, layer1, layer2) frozen.

        Args:
            unfreeze_layers: Layer prefixes to unfreeze (default: ('layer3', 'layer4', 'fc')).
        """
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze requested layers
        for name, param in self.backbone.named_parameters():
            if any(layer_prefix in name for layer_prefix in unfreeze_layers):
                param.requires_grad = True

        total, trainable = self.count_parameters()
        logger.info(
            f"[Phase 2: Fine-Tuning] Unfroze {unfreeze_layers}. "
            f"Trainable params: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)"
        )

    def count_parameters(self) -> Tuple[int, int]:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def get_trainable_param_names(self) -> List[str]:
        """Return list of parameter names where requires_grad is True."""
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def get_parameter_groups(
        self,
        backbone_lr: float,
        head_lr: float,
        weight_decay: float = 0.0001,
    ) -> List[Dict[str, Any]]:
        """Construct parameter groups with differential learning rates for Phase 2.

        Args:
            backbone_lr: Learning rate for unfrozen backbone layers (layer3, layer4).
            head_lr: Learning rate for the classification head (fc).
            weight_decay: L2 regularization penalty.

        Returns:
            List of parameter group dictionaries for torch.optim.Optimizer.
        """
        backbone_params = []
        head_params = []

        for name, param in self.backbone.named_parameters():
            if not param.requires_grad:
                continue
            if "fc" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
            })

        return param_groups

    def __repr__(self) -> str:
        total, trainable = self.count_parameters()
        return (
            f"ResNet18Transfer(\n"
            f"  num_classes={self.num_classes},\n"
            f"  pretrained={self.pretrained} (loaded={self.pretrained_loaded}),\n"
            f"  dropout_rate={self.dropout_rate},\n"
            f"  total_params={total:,},\n"
            f"  trainable_params={trainable:,}\n"
            f")"
        )


# Aliases for backward compatibility and flexibility
ResNet18 = ResNet18Transfer
ResNet18Classifier = ResNet18Transfer
