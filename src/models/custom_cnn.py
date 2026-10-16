"""Custom CNN baseline architecture for EuroSAT RGB classification.

This module defines a 3-block convolutional neural network trained from scratch,
serving as the empirical baseline to benchmark against ResNet-18 transfer learning.
"""

from typing import Any, Dict, Optional, Sequence, Tuple
import torch
import torch.nn as nn

from src.constants import IMAGE_CHANNELS, NUM_CLASSES


class ConvBlock(nn.Module):
    """Standard Convolutional Block: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    """Custom 3-Block Convolutional Neural Network for 64x64 EuroSAT RGB Tiles.

    Architecture:
        Input: [B, 3, 64, 64]
        Block 1: Conv(3->32, k=3, p=1) -> BN(32) -> ReLU -> MaxPool(2)  => [B, 32, 32, 32]
        Block 2: Conv(32->64, k=3, p=1) -> BN(64) -> ReLU -> MaxPool(2) => [B, 64, 16, 16]
        Block 3: Conv(64->128, k=3, p=1) -> BN(128) -> ReLU -> MaxPool(2) => [B, 128, 8, 8]
        Head:
            AdaptiveAvgPool2d((1, 1))                                    => [B, 128, 1, 1]
            Flatten                                                      => [B, 128]
            Dropout(p=0.3)                                               => [B, 128]
            Linear(128 -> num_classes)                                   => [B, 10]

    Note:
        Raw unnormalized logits are returned. Softmax is NOT applied internally
        as torch.nn.CrossEntropyLoss incorporates LogSoftmax for numerical stability.
    """

    def __init__(
        self,
        in_channels: int = IMAGE_CHANNELS,
        num_classes: int = NUM_CLASSES,
        conv_channels: Sequence[int] = (32, 64, 128),
        dropout_rate: float = 0.3,
    ):
        """Initialize CustomCNN.

        Args:
            in_channels: Number of input image channels (default: 3).
            num_classes: Number of target land-cover classes (default: 10).
            conv_channels: Output channels for each convolutional block (default: [32, 64, 128]).
            dropout_rate: Dropout probability in classification head (default: 0.3).
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.conv_channels = tuple(conv_channels)
        self.dropout_rate = dropout_rate

        # Construct convolutional feature extractor
        feature_layers = []
        current_in = in_channels
        for out_ch in self.conv_channels:
            feature_layers.append(ConvBlock(current_in, out_ch, pool=True))
            current_in = out_ch

        self.features = nn.Sequential(*feature_layers)

        # Global average pooling and classification head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(current_in, num_classes),
        )

        # Initialize network weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using He (Kaiming) normal initialization for Conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extractor and classification head.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Logits tensor of shape [B, num_classes].
        """
        features = self.features(x)
        pooled = self.global_pool(features)
        logits = self.classifier(pooled)
        return logits

    def count_parameters(self) -> Tuple[int, int]:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def __repr__(self) -> str:
        total, trainable = self.count_parameters()
        return (
            f"CustomCNN(\n"
            f"  in_channels={self.in_channels},\n"
            f"  num_classes={self.num_classes},\n"
            f"  conv_channels={self.conv_channels},\n"
            f"  dropout_rate={self.dropout_rate},\n"
            f"  total_params={total:,},\n"
            f"  trainable_params={trainable:,}\n"
            f")"
        )
