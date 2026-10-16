"""Unit tests for ResNet-18 Transfer Learning architecture, freezing logic, and model factory."""

import pytest
import torch
import torch.nn as nn

from src.constants import NUM_CLASSES
from src.models import build_model
from src.models.resnet18 import ResNet18Transfer


def test_resnet18_instantiation():
    """Verify default instantiation with random weights (pretrained=False for fast tests)."""
    model = ResNet18Transfer(num_classes=NUM_CLASSES, pretrained=False)
    assert isinstance(model, nn.Module)
    assert model.num_classes == 10
    assert model.pretrained is False
    assert model.dropout_rate == 0.0

    total_params, trainable_params = model.count_parameters()
    assert total_params > 11_000_000
    assert total_params == trainable_params
    assert "ResNet18Transfer" in repr(model)


def test_resnet18_forward_shape():
    """Verify input shape [2, 3, 64, 64] produces output logits of shape [2, 10]."""
    model = ResNet18Transfer(num_classes=10, pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 10)
    assert output.dtype == torch.float32


def test_resnet18_batch_size_preservation():
    """Verify arbitrary batch sizes are preserved accurately."""
    model = ResNet18Transfer(num_classes=10, pretrained=False)
    model.eval()

    for batch_size in [1, 4, 8, 16]:
        dummy_input = torch.randn(batch_size, 3, 64, 64)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (batch_size, 10)


def test_resnet18_configurable_classes():
    """Verify configurable number of output classes and dropout."""
    model = ResNet18Transfer(num_classes=5, pretrained=False, dropout_rate=0.2)
    model.eval()

    dummy_input = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (3, 5)
    assert model.num_classes == 5


def test_resnet18_final_classifier_head():
    """Verify that classification head maps from 512 backbone features to 10 classes."""
    model = ResNet18Transfer(num_classes=10, pretrained=False)
    assert hasattr(model, "fc")
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.in_features == 512
    assert model.fc.out_features == 10


def test_resnet18_no_softmax_applied():
    """Verify that raw logits are returned without internal Softmax."""
    model = ResNet18Transfer(num_classes=10, pretrained=False)
    dummy_input = torch.randn(4, 3, 64, 64)
    logits = model(dummy_input)

    row_sums = logits.sum(dim=-1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums)), (
        "Softmax was incorrectly applied inside ResNet-18 forward pass!"
    )


def test_resnet18_outputs_finite_values():
    """Verify that forward pass produces finite numbers without NaN or Inf."""
    model = ResNet18Transfer(num_classes=10, pretrained=False)
    dummy_input = torch.randn(4, 3, 64, 64)
    output = model(dummy_input)

    assert torch.isfinite(output).all(), "ResNet-18 output contains NaN or Inf values!"


def test_resnet18_phase1_freezing_logic():
    """Verify Phase 1: Feature Extraction freezes all backbone parameters and leaves fc trainable."""
    model = ResNet18Transfer(num_classes=10, pretrained=False)
    model.freeze_for_phase1()

    total_params, trainable_params = model.count_parameters()
    expected_fc_params = 512 * 10 + 10  # weights (512*10) + bias (10) = 5,130
    assert trainable_params == expected_fc_params

    # Check each parameter individually
    for name, param in model.backbone.named_parameters():
        if "fc" in name:
            assert param.requires_grad is True, f"Classifier parameter {name} should be trainable in Phase 1!"
        else:
            assert param.requires_grad is False, f"Backbone parameter {name} should be frozen in Phase 1!"


def test_resnet18_phase2_unfreezing_logic():
    """Verify Phase 2: Fine-Tuning unfreezes layer3, layer4, and fc while keeping early layers frozen."""
    model = ResNet18Transfer(num_classes=10, pretrained=False)
    model.unfreeze_for_phase2(unfreeze_layers=("layer3", "layer4", "fc"))

    # Verify early layers remain frozen
    frozen_prefixes = ("conv1", "bn1", "layer1", "layer2")
    unfrozen_prefixes = ("layer3", "layer4", "fc")

    for name, param in model.backbone.named_parameters():
        if any(name.startswith(pfx) for pfx in frozen_prefixes):
            assert param.requires_grad is False, f"Early layer parameter {name} should be frozen in Phase 2!"
        elif any(name.startswith(pfx) for pfx in unfrozen_prefixes):
            assert param.requires_grad is True, f"High-level parameter {name} should be trainable in Phase 2!"

    total_params, trainable_params = model.count_parameters()
    assert trainable_params > 10_000_000
    assert trainable_params < total_params


def test_resnet18_parameter_groups():
    """Verify parameter group construction for differential learning rates."""
    model = ResNet18Transfer(num_classes=10, pretrained=False)
    model.unfreeze_for_phase2(["layer3", "layer4", "fc"])

    param_groups = model.get_parameter_groups(backbone_lr=5e-5, head_lr=2e-4, weight_decay=0.0001)
    assert len(param_groups) == 2
    assert param_groups[0]["lr"] == 5e-5
    assert param_groups[1]["lr"] == 2e-4


def test_build_model_factory_resnet18():
    """Verify model factory supports both custom_cnn and resnet18."""
    # Test ResNet-18 instantiation via factory
    config_resnet = {
        "model": {
            "name": "resnet18",
            "num_classes": 10,
            "pretrained": False,
            "dropout_rate": 0.0,
        }
    }
    model_resnet = build_model("resnet18", config=config_resnet)
    assert isinstance(model_resnet, ResNet18Transfer)
    assert model_resnet.num_classes == 10

    # Test Custom CNN still works via factory
    config_cnn = {
        "model": {
            "name": "custom_cnn",
            "num_classes": 10,
            "in_channels": 3,
            "conv_channels": [32, 64, 128],
            "dropout_rate": 0.3,
        }
    }
    model_cnn = build_model("custom_cnn", config=config_cnn)
    assert model_cnn.__class__.__name__ == "CustomCNN"

    # Test invalid model name
    with pytest.raises(ValueError, match="Unsupported model name"):
        build_model("vgg16")
