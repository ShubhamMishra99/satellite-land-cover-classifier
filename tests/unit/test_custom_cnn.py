"""Unit tests for Custom CNN baseline architecture and model factory."""

import pytest
import torch

from src.constants import IMAGE_CHANNELS, NUM_CLASSES
from src.models.custom_cnn import CustomCNN
from src.models import build_model


def test_custom_cnn_instantiation():
    """Verify default instantiation and parameter initialization."""
    model = CustomCNN()
    assert isinstance(model, torch.nn.Module)
    assert model.in_channels == IMAGE_CHANNELS
    assert model.num_classes == NUM_CLASSES
    assert model.conv_channels == (32, 64, 128)
    assert model.dropout_rate == 0.3

    total_params, trainable_params = model.count_parameters()
    assert total_params > 0
    assert total_params == trainable_params
    assert "CustomCNN" in repr(model)


def test_custom_cnn_forward_shape():
    """Verify input shape [2, 3, 64, 64] produces logits shape [2, 10]."""
    model = CustomCNN()
    model.eval()

    dummy_input = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 10)
    assert output.dtype == torch.float32


def test_custom_cnn_batch_size_preservation():
    """Verify arbitrary batch sizes are preserved accurately."""
    model = CustomCNN()
    model.eval()

    for batch_size in [1, 4, 8, 16]:
        dummy_input = torch.randn(batch_size, 3, 64, 64)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (batch_size, 10)


def test_custom_cnn_outputs_finite_values():
    """Verify that model outputs finite real numbers without NaN or Inf."""
    model = CustomCNN()
    dummy_input = torch.randn(4, 3, 64, 64)
    output = model(dummy_input)

    assert torch.isfinite(output).all(), "Model output contains NaN or Inf values!"


def test_custom_cnn_no_softmax_applied():
    """Verify that raw logits are returned (not probabilities summing to 1 or strictly in [0, 1])."""
    model = CustomCNN()
    dummy_input = torch.randn(5, 3, 64, 64)
    logits = model(dummy_input)

    # Logits should generally not sum to 1 across classes
    row_sums = logits.sum(dim=-1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums)), (
        "Softmax was incorrectly applied inside the model forward pass!"
    )


def test_custom_cnn_configurable_classes():
    """Verify custom number of classes and channels."""
    model = CustomCNN(in_channels=3, num_classes=5, conv_channels=[16, 32], dropout_rate=0.1)
    dummy_input = torch.randn(2, 3, 64, 64)
    output = model(dummy_input)

    assert output.shape == (2, 5)


def test_build_model_factory():
    """Verify build_model helper for custom_cnn."""
    config = {
        "model": {
            "name": "custom_cnn",
            "in_channels": 3,
            "num_classes": 10,
            "conv_channels": [32, 64, 128],
            "dropout_rate": 0.3,
        }
    }

    model = build_model("custom_cnn", config=config)
    assert isinstance(model, CustomCNN)
    assert model.num_classes == 10

    with pytest.raises(ValueError, match="Unsupported model name"):
        build_model("unknown_architecture")
