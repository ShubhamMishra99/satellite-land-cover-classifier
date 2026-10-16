"""Unit tests for single-image inference pipeline, image validation, and prediction formatting."""

import json
from pathlib import Path
import tempfile
import pytest
from PIL import Image
import torch
import torch.nn as nn

from src.constants import EUROSAT_CLASSES, NUM_CLASSES
from src.inference.predictor import (
    DOMAIN_WARNING_MESSAGE,
    SUPPORTED_IMAGE_EXTENSIONS,
    load_and_preprocess_image,
    load_trained_model,
    predict_image,
    predict_single_tensor,
    validate_image_path,
)
from src.models.custom_cnn import CustomCNN
from src.models.resnet18 import ResNet18Transfer
from src.training.checkpoint import save_checkpoint


@pytest.fixture
def sample_rgb_image(tmp_path: Path) -> Path:
    """Create a temporary valid RGB JPEG image."""
    img_path = tmp_path / "sample_test.jpg"
    img = Image.new("RGB", (100, 120), color=(45, 120, 200))
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_rgba_image(tmp_path: Path) -> Path:
    """Create a temporary RGBA PNG image."""
    img_path = tmp_path / "sample_rgba.png"
    img = Image.new("RGBA", (80, 80), color=(100, 200, 50, 255))
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_grayscale_image(tmp_path: Path) -> Path:
    """Create a temporary grayscale image."""
    img_path = tmp_path / "sample_gray.png"
    img = Image.new("L", (64, 64), color=128)
    img.save(img_path)
    return img_path


def test_validate_image_path_valid(sample_rgb_image: Path):
    """Verify validate_image_path resolves valid image path."""
    resolved = validate_image_path(sample_rgb_image)
    assert resolved.exists()
    assert resolved.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def test_validate_image_path_missing():
    """Verify validate_image_path raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        validate_image_path("non_existent_image_12345.jpg")


def test_validate_image_path_unsupported_ext(tmp_path: Path):
    """Verify validate_image_path raises ValueError for unsupported extensions."""
    bad_file = tmp_path / "document.pdf"
    bad_file.write_text("dummy")
    with pytest.raises(ValueError, match="Unsupported image format"):
        validate_image_path(bad_file)


def test_load_and_preprocess_image_rgb(sample_rgb_image: Path):
    """Verify load_and_preprocess_image returns tensor of shape [1, 3, 64, 64] and original dimensions."""
    tensor, orig_size = load_and_preprocess_image(sample_rgb_image)
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == torch.float32
    assert orig_size == (100, 120)


def test_load_and_preprocess_image_rgba_conversion(sample_rgba_image: Path):
    """Verify RGBA images are converted to 3-channel RGB without crashing."""
    tensor, orig_size = load_and_preprocess_image(sample_rgba_image)
    assert tensor.shape == (1, 3, 64, 64)
    assert orig_size == (80, 80)


def test_load_and_preprocess_image_grayscale_conversion(sample_grayscale_image: Path):
    """Verify Grayscale images are converted to 3-channel RGB tensor."""
    tensor, orig_size = load_and_preprocess_image(sample_grayscale_image)
    assert tensor.shape == (1, 3, 64, 64)


def test_class_mapping_has_10_classes():
    """Verify canonical EuroSAT class mapping has exactly 10 classes."""
    assert len(EUROSAT_CLASSES) == 10
    assert NUM_CLASSES == 10


def test_predict_single_tensor_probabilities():
    """Verify predict_single_tensor returns probabilities that sum to ~1.0."""
    model = CustomCNN(num_classes=10)
    model.eval()

    dummy_tensor = torch.randn(1, 3, 64, 64)
    results = predict_single_tensor(dummy_tensor, model=model, top_k=3)

    assert results["predicted_class"] in EUROSAT_CLASSES
    assert 0.0 <= results["confidence"] <= 1.0
    assert len(results["top3"]) == 3

    # Check top3 ranking order
    probs = [item["probability"] for item in results["top3"]]
    assert probs == sorted(probs, reverse=True)

    # Check that sum of all class probabilities equals 1.0
    all_probs_sum = sum(results["all_probabilities"].values())
    assert pytest.approx(all_probs_sum, abs=1e-3) == 1.0


def test_custom_cnn_inference_with_synthetic_checkpoint(tmp_path: Path, sample_rgb_image: Path):
    """Verify Custom CNN inference pipeline with a synthetic model checkpoint."""
    model = CustomCNN(num_classes=10)
    ckpt_path = tmp_path / "custom_cnn_mock.pt"
    save_checkpoint(
        model=model,
        epoch=1,
        best_val_macro_f1=0.85,
        config={"model": {"name": "custom_cnn", "num_classes": 10}},
        filepath=ckpt_path,
    )

    loaded_model = load_trained_model(
        model_name="custom_cnn",
        checkpoint_path=ckpt_path,
        device=torch.device("cpu"),
    )
    assert isinstance(loaded_model, CustomCNN)

    results, orig_size = predict_image(sample_rgb_image, model=loaded_model, device=torch.device("cpu"))
    assert results["predicted_class"] in EUROSAT_CLASSES
    assert len(results["top3"]) == 3
    assert orig_size == (100, 120)


def test_resnet18_inference_with_synthetic_checkpoint(tmp_path: Path, sample_rgb_image: Path):
    """Verify ResNet-18 inference pipeline with a synthetic checkpoint (pretrained=False for fast test)."""
    model = ResNet18Transfer(num_classes=10, pretrained=False)
    ckpt_path = tmp_path / "resnet18_mock.pt"
    save_checkpoint(
        model=model,
        epoch=1,
        best_val_macro_f1=0.92,
        config={"model": {"name": "resnet18", "num_classes": 10, "pretrained": False}},
        filepath=ckpt_path,
    )

    loaded_model = load_trained_model(
        model_name="resnet18",
        checkpoint_path=ckpt_path,
        device=torch.device("cpu"),
    )
    assert isinstance(loaded_model, ResNet18Transfer)

    results, orig_size = predict_image(sample_rgb_image, model=loaded_model, device=torch.device("cpu"))
    assert results["predicted_class"] in EUROSAT_CLASSES
    assert len(results["top3"]) == 3


def test_both_models_comparison(tmp_path: Path, sample_rgb_image: Path):
    """Verify running both Custom CNN and ResNet-18 side-by-side on the same input tensor."""
    cnn = CustomCNN(num_classes=10)
    resnet = ResNet18Transfer(num_classes=10, pretrained=False)

    tensor, _ = load_and_preprocess_image(sample_rgb_image)

    res_cnn = predict_single_tensor(tensor, model=cnn)
    res_resnet = predict_single_tensor(tensor, model=resnet)

    assert "predicted_class" in res_cnn
    assert "predicted_class" in res_resnet
    assert isinstance(res_cnn["predicted_class"] == res_resnet["predicted_class"], bool)
