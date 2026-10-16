"""Unit tests for project setup, constants, and configuration loading."""

from pathlib import Path
import pytest

from src.constants import (
    EUROSAT_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    NUM_CLASSES,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
    PROJECT_ROOT,
    CLASS_COLORS,
)
from src.config import get_base_config, get_custom_cnn_config, get_resnet18_config


def test_eurosat_classes_integrity():
    """Verify that EuroSAT class constants adhere to the PRD specifications."""
    assert NUM_CLASSES == 10
    assert len(EUROSAT_CLASSES) == 10
    assert len(CLASS_TO_IDX) == 10
    assert len(IDX_TO_CLASS) == 10

    # Ensure all 10 canonical classes are present
    expected_classes = [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    ]
    assert list(EUROSAT_CLASSES) == expected_classes

    # Ensure bidirectional mapping is consistent
    for idx, name in enumerate(EUROSAT_CLASSES):
        assert CLASS_TO_IDX[name] == idx
        assert IDX_TO_CLASS[idx] == name
        assert name in CLASS_COLORS


def test_image_dimensions():
    """Verify standard EuroSAT tile dimensions."""
    assert IMAGE_HEIGHT == 64
    assert IMAGE_WIDTH == 64
    assert IMAGE_CHANNELS == 3


def test_config_loading():
    """Verify YAML configuration files load correctly."""
    base_cfg = get_base_config()
    assert base_cfg["project"]["name"] == "satellite-land-use-classifier"
    assert base_cfg["dataset"]["num_classes"] == 10
    assert len(base_cfg["classes"]) == 10

    cnn_cfg = get_custom_cnn_config()
    assert cnn_cfg["model"]["name"] == "custom_cnn"
    assert cnn_cfg["model"]["num_classes"] == 10

    resnet_cfg = get_resnet18_config()
    assert resnet_cfg["model"]["name"] == "resnet18_transfer"
    assert resnet_cfg["model"]["num_classes"] == 10
