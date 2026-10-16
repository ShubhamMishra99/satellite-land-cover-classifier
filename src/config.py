"""Configuration loader and schema validator for Satellite Land-Use Classifier."""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from src.constants import CONFIGS_DIR, PROJECT_ROOT


def load_yaml(config_path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file safely.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Parsed dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If YAML parsing fails.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML file {config_path}: {exc}") from exc


def get_base_config() -> Dict[str, Any]:
    """Load base configuration."""
    return load_yaml(CONFIGS_DIR / "base_config.yaml")


def get_custom_cnn_config() -> Dict[str, Any]:
    """Load Custom CNN configuration."""
    return load_yaml(CONFIGS_DIR / "custom_cnn.yaml")


def get_resnet18_config() -> Dict[str, Any]:
    """Load ResNet-18 configuration."""
    return load_yaml(CONFIGS_DIR / "resnet18.yaml")
