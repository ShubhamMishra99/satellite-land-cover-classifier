"""Constants and single source of truth for Satellite Land-Use Classifier.

This module defines immutable constants, canonical EuroSAT class names,
class index mappings, color palettes for visualization, and default paths.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# Base Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

# EuroSAT RGB Dataset Specifications
IMAGE_HEIGHT: int = 64
IMAGE_WIDTH: int = 64
IMAGE_CHANNELS: int = 3
NUM_CLASSES: int = 10

# Canonical EuroSAT RGB Class Names (Alphabetical order standard)
EUROSAT_CLASSES: Tuple[str, ...] = (
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
)

# Class-to-Index and Index-to-Class Mappings
CLASS_TO_IDX: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(EUROSAT_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls_name for idx, cls_name in enumerate(EUROSAT_CLASSES)}

# Official EuroSAT / ImageNet Normalization Constants
# Sentinel-2 EuroSAT RGB channel statistics (RGB approx.) and ImageNet stats
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

EUROSAT_MEAN: Tuple[float, float, float] = (0.3444, 0.3803, 0.4078)
EUROSAT_STD: Tuple[float, float, float] = (0.2026, 0.1368, 0.1156)

# UI & Visualization Hex Colors per Land-Use Class
CLASS_COLORS: Dict[str, str] = {
    "AnnualCrop": "#E6C229",            # Golden yellow
    "Forest": "#2E7D32",                # Deep forest green
    "HerbaceousVegetation": "#81C784",  # Light meadow green
    "Highway": "#78909C",               # Asphalt blue-grey
    "Industrial": "#D81B60",            # Crimson / factory red
    "Pasture": "#AEEA00",               # Lime grassland
    "PermanentCrop": "#FB8C00",         # Warm orchard orange
    "Residential": "#8E24AA",           # Urban violet
    "River": "#0288D1",                 # Flowing blue
    "SeaLake": "#01579B",               # Deep marine navy
}

# Empirical Maximum Softmax Probability (MSP) Uncertainty Thresholds
# Evaluated strictly on the EuroSAT validation split (4,050 samples):
# - ResNet-18 at 0.60 achieves 96.42% acceptance (3.58% false rejection)
# - Custom CNN at 0.40 achieves 88.35% acceptance (11.65% false rejection)
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.60

MODEL_CONFIDENCE_THRESHOLDS: Dict[str, float] = {
    "resnet18": 0.60,
    "custom_cnn": 0.40,
}

# Image Suitability and Out-of-Distribution (OOD) Detection Constants
SUITABILITY_PROTOTYPES_PATH = MODELS_DIR / "suitability_prototypes.pt"
SUITABILITY_DISTANCE_THRESHOLD: float = 0.38
SUITABILITY_MIN_VARIANCE: float = 1e-4
SUITABILITY_MIN_ENTROPY: float = 0.5
SUITABILITY_MAX_LAPLACIAN_VAR: float = 0.35

SUITABILITY_UNSUPPORTED_TITLE: str = "Unsupported Image"
SUITABILITY_UNSUPPORTED_MESSAGE: str = (
    "This image does not appear suitable for satellite land-use classification. "
    "Please upload a satellite image containing one of the supported EuroSAT land-use classes."
)

