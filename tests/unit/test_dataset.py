"""Unit tests for dataset validation, integrity checks, splitting, and EuroSATDataset."""

import json
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import pytest
import torch

from src.constants import (
    CLASS_TO_IDX,
    EUROSAT_CLASSES,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.data.dataset import EuroSATDataset
from src.data.split import SplitManager
from src.data.validator import DatasetValidator
from src.preprocessing.transforms import get_eval_transforms, get_train_transforms


@pytest.fixture
def mock_eurosat_dataset(tmp_path: Path) -> Path:
    """Create a temporary mock EuroSAT dataset with synthetic 64x64 RGB images."""
    data_dir = tmp_path / "EuroSAT_RGB"
    data_dir.mkdir()

    # Create 10 class folders with 10 synthetic images each
    for cls_name in EUROSAT_CLASSES:
        cls_dir = data_dir / cls_name
        cls_dir.mkdir()
        for i in range(10):
            # Create a valid 64x64 RGB image
            arr = np.random.randint(0, 255, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)
            img = Image.fromarray(arr, mode="RGB")
            img.save(cls_dir / f"{cls_name}_{i}.jpg")

    return data_dir


@pytest.fixture
def mock_dataset_and_splits(mock_eurosat_dataset: Path, tmp_path: Path) -> Tuple[Path, Path]:
    """Create a temporary mock EuroSAT dataset and corresponding splits.json."""
    manager = SplitManager(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    splits = manager.create_stratified_splits(mock_eurosat_dataset)
    splits_file = tmp_path / "splits.json"
    splits.save_json(splits_file)
    return mock_eurosat_dataset, splits_file


# --- Milestone 1 Validator & Split Tests ---


def test_validator_valid_dataset(mock_eurosat_dataset: Path):
    """Test that DatasetValidator passes a structurally sound dataset."""
    validator = DatasetValidator()
    report = validator.validate_dataset(mock_eurosat_dataset)

    assert report.is_valid is True
    assert report.total_images_scanned == 100
    assert len(report.missing_classes) == 0
    assert len(report.unexpected_classes) == 0
    assert len(report.corrupt_images) == 0
    assert len(report.invalid_dimension_images) == 0
    assert len(report.invalid_channel_images) == 0


def test_validator_detects_missing_class(mock_eurosat_dataset: Path):
    """Test that validator catches missing class directories."""
    import shutil
    shutil.rmtree(mock_eurosat_dataset / "River")

    validator = DatasetValidator()
    report = validator.validate_dataset(mock_eurosat_dataset)

    assert report.is_valid is False
    assert "River" in report.missing_classes


def test_validator_detects_corrupt_and_wrong_dim(mock_eurosat_dataset: Path):
    """Test that validator detects corrupt files and invalid dimensions."""
    cls_dir = mock_eurosat_dataset / "Forest"

    # 1. Add a corrupt image (garbage bytes)
    corrupt_file = cls_dir / "Forest_corrupt.jpg"
    corrupt_file.write_bytes(b"NOT_AN_IMAGE_DATA_12345")

    # 2. Add an image with wrong dimensions (32x32)
    wrong_dim_file = cls_dir / "Forest_wrong_dim.jpg"
    small_img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    small_img.save(wrong_dim_file)

    validator = DatasetValidator()
    report = validator.validate_dataset(mock_eurosat_dataset)

    assert report.is_valid is False
    assert len(report.corrupt_images) >= 1
    assert len(report.invalid_dimension_images) >= 1


def test_split_manager_zero_leakage(mock_eurosat_dataset: Path, tmp_path: Path):
    """Test that SplitManager generates disjoint stratified splits with exact ratios."""
    manager = SplitManager(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    splits = manager.create_stratified_splits(mock_eurosat_dataset)

    assert splits.total_samples == 100
    assert splits.train_count == 70
    assert splits.val_count == 15
    assert splits.test_count == 15

    # Check zero overlap
    train_paths = {x["path"] for x in splits.train}
    val_paths = {x["path"] for x in splits.val}
    test_paths = {x["path"] for x in splits.test}

    assert len(train_paths.intersection(val_paths)) == 0
    assert len(train_paths.intersection(test_paths)) == 0
    assert len(val_paths.intersection(test_paths)) == 0

    # Save and reload manifest
    out_file = tmp_path / "splits.json"
    splits.save_json(out_file)
    assert out_file.exists()

    with open(out_file, "r") as f:
        data = json.load(f)
    assert data["total_samples"] == 100
    assert len(data["train"]) == 70


# --- Milestone 2 EuroSATDataset Tests ---


def test_eurosat_dataset_mock_initialization(mock_dataset_and_splits: Tuple[Path, Path]):
    """Test EuroSATDataset loads correct counts for all splits on mock data."""
    data_dir, splits_file = mock_dataset_and_splits

    train_ds = EuroSATDataset(split="train", root_dir=data_dir, splits_file=splits_file)
    val_ds = EuroSATDataset(split="val", root_dir=data_dir, splits_file=splits_file)
    test_ds = EuroSATDataset(split="test", root_dir=data_dir, splits_file=splits_file)

    assert len(train_ds) == 70
    assert len(val_ds) == 15
    assert len(test_ds) == 15

    # Verify property access
    assert train_ds.classes == EUROSAT_CLASSES
    assert train_ds.class_to_idx == CLASS_TO_IDX
    assert len(train_ds.get_class_distribution()) == 10


def test_eurosat_dataset_getitem_output(mock_dataset_and_splits: Tuple[Path, Path]):
    """Test __getitem__ returns correct tensor shape and label range."""
    data_dir, splits_file = mock_dataset_and_splits
    transform = get_eval_transforms()

    dataset = EuroSATDataset(
        split="train",
        root_dir=data_dir,
        splits_file=splits_file,
        transform=transform,
    )

    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert isinstance(label, int)
    assert 0 <= label < NUM_CLASSES


def test_eurosat_dataset_return_metadata(mock_dataset_and_splits: Tuple[Path, Path]):
    """Test __getitem__ with return_metadata=True."""
    data_dir, splits_file = mock_dataset_and_splits
    transform = get_eval_transforms()

    dataset = EuroSATDataset(
        split="val",
        root_dir=data_dir,
        splits_file=splits_file,
        transform=transform,
        return_metadata=True,
    )

    image, label, meta = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, int)
    assert isinstance(meta, dict)
    assert "path" in meta
    assert "class_name" in meta
    assert "split" in meta
    assert meta["split"] == "val"


def test_eurosat_dataset_invalid_split(mock_dataset_and_splits: Tuple[Path, Path]):
    """Test EuroSATDataset raises ValueError on invalid split name."""
    data_dir, splits_file = mock_dataset_and_splits
    with pytest.raises(ValueError, match="Invalid split"):
        EuroSATDataset(split="unknown_split", root_dir=data_dir, splits_file=splits_file)


def test_eurosat_dataset_missing_files(tmp_path: Path):
    """Test EuroSATDataset raises FileNotFoundError when files or directories are missing."""
    non_existent_file = tmp_path / "does_not_exist_splits.json"
    non_existent_dir = tmp_path / "does_not_exist_dir"

    with pytest.raises(FileNotFoundError, match="Splits manifest not found"):
        EuroSATDataset(split="train", root_dir=tmp_path, splits_file=non_existent_file)

    valid_splits_file = tmp_path / "dummy_splits.json"
    valid_splits_file.write_text(json.dumps({"train": []}))

    with pytest.raises(FileNotFoundError, match="Dataset root directory not found"):
        EuroSATDataset(split="train", root_dir=non_existent_dir, splits_file=valid_splits_file)


def test_eurosat_dataset_missing_image_file(mock_dataset_and_splits: Tuple[Path, Path]):
    """Test EuroSATDataset raises FileNotFoundError when an individual image file is missing."""
    data_dir, splits_file = mock_dataset_and_splits

    # Delete an image that is in the split
    with open(splits_file, "r") as f:
        data = json.load(f)
    first_train_img_rel = data["train"][0]["path"]
    (data_dir / first_train_img_rel).unlink()

    dataset = EuroSATDataset(split="train", root_dir=data_dir, splits_file=splits_file)
    with pytest.raises(FileNotFoundError, match="Image file does not exist"):
        _ = dataset[0]


# --- Real Dataset Integration Verification ---

@pytest.mark.skipif(
    not (RAW_DATA_DIR / "EuroSAT_RGB").exists() or not (PROCESSED_DATA_DIR / "splits.json").exists(),
    reason="Real EuroSAT dataset or splits.json not present on disk",
)
def test_real_eurosat_dataset_counts_and_dimensions():
    """Verify EuroSATDataset with real 27,000 image dataset on disk."""
    train_ds = EuroSATDataset(split="train", transform=get_train_transforms())
    val_ds = EuroSATDataset(split="val", transform=get_eval_transforms())
    test_ds = EuroSATDataset(split="test", transform=get_eval_transforms())

    # Exact split counts per PRD
    assert len(train_ds) == 18900, f"Expected 18,900 train samples, got {len(train_ds)}"
    assert len(val_ds) == 4050, f"Expected 4,050 val samples, got {len(val_ds)}"
    assert len(test_ds) == 4050, f"Expected 4,050 test samples, got {len(test_ds)}"
    assert len(train_ds) + len(val_ds) + len(test_ds) == 27000

    # Test item loading and shapes
    img_train, lbl_train = train_ds[0]
    assert img_train.shape == (3, 64, 64)
    assert 0 <= lbl_train < 10

    img_val, lbl_val = val_ds[0]
    assert img_val.shape == (3, 64, 64)
    assert 0 <= lbl_val < 10

    img_test, lbl_test = test_ds[0]
    assert img_test.shape == (3, 64, 64)
    assert 0 <= lbl_test < 10
