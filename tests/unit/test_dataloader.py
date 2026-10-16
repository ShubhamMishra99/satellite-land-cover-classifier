"""Unit tests for EuroSAT DataLoader factory and partition batching."""

from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from src.constants import (
    EUROSAT_CLASSES,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.data.dataloader import DataLoadersContainer, create_dataloader, create_dataloaders
from src.data.split import SplitManager


@pytest.fixture
def mock_dataset_and_splits(tmp_path: Path) -> Tuple[Path, Path]:
    """Create a temporary mock EuroSAT dataset and corresponding splits.json."""
    data_dir = tmp_path / "EuroSAT_RGB"
    data_dir.mkdir()

    for cls_name in EUROSAT_CLASSES:
        cls_dir = data_dir / cls_name
        cls_dir.mkdir()
        for i in range(10):
            arr = np.random.randint(0, 255, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)
            img = Image.fromarray(arr, mode="RGB")
            img.save(cls_dir / f"{cls_name}_{i}.jpg")

    manager = SplitManager(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    splits = manager.create_stratified_splits(data_dir)
    splits_file = tmp_path / "splits.json"
    splits.save_json(splits_file)

    return data_dir, splits_file


def test_create_single_dataloader(mock_dataset_and_splits: Tuple[Path, Path]):
    """Test create_dataloader creates a functional DataLoader for a single split."""
    data_dir, splits_file = mock_dataset_and_splits

    loader, dataset = create_dataloader(
        split="train",
        root_dir=data_dir,
        splits_file=splits_file,
        batch_size=16,
        num_workers=0,
    )

    assert isinstance(loader, DataLoader)
    assert len(dataset) == 70

    batch = next(iter(loader))
    images, labels = batch

    assert images.shape == (16, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert labels.shape == (16,)
    assert labels.dtype == torch.int64


def test_create_dataloaders_container(mock_dataset_and_splits: Tuple[Path, Path]):
    """Test create_dataloaders returns container with train, val, and test loaders."""
    data_dir, splits_file = mock_dataset_and_splits

    loaders = create_dataloaders(
        root_dir=data_dir,
        splits_file=splits_file,
        batch_size=8,
        num_workers=0,
        seed=42,
    )

    assert isinstance(loaders, DataLoadersContainer)
    assert len(loaders.train.dataset) == 70
    assert len(loaders.val.dataset) == 15
    assert len(loaders.test.dataset) == 15

    # Check indexing
    assert loaders["train"] is loaders.train
    assert loaders["val"] is loaders.val
    assert loaders["test"] is loaders.test

    with pytest.raises(KeyError):
        _ = loaders["unknown_split"]


def test_val_test_dataloaders_deterministic_order(mock_dataset_and_splits: Tuple[Path, Path]):
    """Test that validation and test DataLoaders return batches in deterministic order."""
    data_dir, splits_file = mock_dataset_and_splits

    loader_1, _ = create_dataloader(
        split="val",
        root_dir=data_dir,
        splits_file=splits_file,
        batch_size=15,
        shuffle=False,
    )
    loader_2, _ = create_dataloader(
        split="val",
        root_dir=data_dir,
        splits_file=splits_file,
        batch_size=15,
        shuffle=False,
    )

    _, labels_1 = next(iter(loader_1))
    _, labels_2 = next(iter(loader_2))

    assert torch.equal(labels_1, labels_2), "Validation DataLoader labels were not in deterministic order!"


def test_dataloaders_from_config(mock_dataset_and_splits: Tuple[Path, Path]):
    """Test create_dataloaders configuring options from project config dict."""
    data_dir, splits_file = mock_dataset_and_splits

    config = {
        "seed": 123,
        "dataset": {"image_size": [64, 64]},
        "dataloader": {
            "batch_size": 16,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last_train": True,
        },
    }

    loaders = create_dataloaders(
        config=config,
        root_dir=data_dir,
        splits_file=splits_file,
    )

    assert loaders.train.batch_size == 16
    assert loaders.train.drop_last is True


# --- Real EuroSAT DataLoader Tests ---

@pytest.mark.skipif(
    not (RAW_DATA_DIR / "EuroSAT_RGB").exists() or not (PROCESSED_DATA_DIR / "splits.json").exists(),
    reason="Real EuroSAT dataset or splits.json not present on disk",
)
def test_real_dataset_dataloaders_batch_loading():
    """Verify batch iteration on real 27,000 image EuroSAT dataset."""
    loaders = create_dataloaders(batch_size=64, num_workers=0, seed=42)

    # Train batch test
    train_imgs, train_lbls = next(iter(loaders.train))
    assert train_imgs.shape == (64, 3, 64, 64)
    assert train_lbls.shape == (64,)
    assert train_lbls.min() >= 0 and train_lbls.max() < 10

    # Val batch test
    val_imgs, val_lbls = next(iter(loaders.val))
    assert val_imgs.shape == (64, 3, 64, 64)
    assert val_lbls.shape == (64,)

    # Test batch test
    test_imgs, test_lbls = next(iter(loaders.test))
    assert test_imgs.shape == (64, 3, 64, 64)
    assert test_lbls.shape == (64,)
