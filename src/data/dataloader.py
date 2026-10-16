"""PyTorch DataLoader factory and partition loader utilities for EuroSAT RGB.

This module provides convenient factories for creating training, validation,
and testing DataLoaders with deterministic seeding, configurable batch size,
worker counts, pinning, and transform pipelines.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from src.constants import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.data.dataset import EuroSATDataset
from src.preprocessing.transforms import get_transforms

logger = logging.getLogger(__name__)


@dataclass
class DataLoadersContainer:
    """Container holding DataLoaders and underlying datasets for all partitions."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_dataset: EuroSATDataset
    val_dataset: EuroSATDataset
    test_dataset: EuroSATDataset

    def __getitem__(self, split: str) -> DataLoader:
        norm_split = split.lower().strip()
        if norm_split == "train":
            return self.train
        elif norm_split in ("val", "validation"):
            return self.val
        elif norm_split == "test":
            return self.test
        else:
            raise KeyError(f"Unknown split key: '{split}'. Expected 'train', 'val', or 'test'.")


def create_dataloader(
    split: str,
    root_dir: Optional[Union[str, Path]] = None,
    splits_file: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    pin_memory: bool = False,
    drop_last: bool = False,
    seed: int = 42,
    norm_type: str = "eurosat",
    config: Optional[Dict[str, Any]] = None,
    return_metadata: bool = False,
) -> Tuple[DataLoader, EuroSATDataset]:
    """Create a PyTorch DataLoader and EuroSATDataset for a specific partition.

    Args:
        split: Partition to load ('train', 'val', or 'test').
        root_dir: Directory containing EuroSAT class folders.
        splits_file: Path to JSON split manifest.
        batch_size: Batch size for DataLoader.
        num_workers: Subprocesses for data loading (0 recommended on Windows).
        shuffle: Whether to shuffle data. If None, True for train, False for val/test.
        pin_memory: If True, copy Tensors into CUDA pinned memory before returning.
        drop_last: If True, drop the last incomplete batch.
        seed: Random seed for generator reproducibility.
        norm_type: Normalization preset ('eurosat' or 'imagenet').
        config: Optional full project configuration dictionary.
        return_metadata: If True, batches include sample metadata dictionary.

    Returns:
        Tuple of (DataLoader, EuroSATDataset).
    """
    norm_split = split.lower().strip()
    is_train = norm_split == "train"

    # Default shuffle behavior
    if shuffle is None:
        shuffle = is_train

    # Obtain transform pipeline
    transform = get_transforms(split=norm_split, config=config, norm_type=norm_type)

    # Initialize Dataset
    dataset = EuroSATDataset(
        split=norm_split,
        root_dir=root_dir,
        splits_file=splits_file,
        transform=transform,
        return_metadata=return_metadata,
    )

    # Create generator for deterministic reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator if shuffle else None,
    )

    return loader, dataset


def create_dataloaders(
    config: Optional[Dict[str, Any]] = None,
    root_dir: Optional[Union[str, Path]] = None,
    splits_file: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last_train: bool = False,
    seed: int = 42,
    norm_type: str = "eurosat",
) -> DataLoadersContainer:
    """Create DataLoaders for Train, Validation, and Test partitions.

    Args:
        config: Optional project configuration dictionary. If provided, parameters
                are extracted from config where available.
        root_dir: Directory containing EuroSAT class folders.
        splits_file: Path to JSON split manifest.
        batch_size: Default batch size (can be overridden by config).
        num_workers: Number of worker processes (default 0 for Windows compatibility).
        pin_memory: Whether to pin memory for CUDA transfer.
        drop_last_train: Whether to drop the last incomplete batch in training.
        seed: Random seed for deterministic shuffling.
        norm_type: Normalization preset ('eurosat' or 'imagenet').

    Returns:
        DataLoadersContainer holding train, val, and test DataLoaders and datasets.
    """
    # Parse options from config if provided
    if config is not None:
        seed = config.get("seed", seed)
        dl_cfg = config.get("dataloader", {})
        batch_size = dl_cfg.get("batch_size", config.get("training", {}).get("batch_size", batch_size))
        num_workers = dl_cfg.get("num_workers", num_workers)
        pin_memory = dl_cfg.get("pin_memory", pin_memory)
        drop_last_train = dl_cfg.get("drop_last_train", drop_last_train)

    train_loader, train_ds = create_dataloader(
        split="train",
        root_dir=root_dir,
        splits_file=splits_file,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        seed=seed,
        norm_type=norm_type,
        config=config,
    )

    val_loader, val_ds = create_dataloader(
        split="val",
        root_dir=root_dir,
        splits_file=splits_file,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        drop_last=False,
        seed=seed,
        norm_type=norm_type,
        config=config,
    )

    test_loader, test_ds = create_dataloader(
        split="test",
        root_dir=root_dir,
        splits_file=splits_file,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
        drop_last=False,
        seed=seed,
        norm_type=norm_type,
        config=config,
    )

    logger.info(
        f"DataLoaders Created -> Train: {len(train_ds)} samples ({len(train_loader)} batches), "
        f"Val: {len(val_ds)} samples ({len(val_loader)} batches), "
        f"Test: {len(test_ds)} samples ({len(test_loader)} batches), "
        f"Batch Size: {batch_size}, Seed: {seed}"
    )

    return DataLoadersContainer(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
    )
