"""PyTorch Dataset implementation for EuroSAT RGB satellite imagery.

This module provides the `EuroSATDataset` class, which loads images lazily
from disk based on the deterministic, reproducible stratified partitions defined
in `data/processed/splits.json`.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.constants import (
    CLASS_TO_IDX,
    EUROSAT_CLASSES,
    IDX_TO_CLASS,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_CLASSES,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)

logger = logging.getLogger(__name__)


class EuroSATDataset(Dataset):
    """PyTorch Dataset for EuroSAT RGB Land-Use / Land-Cover classification.

    Loads images lazily from disk using pre-computed, stratified split manifests
    (zero data leakage between Train, Validation, and Test partitions).

    Attributes:
        split: Normalized partition name ('train', 'val', or 'test').
        root_dir: Absolute path to the raw EuroSAT RGB directory.
        splits_file: Path to the JSON split manifest.
        transform: Optional torchvision transform pipeline to apply to images.
        target_transform: Optional transform to apply to target labels.
        return_metadata: If True, `__getitem__` returns `(image, label, metadata)`.
        samples: List of sample dictionaries containing path, label, and class_name.
    """

    VALID_SPLITS = {"train", "val", "validation", "test"}

    def __init__(
        self,
        split: str = "train",
        root_dir: Optional[Union[str, Path]] = None,
        splits_file: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ):
        """Initialize EuroSATDataset.

        Args:
            split: Partition to load ('train', 'val', 'validation', or 'test').
            root_dir: Directory containing EuroSAT class folders (default: data/raw/EuroSAT_RGB).
            splits_file: Path to JSON split manifest (default: data/processed/splits.json).
            transform: Callable transform pipeline applied to PIL images.
            target_transform: Callable transform applied to class index labels.
            return_metadata: Whether to return sample metadata dict alongside image and label.

        Raises:
            ValueError: If split name is invalid or split manifest contains no samples.
            FileNotFoundError: If splits_file or root_dir does not exist.
        """
        norm_split = split.lower().strip()
        if norm_split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Expected one of: {sorted(list(self.VALID_SPLITS))}"
            )
        # Canonicalize 'validation' -> 'val'
        self.split = "val" if norm_split == "validation" else norm_split

        # Resolve paths
        self.root_dir = Path(root_dir) if root_dir is not None else (RAW_DATA_DIR / "EuroSAT_RGB")
        self.splits_file = (
            Path(splits_file) if splits_file is not None else (PROCESSED_DATA_DIR / "splits.json")
        )

        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = return_metadata

        # Validate existence of directory and splits file
        if not self.splits_file.exists():
            raise FileNotFoundError(
                f"Splits manifest not found at: {self.splits_file}. "
                "Please run 'scripts/create_splits.py' to generate deterministic partition indices."
            )

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset root directory not found at: {self.root_dir}. "
                "Please run 'scripts/download_data.py' to acquire the EuroSAT dataset."
            )

        # Load samples from split manifest
        self.samples: List[Dict[str, Any]] = self._load_split_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split '{self.split}' in manifest {self.splits_file}")

    def _load_split_samples(self) -> List[Dict[str, Any]]:
        """Load and validate sample entries for the target partition from JSON manifest."""
        try:
            with open(self.splits_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise ValueError(f"Failed to parse splits manifest at {self.splits_file}: {exc}") from exc

        if self.split not in data:
            raise ValueError(f"Partition key '{self.split}' not found in {self.splits_file}")

        samples = data[self.split]
        if not isinstance(samples, list):
            raise ValueError(f"Expected list of samples for partition '{self.split}', got {type(samples)}")

        # Validate entries
        for idx, sample in enumerate(samples):
            if "path" not in sample or "label" not in sample:
                raise ValueError(
                    f"Sample at index {idx} in '{self.split}' is missing required keys 'path' or 'label'."
                )
            label = sample["label"]
            if not isinstance(label, int) or label < 0 or label >= NUM_CLASSES:
                raise ValueError(
                    f"Sample at index {idx} has invalid class label '{label}' (expected integer 0-{NUM_CLASSES-1})."
                )

        return samples

    @property
    def classes(self) -> Tuple[str, ...]:
        """Return canonical EuroSAT class names."""
        return EUROSAT_CLASSES

    @property
    def class_to_idx(self) -> Dict[str, int]:
        """Return class name to integer index mapping."""
        return CLASS_TO_IDX

    @property
    def idx_to_class(self) -> Dict[int, str]:
        """Return integer index to class name mapping."""
        return IDX_TO_CLASS

    def get_class_distribution(self) -> Dict[str, int]:
        """Compute the sample count per class in this partition."""
        dist: Dict[str, int] = {cls_name: 0 for cls_name in EUROSAT_CLASSES}
        for sample in self.samples:
            cls_name = sample.get("class_name", IDX_TO_CLASS[sample["label"]])
            dist[cls_name] = dist.get(cls_name, 0) + 1
        return dist

    def __len__(self) -> int:
        """Return total number of samples in this split."""
        return len(self.samples)

    def _resolve_image_path(self, rel_or_abs_path: str) -> Path:
        """Resolve image path safely relative to root_dir."""
        p = Path(rel_or_abs_path)
        if p.is_absolute():
            return p
        return self.root_dir / p

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, Dict[str, Any]]]:
        """Retrieve a single sample by index (lazily loaded from disk).

        Args:
            idx: Sample index in partition.

        Returns:
            If return_metadata is False:
                (image, label) where image is Tensor[C, H, W] (if transform applied) or PIL.Image.
            If return_metadata is True:
                (image, label, metadata_dict)

        Raises:
            IndexError: If idx is out of bounds.
            FileNotFoundError: If image file does not exist on disk.
            RuntimeError: If image cannot be read or decoded.
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for EuroSATDataset split '{self.split}' of length {len(self)}")

        sample = self.samples[idx]
        image_path = self._resolve_image_path(sample["path"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file does not exist: {image_path} (referenced in {self.split} split index {idx})"
            )

        try:
            with Image.open(image_path) as img:
                # Convert to RGB mode to guarantee 3-channel standard representation
                image = img.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to read and decode image file {image_path}: {exc}") from exc

        label = sample["label"]

        # Apply transformations if specified
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.return_metadata:
            metadata = {
                "path": str(image_path),
                "relative_path": sample["path"],
                "class_name": sample.get("class_name", IDX_TO_CLASS[label]),
                "index": idx,
                "split": self.split,
            }
            return image, label, metadata

        return image, label

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split='{self.split}', "
            f"num_samples={len(self)}, "
            f"root_dir='{self.root_dir}', "
            f"transform={self.transform is not None})"
        )
