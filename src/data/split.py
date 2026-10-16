"""Reproducible Stratified Train/Val/Test Split Generator & Validator.

Official EuroSAT Distribution Split Analysis:
--------------------------------------------
The official EuroSAT dataset archive distributed by DFKI (EuroSAT.zip) contains
27,000 RGB images grouped into 10 class directories ('2750/'). The archive itself
does NOT provide bundled official partition index files (train/val/test split CSV/JSON).

To ensure complete scientific rigor and zero data leakage:
1. We document this absence transparently rather than fabricating an 'official' split.
2. We implement a deterministic, stratified split with a fixed random seed (default: 42).
3. We guarantee exact zero overlap across partitions:
   (Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅).
4. We preserve exact class balance across Train (70%), Val (15%), and Test (15%).
5. We persist the exact split manifest as `data/processed/splits.json` so every subsequent
   training, evaluation, and test run uses identical partition indices.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.constants import CLASS_TO_IDX, EUROSAT_CLASSES, PROCESSED_DATA_DIR, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SplitItem:
    """Single sample entry in a split."""

    path: str  # Relative path to dataset root (e.g. "AnnualCrop/AnnualCrop_1.jpg")
    label: int
    class_name: str


@dataclass
class DatasetSplits:
    """Container for train, val, test partitions and metadata."""

    methodology: str
    seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    total_samples: int
    train_count: int
    val_count: int
    test_count: int
    class_distribution: Dict[str, Dict[str, int]]
    train: List[Dict[str, Any]]
    val: List[Dict[str, Any]]
    test: List[Dict[str, Any]]

    def save_json(self, output_path: Path) -> None:
        """Persist split manifest to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Dataset splits saved to: {output_path}")


class SplitManager:
    """Manager for generating and verifying reproducible EuroSAT partitions."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(
                f"Split ratios must sum to 1.0 (got {train_ratio} + {val_ratio} + {test_ratio} = {train_ratio + val_ratio + test_ratio})"
            )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def check_for_official_splits(self, data_dir: Path) -> Optional[Path]:
        """Check if official split files exist in the dataset directory."""
        candidates = [
            data_dir / "train.csv",
            data_dir / "splits.json",
            data_dir / "train.txt",
            data_dir.parent / "splits.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                logger.info(f"Found existing split file at: {candidate}")
                return candidate
        return None

    def create_stratified_splits(self, data_dir: Path) -> DatasetSplits:
        """Generate deterministic, stratified splits with zero overlap.

        Args:
            data_dir: Directory containing class subfolders (e.g. data/raw/EuroSAT_RGB).

        Returns:
            DatasetSplits object containing train, val, and test partitions.
        """
        all_items: List[SplitItem] = []

        # Collect all valid image paths across all 10 canonical classes
        for cls_name in EUROSAT_CLASSES:
            cls_dir = data_dir / cls_name
            if not cls_dir.exists():
                raise FileNotFoundError(f"Class folder not found: {cls_dir}")

            label_idx = CLASS_TO_IDX[cls_name]
            image_files = sorted(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.tif*")))

            for img_path in image_files:
                rel_path = f"{cls_name}/{img_path.name}"
                all_items.append(SplitItem(path=rel_path, label=label_idx, class_name=cls_name))

        if len(all_items) == 0:
            raise ValueError(f"No image files found in {data_dir}")

        logger.info(f"Collected {len(all_items)} total images across {len(EUROSAT_CLASSES)} classes.")

        # Extract features (indices) and stratification labels
        indices = np.arange(len(all_items))
        labels = np.array([item.label for item in all_items])

        # Step 1: Split into Train and (Val + Test)
        val_test_ratio = self.val_ratio + self.test_ratio
        train_idx, val_test_idx = train_test_split(
            indices,
            test_size=val_test_ratio,
            stratify=labels,
            random_state=self.seed,
            shuffle=True,
        )

        # Step 2: Split (Val + Test) into Val and Test
        val_proportion_of_val_test = self.val_ratio / val_test_ratio
        val_idx, test_idx = train_test_split(
            val_test_idx,
            train_size=val_proportion_of_val_test,
            stratify=labels[val_test_idx],
            random_state=self.seed,
            shuffle=True,
        )

        # Build partition lists
        train_items = [asdict(all_items[i]) for i in train_idx]
        val_items = [asdict(all_items[i]) for i in val_idx]
        test_items = [asdict(all_items[i]) for i in test_idx]

        # Verify strict zero leakage
        self.verify_zero_leakage(train_items, val_items, test_items)

        # Compute per-class distribution summary
        class_dist: Dict[str, Dict[str, int]] = {cls_name: {"train": 0, "val": 0, "test": 0, "total": 0} for cls_name in EUROSAT_CLASSES}
        for item in train_items:
            class_dist[item["class_name"]]["train"] += 1
            class_dist[item["class_name"]]["total"] += 1
        for item in val_items:
            class_dist[item["class_name"]]["val"] += 1
            class_dist[item["class_name"]]["total"] += 1
        for item in test_items:
            class_dist[item["class_name"]]["test"] += 1
            class_dist[item["class_name"]]["total"] += 1

        methodology_doc = (
            f"Reproducible Stratified Fixed Split (Train={self.train_ratio*100:.0f}%, "
            f"Val={self.val_ratio*100:.0f}%, Test={self.test_ratio*100:.0f}%, seed={self.seed}). "
            "Note: Official EuroSAT ZIP release does not bundle pre-split partition files."
        )

        splits = DatasetSplits(
            methodology=methodology_doc,
            seed=self.seed,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            total_samples=len(all_items),
            train_count=len(train_items),
            val_count=len(val_items),
            test_count=len(test_items),
            class_distribution=class_dist,
            train=train_items,
            val=val_items,
            test=test_items,
        )

        logger.info(
            f"Splits Created Successfully -> Train: {len(train_items)}, Val: {len(val_items)}, Test: {len(test_items)} "
            f"(Total: {len(all_items)})"
        )
        return splits

    @staticmethod
    def verify_zero_leakage(
        train: List[Dict[str, Any]],
        val: List[Dict[str, Any]],
        test: List[Dict[str, Any]],
    ) -> None:
        """Mathematically verify that train, val, and test sets have zero overlap."""
        train_paths: Set[str] = {x["path"] for x in train}
        val_paths: Set[str] = {x["path"] for x in val}
        test_paths: Set[str] = {x["path"] for x in test}

        train_val_overlap = train_paths.intersection(val_paths)
        train_test_overlap = train_paths.intersection(test_paths)
        val_test_overlap = val_paths.intersection(test_paths)

        if train_val_overlap:
            raise AssertionError(f"Data Leakage detected! {len(train_val_overlap)} overlapping items between Train and Val.")
        if train_test_overlap:
            raise AssertionError(f"Data Leakage detected! {len(train_test_overlap)} overlapping items between Train and Test.")
        if val_test_overlap:
            raise AssertionError(f"Data Leakage detected! {len(val_test_overlap)} overlapping items between Val and Test.")

        logger.info("[VERIFICATION] Zero data leakage confirmed: Train, Val, and Test partitions are strictly disjoint.")


def load_splits(splits_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load previously generated split manifest JSON.

    Args:
        splits_path: Path to splits.json (defaults to data/processed/splits.json).

    Returns:
        Dictionary containing split partitions and metadata.
    """
    path = splits_path or (PROCESSED_DATA_DIR / "splits.json")
    if not path.exists():
        raise FileNotFoundError(f"Splits file not found at: {path}. Please run 'scripts/create_splits.py' first.")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
