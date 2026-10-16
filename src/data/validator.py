"""Dataset validation and integrity verification for EuroSAT RGB (Multi-threaded)."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.constants import EUROSAT_CLASSES, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Structured validation report for EuroSAT dataset."""

    is_valid: bool
    total_images_scanned: int
    class_counts: Dict[str, int]
    missing_classes: List[str]
    unexpected_classes: List[str]
    corrupt_images: List[Dict[str, str]]
    invalid_dimension_images: List[Dict[str, Any]]
    invalid_channel_images: List[Dict[str, Any]]
    duplicate_image_count: int
    validation_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Validation report saved to: {output_path}")


class DatasetValidator:
    """High-throughput multi-threaded validator for EuroSAT dataset."""

    def __init__(
        self,
        expected_height: int = IMAGE_HEIGHT,
        expected_width: int = IMAGE_WIDTH,
        expected_channels: int = IMAGE_CHANNELS,
        expected_classes: Tuple[str, ...] = EUROSAT_CLASSES,
        num_workers: int = 16,
    ):
        self.expected_height = expected_height
        self.expected_width = expected_width
        self.expected_channels = expected_channels
        self.expected_classes = set(expected_classes)
        self.expected_num_classes = len(expected_classes)
        self.num_workers = num_workers

    def validate_directory_structure(self, data_dir: Path) -> Tuple[bool, List[str], List[str]]:
        """Validate presence of required class subdirectories."""
        if not data_dir.exists() or not data_dir.is_dir():
            return False, list(self.expected_classes), []

        found_subdirs = {p.name for p in data_dir.iterdir() if p.is_dir()}
        missing = sorted(list(self.expected_classes - found_subdirs))
        unexpected = sorted(list(found_subdirs - self.expected_classes))

        is_valid = (len(missing) == 0) and (len(unexpected) == 0)
        return is_valid, missing, unexpected

    def validate_single_image(self, image_path: Path) -> Tuple[bool, Optional[str], Optional[Tuple[int, int, int]], Optional[str], str]:
        """Verify image in a single pass.

        Returns:
            Tuple of (is_valid, error_reason, shape, md5_hash, file_rel_path)
        """
        rel_path = f"{image_path.parent.name}/{image_path.name}"
        try:
            raw_bytes = image_path.read_bytes()
            if len(raw_bytes) == 0:
                return False, "Zero byte file", None, None, rel_path
            file_hash = hashlib.md5(raw_bytes).hexdigest()
        except Exception as exc:
            return False, f"File read error: {exc}", None, None, rel_path

        # PIL validation
        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                img_mode = img.mode
                w, h = img.size
        except Exception as exc:
            return False, f"PIL decode error: {exc}", None, file_hash, rel_path

        if img_mode != "RGB":
            return False, f"Unexpected mode: {img_mode}", (h, w, len(img_mode)), file_hash, rel_path

        if h != self.expected_height or w != self.expected_width:
            return False, f"Unexpected dimensions: ({w}x{h})", (h, w, self.expected_channels), file_hash, rel_path

        # OpenCV decode
        try:
            np_buf = np.frombuffer(raw_bytes, dtype=np.uint8)
            cv_img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
            if cv_img is None:
                return False, "OpenCV returned None on imdecode", None, file_hash, rel_path
            c = cv_img.shape[2] if len(cv_img.shape) == 3 else 1
            if c != self.expected_channels:
                return False, f"Unexpected channel count: {c}", (h, w, c), file_hash, rel_path
        except Exception as exc:
            return False, f"OpenCV decode error: {exc}", None, file_hash, rel_path

        return True, None, (h, w, self.expected_channels), file_hash, rel_path

    def validate_dataset(
        self,
        data_dir: Path,
        sample_limit_per_class: Optional[int] = None,
        check_duplicates: bool = True,
    ) -> ValidationReport:
        """Run multi-threaded validation scan over the dataset directory.

        Args:
            data_dir: Directory containing class subfolders.
            sample_limit_per_class: Optional limit for rapid testing.
            check_duplicates: If True, tracks duplicate sample hashes.

        Returns:
            ValidationReport object.
        """
        logger.info(f"Starting multi-threaded dataset validation ({self.num_workers} workers) for: {data_dir}")
        struct_valid, missing_classes, unexpected_classes = self.validate_directory_structure(data_dir)

        class_counts: Dict[str, int] = {}
        all_image_paths: List[Path] = []

        for cls_name in sorted(list(self.expected_classes)):
            cls_dir = data_dir / cls_name
            if not cls_dir.exists():
                class_counts[cls_name] = 0
                continue

            image_files = sorted(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.tif*")))
            if sample_limit_per_class:
                image_files = image_files[:sample_limit_per_class]

            class_counts[cls_name] = len(image_files)
            all_image_paths.extend(image_files)

        total_scanned = len(all_image_paths)
        corrupt_images: List[Dict[str, str]] = []
        invalid_dim_images: List[Dict[str, Any]] = []
        invalid_chan_images: List[Dict[str, Any]] = []
        file_hashes: Dict[str, str] = {}
        duplicates: List[Tuple[str, str]] = []

        # Parallel validation across threads
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self.validate_single_image, p) for p in all_image_paths]
            for future in tqdm(as_completed(futures), total=total_scanned, desc="Validating Images", unit="img"):
                is_valid, err_reason, shape, file_hash, rel_path = future.result()

                if not is_valid:
                    if "dimensions" in (err_reason or ""):
                        invalid_dim_images.append({"file": rel_path, "error": err_reason, "shape": shape})
                    elif "channel" in (err_reason or ""):
                        invalid_chan_images.append({"file": rel_path, "error": err_reason, "shape": shape})
                    else:
                        corrupt_images.append({"file": rel_path, "error": err_reason})

                if check_duplicates and file_hash:
                    if file_hash in file_hashes:
                        duplicates.append((rel_path, file_hashes[file_hash]))
                    else:
                        file_hashes[file_hash] = rel_path

        # Summary evaluation
        has_all_classes = len(missing_classes) == 0
        has_min_images = total_scanned > 0
        no_corruptions = len(corrupt_images) == 0 and len(invalid_dim_images) == 0 and len(invalid_chan_images) == 0

        is_overall_valid = has_all_classes and has_min_images and no_corruptions

        summary_lines = [
            f"EuroSAT Dataset Validation: {'PASS' if is_overall_valid else 'FAIL'}",
            f"Total Images Scanned: {total_scanned:,}",
            f"Classes Present: {len(class_counts)}/{self.expected_num_classes}",
            f"Corrupt Files: {len(corrupt_images)}",
            f"Invalid Dimension Files: {len(invalid_dim_images)}",
            f"Invalid Channel Files: {len(invalid_chan_images)}",
            f"Duplicate Hash Count: {len(duplicates)}",
        ]
        summary_str = "\n".join(summary_lines)
        logger.info("\n" + summary_str)

        return ValidationReport(
            is_valid=is_overall_valid,
            total_images_scanned=total_scanned,
            class_counts=class_counts,
            missing_classes=missing_classes,
            unexpected_classes=unexpected_classes,
            corrupt_images=corrupt_images,
            invalid_dimension_images=invalid_dim_images,
            invalid_channel_images=invalid_chan_images,
            duplicate_image_count=len(duplicates),
            validation_summary=summary_str,
            metadata={
                "expected_shape": [self.expected_height, self.expected_width, self.expected_channels],
                "expected_classes": list(self.expected_classes),
            },
        )
