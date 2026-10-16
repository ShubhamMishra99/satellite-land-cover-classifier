"""Automated EuroSAT RGB dataset acquisition, extraction, and fallback mechanisms."""

import logging
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

import requests
from tqdm import tqdm

from src.constants import EUROSAT_CLASSES, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EUROSAT_RGB_URLS: List[str] = [
    "https://huggingface.co/datasets/blanchon/EuroSAT_RGB/resolve/main/EuroSAT_RGB.zip",
    "https://zenodo.org/api/records/7711810/files/EuroSAT_RGB.zip/content",
    "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
]


def download_file_stream(url: str, output_path: Path, chunk_size: int = 1024 * 1024, max_retries: int = 3) -> bool:
    """Download a file with streaming chunks, retries, and a tqdm progress bar.

    Args:
        url: Remote URL.
        output_path: Local destination path.
        chunk_size: Bytes per chunk (default: 1MB).
        max_retries: Number of retry attempts on connection drops.

    Returns:
        True if download succeeded and verified, False otherwise.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_suffix(".tmp")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for attempt in range(1, max_retries + 1):
        logger.info(f"Download attempt {attempt}/{max_retries} from: {url}")
        try:
            with requests.get(url, headers=headers, stream=True, timeout=30) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                with open(temp_output, "wb") as f, tqdm(
                    desc=output_path.name,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            # Verify downloaded zip integrity
            if zipfile.is_zipfile(temp_output):
                if output_path.exists():
                    output_path.unlink()
                temp_output.rename(output_path)
                logger.info(f"Downloaded and verified zip archive: {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB)")
                return True
            else:
                logger.warning(f"Downloaded file is not a valid zip archive (attempt {attempt}).")
                if temp_output.exists():
                    temp_output.unlink()
        except Exception as exc:
            logger.warning(f"Download error on attempt {attempt}: {exc}")
            if temp_output.exists():
                temp_output.unlink()

    return False


def download_via_torchvision(dest_dir: Path) -> bool:
    """Fallback mechanism: acquire EuroSAT using torchvision dataset downloader."""
    try:
        logger.info("Attempting dataset download via torchvision.datasets.EuroSAT...")
        from torchvision.datasets import EuroSAT
        dataset = EuroSAT(root=str(dest_dir), download=True)
        logger.info(f"Torchvision EuroSAT successfully downloaded {len(dataset)} items.")
        return True
    except Exception as exc:
        logger.warning(f"Torchvision download fallback failed: {exc}")
        return False


def extract_zip(zip_path: Path, target_dir: Path) -> Path:
    """Extract a ZIP archive and standardize the directory hierarchy."""
    logger.info(f"Extracting {zip_path.name} to {target_dir}...")
    temp_extract_dir = target_dir / "_temp_extract"
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_extract_dir)

    # Locate class folder
    extracted_roots = [p for p in temp_extract_dir.rglob("*") if p.is_dir()]
    source_class_dir: Optional[Path] = None

    for candidate in [temp_extract_dir] + extracted_roots:
        subdirs = {p.name for p in candidate.iterdir() if p.is_dir()}
        if any(cls_name in subdirs for cls_name in EUROSAT_CLASSES):
            source_class_dir = candidate
            break

    if source_class_dir is None:
        raise RuntimeError(f"Could not locate EuroSAT class folders within extracted archive: {zip_path}")

    # Standardize to target_dir/EuroSAT_RGB/<class_name>
    final_dataset_dir = target_dir / "EuroSAT_RGB"
    final_dataset_dir.mkdir(parents=True, exist_ok=True)

    for cls_path in source_class_dir.iterdir():
        if cls_path.is_dir() and cls_path.name in EUROSAT_CLASSES:
            dest_cls_dir = final_dataset_dir / cls_path.name
            if dest_cls_dir.exists():
                shutil.rmtree(dest_cls_dir)
            shutil.move(str(cls_path), str(dest_cls_dir))

    # Clean up temporary extraction folder
    shutil.rmtree(temp_extract_dir, ignore_errors=True)
    logger.info(f"Dataset extracted and standardized to: {final_dataset_dir}")
    return final_dataset_dir


def acquire_eurosat_dataset(dest_dir: Optional[Path] = None, force_download: bool = False) -> Path:
    """Acquire EuroSAT RGB dataset if not already present."""
    dest_dir = dest_dir or RAW_DATA_DIR
    dataset_dir = dest_dir / "EuroSAT_RGB"

    # Check if dataset already exists and has all 10 classes
    if not force_download and dataset_dir.exists():
        existing_classes = [c.name for c in dataset_dir.iterdir() if c.is_dir()]
        if all(cls_name in existing_classes for cls_name in EUROSAT_CLASSES):
            total_images = sum(len(list(c.glob("*.jpg"))) for c in dataset_dir.iterdir() if c.is_dir())
            if total_images >= 20000:
                logger.info(f"EuroSAT RGB dataset already present at {dataset_dir} ({total_images} images found).")
                return dataset_dir

    # Also check if torchvision extracted to data/raw/eurosat/2750
    tv_dir = dest_dir / "eurosat" / "2750"
    if not force_download and tv_dir.exists():
        logger.info(f"Standardizing torchvision download from {tv_dir} to {dataset_dir}...")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for cls_p in tv_dir.iterdir():
            if cls_p.is_dir() and cls_p.name in EUROSAT_CLASSES:
                dest_p = dataset_dir / cls_p.name
                if dest_p.exists():
                    shutil.rmtree(dest_p)
                shutil.move(str(cls_p), str(dest_p))
        return dataset_dir

    zip_file = dest_dir / "EuroSAT.zip"

    # Attempt direct download across mirrors
    download_success = False
    for url in EUROSAT_RGB_URLS:
        if download_file_stream(url, zip_file):
            download_success = True
            break

    if download_success and zip_file.exists():
        extracted_dir = extract_zip(zip_file, dest_dir)
        if zip_file.exists():
            zip_file.unlink()
        return extracted_dir

    # Fallback to torchvision
    if download_via_torchvision(dest_dir):
        # Locate and standardize
        for candidate in [dest_dir / "eurosat" / "2750", dest_dir / "2750"]:
            if candidate.exists():
                dataset_dir.mkdir(parents=True, exist_ok=True)
                for cls_p in candidate.iterdir():
                    if cls_p.is_dir() and cls_p.name in EUROSAT_CLASSES:
                        dest_p = dataset_dir / cls_p.name
                        if dest_p.exists():
                            shutil.rmtree(dest_p)
                        shutil.move(str(cls_p), str(dest_p))
                return dataset_dir

    raise RuntimeError(
        "Failed to acquire EuroSAT dataset from all mirrors and torchvision. "
        "Please check your internet connection or manually place 'EuroSAT.zip' in data/raw/"
    )
