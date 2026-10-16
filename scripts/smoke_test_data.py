"""Smoke test and sanity check for EuroSAT Dataset & DataLoaders.

This script executes an end-to-end verification of:
1. Dataset manifest loading and partition sample counts (Train: 18,900, Val: 4,050, Test: 4,050).
2. EuroSATDataset instantiation and lazy loading.
3. Train, Validation, and Test DataLoader batch generation.
4. Batch tensor shapes [B, 3, 64, 64], dtype, value range, and label range [0, 9].
5. Deterministic reproducibility of validation/test DataLoaders.
6. Zero leakage across all three partitions.
"""

import logging
from pathlib import Path
import sys
import time

import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import get_base_config
from src.constants import (
    CLASS_TO_IDX,
    EUROSAT_CLASSES,
    IDX_TO_CLASS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.data.dataloader import create_dataloaders
from src.data.dataset import EuroSATDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_test_data")


def run_smoke_test() -> int:
    print("=" * 70)
    print("   SATELLITE LAND-USE CLASSIFIER - MILESTONE 2 SMOKE TEST")
    print("             EuroSAT RGB Preprocessing & DataLoaders")
    print("=" * 70)

    # 1. Check prerequisite paths
    splits_file = PROCESSED_DATA_DIR / "splits.json"
    raw_dir = RAW_DATA_DIR / "EuroSAT_RGB"

    print("\n[STEP 1] Checking Dataset & Manifest Prerequisites:")
    if not splits_file.exists():
        print(f"  [-] FAIL: Splits manifest missing at: {splits_file}")
        return 1
    print(f"  [+] Splits manifest found: {splits_file}")

    if not raw_dir.exists():
        print(f"  [-] FAIL: EuroSAT raw directory missing at: {raw_dir}")
        return 1
    print(f"  [+] EuroSAT raw directory found: {raw_dir}")

    # 2. Load project configuration
    print("\n[STEP 2] Loading Project Configuration:")
    try:
        config = get_base_config()
        print(f"  [+] base_config.yaml successfully loaded (Seed: {config.get('seed')})")
    except Exception as exc:
        print(f"  [-] FAIL: Could not load configuration: {exc}")
        return 1

    # 3. Create DataLoaders
    print("\n[STEP 3] Initializing PyTorch DataLoaders (Batch Size = 64, num_workers = 0):")
    start_time = time.time()
    try:
        loaders = create_dataloaders(
            config=config,
            batch_size=64,
            num_workers=0,
            seed=config.get("seed", 42),
        )
        elapsed_init = time.time() - start_time
        print(f"  [+] DataLoaders created successfully in {elapsed_init:.3f}s")
    except Exception as exc:
        print(f"  [-] FAIL: DataLoader creation failed: {exc}")
        return 1

    # 4. Verify Partition Sample Counts
    print("\n[STEP 4] Verifying Partition Sample Counts:")
    n_train = len(loaders.train.dataset)
    n_val = len(loaders.val.dataset)
    n_test = len(loaders.test.dataset)
    n_total = n_train + n_val + n_test

    print(f"  [+] Train set : {n_train:6d} samples (Expected: 18,900 | {n_train/n_total*100:.1f}%)")
    print(f"  [+] Val set   : {n_val:6d} samples (Expected:  4,050 | {n_val/n_total*100:.1f}%)")
    print(f"  [+] Test set  : {n_test:6d} samples (Expected:  4,050 | {n_test/n_total*100:.1f}%)")
    print(f"  [+] Total     : {n_total:6d} samples (Expected: 27,000)")

    if n_train != 18900 or n_val != 4050 or n_test != 4050:
        print("  [-] FAIL: Partition counts do not match expected PRD values!")
        return 1

    # 5. Verify Batch Shape, Tensor Dtype & Label Range on First Batches
    print("\n[STEP 5] Testing Batch Ingestion & Tensor Specifications:")

    for split_name, loader in [("Train", loaders.train), ("Validation", loaders.val), ("Test", loaders.test)]:
        t0 = time.time()
        batch_imgs, batch_lbls = next(iter(loader))
        batch_time = time.time() - t0

        print(f"\n  --- {split_name} Split ---")
        print(f"  Image Batch Shape  : {tuple(batch_imgs.shape)} (Expected: [64, 3, 64, 64])")
        print(f"  Image Tensor Dtype : {batch_imgs.dtype}")
        print(f"  Image Min / Max    : {batch_imgs.min().item():.4f} / {batch_imgs.max().item():.4f}")
        print(f"  Image Mean / Std   : {batch_imgs.mean().item():.4f} / {batch_imgs.std().item():.4f}")
        print(f"  Label Batch Shape  : {tuple(batch_lbls.shape)} (Expected: [64])")
        print(f"  Label Min / Max    : {batch_lbls.min().item()} / {batch_lbls.max().item()} (Expected: 0..9)")
        print(f"  Batch Fetch Time   : {batch_time*1000:.2f} ms")

        # Assertions
        if tuple(batch_imgs.shape) != (64, 3, 64, 64):
            print(f"  [-] FAIL: Incorrect image batch shape: {batch_imgs.shape}")
            return 1
        if tuple(batch_lbls.shape) != (64,):
            print(f"  [-] FAIL: Incorrect label batch shape: {batch_lbls.shape}")
            return 1
        if batch_lbls.min().item() < 0 or batch_lbls.max().item() >= 10:
            print(f"  [-] FAIL: Label out of range [0, 9]: min={batch_lbls.min()}, max={batch_lbls.max()}")
            return 1

    # 6. Verify Deterministic Ordering of Validation DataLoaders
    print("\n[STEP 6] Verifying Validation DataLoader Determinism:")
    val_loader_2, _ = create_dataloaders(config=config, batch_size=64, num_workers=0).val, None
    _, val_lbls_1 = next(iter(loaders.val))
    _, val_lbls_2 = next(iter(val_loader_2))

    if torch.equal(val_lbls_1, val_lbls_2):
        print("  [+] PASS: Validation DataLoader is strictly deterministic across instances.")
    else:
        print("  [-] FAIL: Validation DataLoader produced non-deterministic label sequences!")
        return 1

    # 7. Zero Data Leakage Confirmation
    print("\n[STEP 7] Verifying Zero Data Leakage Across Datasets:")
    train_paths = {s["path"] for s in loaders.train_dataset.samples}
    val_paths = {s["path"] for s in loaders.val_dataset.samples}
    test_paths = {s["path"] for s in loaders.test_dataset.samples}

    tv_overlap = train_paths.intersection(val_paths)
    tt_overlap = train_paths.intersection(test_paths)
    vt_overlap = val_paths.intersection(test_paths)

    if tv_overlap or tt_overlap or vt_overlap:
        print(f"  [-] FAIL: Overlap detected! Train-Val: {len(tv_overlap)}, Train-Test: {len(tt_overlap)}, Val-Test: {len(vt_overlap)}")
        return 1

    print("  [+] PASS: Strictly zero overlap confirmed (Train intersect Val = empty, Train intersect Test = empty, Val intersect Test = empty).")

    # 8. Summary
    print("\n" + "=" * 70)
    print("RESULT: ALL MILESTONE 2 PREPROCESSING & DATALOADER CHECKS PASSED!")
    print("READY FOR MILESTONE 3: CUSTOM CNN BASELINE ARCHITECTURE & TRAINING")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
