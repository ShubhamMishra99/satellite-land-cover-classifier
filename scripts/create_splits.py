"""CLI script to generate reproducible stratified train/val/test splits."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.constants import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.split import SplitManager


def main():
    parser = argparse.ArgumentParser(description="Create reproducible train/val/test splits for EuroSAT.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_DATA_DIR / "EuroSAT_RGB",
        help="Path to the EuroSAT class directory (default: data/raw/EuroSAT_RGB)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=PROCESSED_DATA_DIR / "splits.json",
        help="Output path for splits manifest (default: data/processed/splits.json)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train set ratio (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation set ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    print("=" * 60)
    print("EUROSAT DATASET PARTITIONING & STRATIFIED SPLITS")
    print("=" * 60)

    if not args.data_dir.exists():
        print(f"ERROR: Dataset directory does not exist: {args.data_dir}")
        print("Please download the dataset first with: python scripts/download_data.py")
        return 1

    manager = SplitManager(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    try:
        splits = manager.create_stratified_splits(args.data_dir)
        splits.save_json(args.output_file)

        print("=" * 60)
        print("SPLIT SUMMARY:")
        print(f"  Methodology : {splits.methodology}")
        print(f"  Total Samples: {splits.total_samples}")
        print(f"  Train Samples: {splits.train_count} ({splits.train_count/splits.total_samples*100:.1f}%)")
        print(f"  Val Samples  : {splits.val_count} ({splits.val_count/splits.total_samples*100:.1f}%)")
        print(f"  Test Samples : {splits.test_count} ({splits.test_count/splits.total_samples*100:.1f}%)")
        print(f"  Manifest     : {args.output_file}")
        print("=" * 60)
        print("SUCCESS: Zero-leakage stratified splits generated and verified.")
        return 0
    except Exception as exc:
        print(f"ERROR: Split generation failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
