"""CLI script to download and extract EuroSAT RGB dataset."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.constants import RAW_DATA_DIR
from src.data.download import acquire_eurosat_dataset


def main():
    parser = argparse.ArgumentParser(description="Download and extract the EuroSAT RGB dataset.")
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help=f"Destination directory for raw dataset (default: {RAW_DATA_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset files already exist",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EUROSAT RGB DATASET ACQUISITION")
    print("=" * 60)
    try:
        dataset_path = acquire_eurosat_dataset(dest_dir=args.dest_dir, force_download=args.force)
        print("=" * 60)
        print(f"SUCCESS: EuroSAT RGB dataset ready at: {dataset_path}")
        print("=" * 60)
        return 0
    except Exception as exc:
        print("=" * 60)
        print(f"ERROR: Dataset acquisition failed: {exc}")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
