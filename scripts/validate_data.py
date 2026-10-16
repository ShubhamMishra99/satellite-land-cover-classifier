"""CLI script to validate integrity, dimensions, and classes of the EuroSAT dataset."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.constants import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.validator import DatasetValidator


def main():
    parser = argparse.ArgumentParser(description="Validate EuroSAT RGB dataset integrity and structure.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RAW_DATA_DIR / "EuroSAT_RGB",
        help="Path to the EuroSAT class directory (default: data/raw/EuroSAT_RGB)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROCESSED_DATA_DIR / "validation_report.json",
        help="Path to save JSON validation report (default: data/processed/validation_report.json)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional max samples per class to scan (for rapid testing)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("EUROSAT RGB DATASET INTEGRITY VALIDATION")
    print("=" * 60)

    if not args.data_dir.exists():
        print(f"ERROR: Dataset directory does not exist: {args.data_dir}")
        print("Please run 'python scripts/download_data.py' first.")
        return 1

    validator = DatasetValidator()
    report = validator.validate_dataset(
        data_dir=args.data_dir,
        sample_limit_per_class=args.sample_limit,
    )

    report.save_json(args.output_report)

    print("=" * 60)
    print(report.validation_summary)
    print("=" * 60)

    if report.is_valid:
        print(f"SUCCESS: Dataset is valid! Report written to {args.output_report}")
        return 0
    else:
        print(f"FAILED: Dataset validation issues detected. Check report: {args.output_report}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
