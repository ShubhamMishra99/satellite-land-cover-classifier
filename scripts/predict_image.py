"""CLI inference tool for Satellite Land-Use Classifier.

Allows testing individual satellite images using trained Custom CNN and ResNet-18 checkpoints,
with Top-3 probability ranking, side-by-side model comparison, and optional JSON export.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.constants import EUROSAT_CLASSES, MODELS_DIR
from src.inference.predictor import (
    DOMAIN_WARNING_MESSAGE,
    load_and_preprocess_image,
    load_trained_model,
    predict_single_tensor,
    validate_image_path,
)


def format_top3_table(top3_list: list) -> str:
    """Format Top-3 predictions into aligned text."""
    lines = []
    for item in top3_list:
        lines.append(f"  {item['rank']}. {item['class']:<24} {item['percentage']:>7}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run inference on an individual satellite image using trained EuroSAT classifiers.",
        epilog=(
            "Examples:\n"
            "  python scripts/predict_image.py --image data/raw/EuroSAT_RGB/Forest/Forest_1.jpg --model resnet18\n"
            "  python scripts/predict_image.py --image data/raw/EuroSAT_RGB/Forest/Forest_1.jpg --model custom_cnn\n"
            "  python scripts/predict_image.py --image data/raw/EuroSAT_RGB/Forest/Forest_1.jpg --model both\n"
            "  python scripts/predict_image.py --image sample.jpg --model both --label Forest --save-json output.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to satellite image file (.jpg, .jpeg, .png, .bmp, .tif, .tiff)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet18", "custom_cnn", "both"],
        default="resnet18",
        help="Model architecture to run: 'resnet18', 'custom_cnn', or 'both' for comparison (default: resnet18)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional expected ground-truth class name for individual visual verification",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to export prediction results as machine-readable JSON",
    )
    parser.add_argument(
        "--custom-cnn-checkpoint",
        type=Path,
        default=None,
        help="Custom checkpoint path for Custom CNN (default: models/custom_cnn_best.pt)",
    )
    parser.add_argument(
        "--resnet18-checkpoint",
        type=Path,
        default=None,
        help="Custom checkpoint path for ResNet-18 (default: models/resnet18_best.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Computation device ('cpu' or 'cuda'). Defaults to auto-detection.",
    )

    args = parser.parse_args()

    # 1. Device Setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Image Validation & Preprocessing
    try:
        input_tensor, orig_size = load_and_preprocess_image(args.image)
    except Exception as exc:
        print(f"\n[ERROR] Failed to load input image: {exc}", file=sys.stderr)
        return 1

    image_path_str = str(Path(args.image))
    device_name = "CUDA" if device.type == "cuda" else "CPU"

    # Optional label validation
    ground_truth = args.label.strip() if args.label else None
    if ground_truth and ground_truth not in EUROSAT_CLASSES:
        print(
            f"\n[WARNING] Supplied label '{ground_truth}' is not one of canonical EuroSAT classes: "
            f"{list(EUROSAT_CLASSES)}",
            file=sys.stderr,
        )

    # Dictionary for structured JSON export
    json_output: Dict[str, Any] = {
        "image": image_path_str,
        "device": device.type,
        "original_size": [orig_size[0], orig_size[1]],
        "model_input_size": [64, 64],
        "models": {},
    }
    if ground_truth:
        json_output["ground_truth"] = ground_truth

    # 3. Model Execution
    models_to_run = ["custom_cnn", "resnet18"] if args.model == "both" else [args.model]
    predictions: Dict[str, Dict[str, Any]] = {}

    for m_name in models_to_run:
        ckpt = args.custom_cnn_checkpoint if m_name == "custom_cnn" else args.resnet18_checkpoint
        try:
            model = load_trained_model(
                model_name=m_name,
                checkpoint_path=ckpt,
                device=device,
            )
        except Exception as exc:
            print(f"\n[ERROR] Failed to load model '{m_name}': {exc}", file=sys.stderr)
            return 1

        pred_dict = predict_single_tensor(
            tensor=input_tensor,
            model=model,
            device=device,
            top_k=3,
        )
        predictions[m_name] = pred_dict
        json_output["models"][m_name] = {
            "predicted_class": pred_dict["predicted_class"],
            "confidence": pred_dict["confidence"],
            "top3": [
                {"class": item["class"], "probability": item["probability"]}
                for item in pred_dict["top3"]
            ],
            "all_probabilities": pred_dict["all_probabilities"],
        }

    # 4. Formatted Terminal Output
    print()
    if args.model == "both":
        print("=" * 60)
        print("                 MODEL COMPARISON")
        print("=" * 60)
        print(f"Image               : {image_path_str}")
        print(f"Device              : {device_name}")
        print(f"Original Image Size : {orig_size[0]}x{orig_size[1]}")
        print(f"Model Input Size    : 64x64 (Converted to RGB)")
        print("-" * 60)

        cnn_pred = predictions["custom_cnn"]
        res_pred = predictions["resnet18"]

        print(f"\nCustom CNN:")
        print(f"  Prediction : {cnn_pred['predicted_class']}")
        print(f"  Confidence : {cnn_pred['confidence_percentage']}")
        print(f"  Top 3:")
        print(format_top3_table(cnn_pred["top3"]))

        print(f"\nResNet-18:")
        print(f"  Prediction : {res_pred['predicted_class']}")
        print(f"  Confidence : {res_pred['confidence_percentage']}")
        print(f"  Top 3:")
        print(format_top3_table(res_pred["top3"]))

        agreement = cnn_pred["predicted_class"] == res_pred["predicted_class"]
        json_output["agreement"] = agreement
        print("-" * 60)
        print(f"Agreement : {'YES' if agreement else 'NO'}")
        if not agreement:
            print(f"  Custom CNN Prediction : {cnn_pred['predicted_class']}")
            print(f"  ResNet-18 Prediction  : {res_pred['predicted_class']}")

        if ground_truth:
            print("-" * 60)
            print(f"Ground Truth          : {ground_truth}")
            print(f"Custom CNN Correct    : {'YES' if cnn_pred['predicted_class'] == ground_truth else 'NO'}")
            print(f"ResNet-18 Correct     : {'YES' if res_pred['predicted_class'] == ground_truth else 'NO'}")

        print("=" * 60)

    else:
        title = "RESNET-18 PREDICTION" if args.model == "resnet18" else "CUSTOM CNN PREDICTION"
        pred = predictions[args.model]

        print("=" * 60)
        print(f"              {title}")
        print("=" * 60)
        print(f"Image               : {image_path_str}")
        print(f"Device              : {device_name}")
        print(f"Original Image Size : {orig_size[0]}x{orig_size[1]}")
        print(f"Model Input Size    : 64x64 (Converted to RGB)")
        print("-" * 60)
        print(f"Predicted class     : {pred['predicted_class']}")
        print(f"Confidence          : {pred['confidence_percentage']}")
        print("\nTop 3 predictions:\n")
        print(format_top3_table(pred["top3"]))

        if ground_truth:
            is_correct = pred["predicted_class"] == ground_truth
            print("-" * 60)
            print(f"Ground truth        : {ground_truth}")
            print(f"Prediction          : {pred['predicted_class']}")
            print(f"Correct             : {'YES' if is_correct else 'NO'}")

        print("=" * 60)

    # 5. Domain Warning
    print(f"\n{DOMAIN_WARNING_MESSAGE}\n")

    # 6. Save JSON if requested
    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2)
        print(f"[SAVED] Prediction results exported to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
