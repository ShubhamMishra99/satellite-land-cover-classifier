"""Script to evaluate confidence distributions on EuroSAT validation partition.
Analyzes maximum Softmax probabilities (MSP) for ResNet-18 and Custom CNN.
Evaluates candidate thresholds and computes acceptance rates / false rejection rates.
"""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.constants import EUROSAT_CLASSES, MODELS_DIR
from src.data.dataloader import create_dataloader
from src.inference.predictor import load_trained_model


def evaluate_model_confidence_on_val(model_name: str, device: torch.device):
    print(f"\n--- Evaluating {model_name} on validation split ---")
    val_loader, _ = create_dataloader(
        split="val",
        batch_size=128,
        num_workers=0,
        shuffle=False,
    )

    model = load_trained_model(model_name=model_name, device=device)
    model.eval()

    all_confidences = []
    all_correct = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch[0].to(device)
            targets = batch[1].to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)

            all_confidences.extend(max_probs.cpu().numpy().tolist())
            all_correct.extend((preds == targets).cpu().numpy().tolist())

    confidences = np.array(all_confidences)
    correct = np.array(all_correct)

    mean_conf = float(np.mean(confidences))
    median_conf = float(np.median(confidences))
    p1 = float(np.percentile(confidences, 1))
    p5 = float(np.percentile(confidences, 5))
    p10 = float(np.percentile(confidences, 10))
    p25 = float(np.percentile(confidences, 25))

    print(f"Total samples: {len(confidences)}")
    print(f"Accuracy: {np.mean(correct) * 100:.2f}%")
    print(f"Confidence Mean: {mean_conf:.4f}, Median: {median_conf:.4f}")
    print(f"Percentiles - 1st: {p1:.4f}, 5th: {p5:.4f}, 10th: {p10:.4f}, 25th: {p25:.4f}")

    thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    print("\nCandidate Threshold Evaluation (Validation Split):")
    print(f"{'Threshold':>10} | {'Accepted':>10} | {'Accept % (Recall)':>18} | {'False Rejection %':>20}")
    print("-" * 65)

    thresh_stats = {}
    for t in thresholds:
        accepted_mask = confidences >= t
        accepted_count = int(np.sum(accepted_mask))
        accepted_pct = float(accepted_count / len(confidences) * 100)
        false_rejection_pct = 100.0 - accepted_pct
        thresh_stats[str(t)] = {
            "accepted_count": accepted_count,
            "acceptance_rate": accepted_pct,
            "false_rejection_rate": false_rejection_pct,
        }
        print(f"{t:>10.2f} | {accepted_count:>10} | {accepted_pct:>17.2f}% | {false_rejection_pct:>19.2f}%")

    return {
        "model": model_name,
        "total_samples": len(confidences),
        "mean_conf": mean_conf,
        "median_conf": median_conf,
        "p1": p1,
        "p5": p5,
        "p10": p10,
        "thresholds": thresh_stats,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    resnet_results = evaluate_model_confidence_on_val("resnet18", device)
    cnn_results = evaluate_model_confidence_on_val("custom_cnn", device)

    out_path = Path("scratch/val_confidence_evaluation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"resnet18": resnet_results, "custom_cnn": cnn_results}, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
