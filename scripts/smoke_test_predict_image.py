"""Smoke test for Milestone 4.5: Real Image Inference Pipeline."""

import logging
from pathlib import Path
import sys
import tempfile
from PIL import Image
import torch

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.constants import EUROSAT_CLASSES, MODELS_DIR
from src.inference.predictor import (
    load_and_preprocess_image,
    load_trained_model,
    predict_image,
    validate_image_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("smoke_test_predict_image")


def run_smoke_test() -> bool:
    print("=" * 70)
    print("   MILESTONE 4.5: IMAGE INFERENCE SMOKE TEST SUITE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Environment] Active Device: {device}")

    # 1. Create a synthetic test image
    print("\n[Step 1/6] Creating Synthetic RGB Test Image...")
    with tempfile.TemporaryDirectory() as tmpdir:
        synth_img_path = Path(tmpdir) / "synth_forest_tile.jpg"
        img = Image.new("RGB", (128, 128), color=(34, 139, 34))  # Forest green
        img.save(synth_img_path)
        print(f"  [PASS] Created temporary test image at: {synth_img_path}")

        # 2. Test Image Validation & Preprocessing
        print("\n[Step 2/6] Verifying Image Validation & Preprocessing Pipeline...")
        tensor, orig_dim = load_and_preprocess_image(synth_img_path)
        assert tensor.shape == (1, 3, 64, 64), f"Expected tensor shape (1, 3, 64, 64), got {tensor.shape}"
        assert orig_dim == (128, 128), f"Expected orig_dim (128, 128), got {orig_dim}"
        print(f"  [PASS] Input tensor shape: {tuple(tensor.shape)}, Original size: {orig_dim}")

        # 3. Test Custom CNN Checkpoint Loading & Prediction
        print("\n[Step 3/6] Testing Custom CNN Model Loading & Prediction...")
        cnn_ckpt = MODELS_DIR / "custom_cnn_best.pt"
        if not cnn_ckpt.exists():
            print(f"  [FAIL] Custom CNN checkpoint not found at: {cnn_ckpt}")
            return False

        cnn_model = load_trained_model("custom_cnn", checkpoint_path=cnn_ckpt, device=device)
        cnn_results, _ = predict_image(synth_img_path, model=cnn_model, device=device, top_k=3)

        assert cnn_results["predicted_class"] in EUROSAT_CLASSES, f"Invalid class: {cnn_results['predicted_class']}"
        assert 0.0 <= cnn_results["confidence"] <= 1.0, f"Invalid confidence: {cnn_results['confidence']}"
        assert len(cnn_results["top3"]) == 3, f"Expected 3 top predictions, got {len(cnn_results['top3'])}"
        all_sum = sum(cnn_results["all_probabilities"].values())
        assert abs(all_sum - 1.0) < 0.01, f"Probabilities do not sum to 1: {all_sum}"

        print(f"  [PASS] Custom CNN Prediction : {cnn_results['predicted_class']}")
        print(f"         Confidence            : {cnn_results['confidence_percentage']}")
        print(f"         Top-3 Classes         : {[item['class'] for item in cnn_results['top3']]}")

        # 4. Test ResNet-18 Checkpoint Loading & Prediction
        print("\n[Step 4/6] Testing ResNet-18 Model Loading & Prediction...")
        resnet_ckpt = MODELS_DIR / "resnet18_best.pt"
        if not resnet_ckpt.exists():
            print(f"  [FAIL] ResNet-18 checkpoint not found at: {resnet_ckpt}")
            return False

        resnet_model = load_trained_model("resnet18", checkpoint_path=resnet_ckpt, device=device)
        resnet_results, _ = predict_image(synth_img_path, model=resnet_model, device=device, top_k=3)

        assert resnet_results["predicted_class"] in EUROSAT_CLASSES, f"Invalid class: {resnet_results['predicted_class']}"
        assert 0.0 <= resnet_results["confidence"] <= 1.0, f"Invalid confidence: {resnet_results['confidence']}"
        assert len(resnet_results["top3"]) == 3, f"Expected 3 top predictions, got {len(resnet_results['top3'])}"
        all_sum = sum(resnet_results["all_probabilities"].values())
        assert abs(all_sum - 1.0) < 0.01, f"Probabilities do not sum to 1: {all_sum}"

        print(f"  [PASS] ResNet-18 Prediction  : {resnet_results['predicted_class']}")
        print(f"         Confidence            : {resnet_results['confidence_percentage']}")
        print(f"         Top-3 Classes         : {[item['class'] for item in resnet_results['top3']]}")

        # 5. Test Real EuroSAT Tile (if available)
        print("\n[Step 5/6] Testing Real EuroSAT Tile Prediction...")
        real_sample = Path("data/raw/EuroSAT_RGB/Forest/Forest_1.jpg")
        if real_sample.exists():
            res_real, _ = predict_image(real_sample, model=resnet_model, device=device, top_k=3)
            print(f"  [PASS] Real Image: {real_sample.name}")
            print(f"         ResNet-18 Prediction: {res_real['predicted_class']} ({res_real['confidence_percentage']})")
        else:
            print("  [SKIP] Real sample Forest_1.jpg not in expected location.")

        # 6. Verify Model Comparison Logic
        print("\n[Step 6/6] Verifying Side-by-Side Model Comparison Agreement...")
        agreement = cnn_results["predicted_class"] == resnet_results["predicted_class"]
        print(f"  [PASS] Both-model comparison completed. Agreement: {'YES' if agreement else 'NO'}")

    print("\n" + "=" * 70)
    print("ALL MILESTONE 4.5 INFERENCE SMOKE TESTS PASSED!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
