"""Evaluate raw predictions of ResNet-18 and Custom CNN on real non-satellite (OOD) images."""

from pathlib import Path
from PIL import Image
import torch

from src.inference.predictor import load_trained_model, predict_image

OOD_DIR = Path("tests/fixtures/ood")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet = load_trained_model("resnet18", device=device)
custom_cnn = load_trained_model("custom_cnn", device=device)

images = list(OOD_DIR.glob("*.jpg"))

print(f"{'Image':<30} | {'Model':<12} | {'Predicted Class':<20} | {'Confidence':<12}")
print("-" * 80)

for img_path in sorted(images):
    # ResNet-18
    res_r, _ = predict_image(img_path, resnet, device=device)
    print(f"{img_path.name:<30} | {'resnet18':<12} | {res_r['predicted_class']:<20} | {res_r['confidence']:<12.4f}")

    # Custom CNN
    res_c, _ = predict_image(img_path, custom_cnn, device=device)
    print(f"{img_path.name:<30} | {'custom_cnn':<12} | {res_c['predicted_class']:<20} | {res_c['confidence']:<12.4f}")
    print("-" * 80)
