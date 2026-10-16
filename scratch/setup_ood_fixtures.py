"""Setup OOD test fixtures and evaluate raw model predictions without threshold."""

import shutil
from pathlib import Path
from PIL import Image
import torch

from src.inference.predictor import load_trained_model, predict_image

OOD_DIR = Path("tests/fixtures/ood")
OOD_DIR.mkdir(parents=True, exist_ok=True)

# Copy the generated photos into test fixtures
brain_dir = Path(r"C:\Users\DELL\.gemini\antigravity-ide\brain\599169d4-55c0-44dc-9b11-1ddb9e9b19f9")

source_files = {
    "dog_golden_retriever.jpg": list(brain_dir.glob("golden_retriever_dog_*.jpg"))[0],
    "human_portrait.jpg": list(brain_dir.glob("human_portrait_*.jpg"))[0],
    "indoor_living_room.jpg": list(brain_dir.glob("indoor_room_*.jpg"))[0],
    "food_pasta_plate.jpg": list(brain_dir.glob("food_plate_*.jpg"))[0],
}

for dest_name, src_path in source_files.items():
    dest_path = OOD_DIR / dest_name
    shutil.copy(src_path, dest_path)
    print(f"Copied {src_path.name} -> {dest_path}")

# Add synthetic noise and uniform blank images
noise_img = Image.fromarray(torch.randint(0, 256, (64, 64, 3), dtype=torch.uint8).numpy())
noise_img.save(OOD_DIR / "random_uniform_noise.jpg")

blank_white = Image.new("RGB", (64, 64), (255, 255, 255))
blank_white.save(OOD_DIR / "blank_white.jpg")

print(f"Created additional synthetic OOD fixtures in {OOD_DIR}")
