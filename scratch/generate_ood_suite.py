import json
import math
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.models as tv_models
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EmpiricalCovariance

# Create additional synthetic and naturalistic unsupported images for rigorous evaluation
ood_dir = Path("scratch/ood_eval_set")
ood_dir.mkdir(parents=True, exist_ok=True)

# 1. Blank black
im_black = Image.new("RGB", (256, 256), (0, 0, 0))
im_black.save(ood_dir / "solid_black.jpg")

# 2. Blank grey
im_grey = Image.new("RGB", (256, 256), (128, 128, 128))
im_grey.save(ood_dir / "solid_grey.jpg")

# 3. Solid vibrant red
im_red = Image.new("RGB", (256, 256), (255, 0, 0))
im_red.save(ood_dir / "solid_red.jpg")

# 4. Solid vibrant blue
im_blue = Image.new("RGB", (256, 256), (0, 0, 255))
im_blue.save(ood_dir / "solid_blue.jpg")

# 5. Gaussian noise
np.random.seed(42)
gauss_noise = np.random.normal(128, 40, (128, 128, 3)).clip(0, 255).astype(np.uint8)
Image.fromarray(gauss_noise).save(ood_dir / "gaussian_noise.jpg")

# 6. Color horizontal stripes / gradient
gradient = np.zeros((128, 128, 3), dtype=np.uint8)
for i in range(128):
    gradient[i, :, 0] = int(255 * (i / 128.0))
    gradient[:, i, 1] = int(255 * (1.0 - i / 128.0))
    gradient[i, :, 2] = int(128 * math.sin(i / 10.0) + 128)
Image.fromarray(gradient).save(ood_dir / "color_gradient.jpg")

# 7. Text document / code page
im_text = Image.new("RGB", (256, 256), (255, 255, 255))
draw = ImageDraw.Draw(im_text)
for row, line in enumerate(["Lorem ipsum dolor sit amet,", "consectetur adipiscing elit.", "Def function predict(image):", "    return result"]):
    draw.text((10, 20 + row * 25), line, fill=(0, 0, 0))
im_text.save(ood_dir / "text_document.jpg")

# 8. Geometric diagram / shapes
im_shapes = Image.new("RGB", (256, 256), (240, 240, 240))
draw = ImageDraw.Draw(im_shapes)
draw.rectangle([30, 30, 100, 100], fill=(220, 50, 50), outline=(0, 0, 0))
draw.ellipse([120, 120, 220, 220], fill=(50, 180, 50), outline=(0, 0, 0))
draw.polygon([(150, 20), (230, 80), (120, 90)], fill=(50, 80, 220))
im_shapes.save(ood_dir / "geometric_shapes.jpg")

# 9. Checkerboard pattern
cb = np.zeros((64, 64, 3), dtype=np.uint8)
for i in range(64):
    for j in range(64):
        if (i // 8 + j // 8) % 2 == 0:
            cb[i, j] = [255, 255, 255]
Image.fromarray(cb).save(ood_dir / "checkerboard.jpg")

# 10. Copy fixtures into evaluation set as well
for f in Path("tests/fixtures/ood").glob("*.*"):
    im = Image.open(f)
    im.save(ood_dir / f.name)

print(f"Created/collected {len(list(ood_dir.glob('*.jpg')))} OOD test images in {ood_dir}")
