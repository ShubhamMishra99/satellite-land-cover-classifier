import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.models as tv_models
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

from src.constants import EUROSAT_CLASSES, RAW_DATA_DIR
from src.inference.predictor import load_trained_model, load_and_preprocess_image

# 1. Load splits
with open('data/processed/splits.json', 'r') as f:
    splits = json.load(f)

# Use train for fitting/modeling, val for evaluation!
train_items = splits['train']
val_items = splits['val']

print(f"Total train items: {len(train_items)}, val items: {len(val_items)}")

# Let's sample a subset of train (e.g. 1000) for fast fitting and val (e.g. 500) for evaluation
np.random.seed(42)
sampled_train_indices = np.random.choice(len(train_items), size=1000, replace=False)
sampled_val_indices = np.random.choice(len(val_items), size=500, replace=False)

train_sample = [train_items[i] for i in sampled_train_indices]
val_sample = [val_items[i] for i in sampled_val_indices]

# Load fixtures
ood_fixtures = sorted(Path('tests/fixtures/ood').glob('*.*'))
print(f"OOD fixtures: {[f.name for f in ood_fixtures]}")

# Let's test Feature Extractors:
# Extractor A: EuroSAT ResNet-18 (trained checkpoint) embedding
eurosat_resnet = load_trained_model('resnet18')
# Remove final linear fc to get 512-dim embedding
eurosat_backbone = torch.nn.Sequential(*(list(eurosat_resnet.backbone.children())[:-1]))
eurosat_backbone.eval()

# Extractor B: Pretrained ImageNet ResNet-18 embedding
imagenet_resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
imagenet_backbone = torch.nn.Sequential(*(list(imagenet_resnet.children())[:-1]))
imagenet_backbone.eval()

# Standard transform
eval_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(img_path, backbone):
    pil_img = Image.open(img_path).convert('RGB')
    tensor = eval_transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        feat = backbone(tensor).squeeze().cpu().numpy()
    return feat

def extract_color_stats(img_path):
    pil_img = Image.open(img_path).convert('RGB').resize((64, 64))
    arr = np.array(pil_img, dtype=np.float32) / 255.0  # (64, 64, 3)
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    # Channel correlations
    r, g, b = arr[:, :, 0].flatten(), arr[:, :, 1].flatten(), arr[:, :, 2].flatten()
    rg_corr = np.corrcoef(r, g)[0, 1] if stds[0] > 1e-4 and stds[1] > 1e-4 else 0.0
    rb_corr = np.corrcoef(r, b)[0, 1] if stds[0] > 1e-4 and stds[2] > 1e-4 else 0.0
    gb_corr = np.corrcoef(g, b)[0, 1] if stds[1] > 1e-4 and stds[2] > 1e-4 else 0.0
    # Laplacian variance (edge sharpness)
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray_img = gray.reshape((64, 64))
    # Simple discrete laplacian
    lap = -4 * gray_img[1:-1, 1:-1] + gray_img[:-2, 1:-1] + gray_img[2:, 1:-1] + gray_img[1:-1, :-2] + gray_img[1:-1, 2:]
    lap_var = float(np.var(lap))
    return np.concatenate([means, stds, [rg_corr, rb_corr, gb_corr, lap_var]])

print("Extracting features on train sample (1000)...")
train_feats_eurosat = []
train_feats_imagenet = []
train_color_stats = []

for item in train_sample:
    p = RAW_DATA_DIR / 'EuroSAT_RGB' / item['path']
    train_feats_eurosat.append(extract_features(p, eurosat_backbone))
    train_feats_imagenet.append(extract_features(p, imagenet_backbone))
    train_color_stats.append(extract_color_stats(p))

train_feats_eurosat = np.array(train_feats_eurosat)
train_feats_imagenet = np.array(train_feats_imagenet)
train_color_stats = np.array(train_color_stats)

print("Extracting features on val sample (500)...")
val_feats_eurosat = []
val_feats_imagenet = []
val_color_stats = []
for item in val_sample:
    p = RAW_DATA_DIR / 'EuroSAT_RGB' / item['path']
    val_feats_eurosat.append(extract_features(p, eurosat_backbone))
    val_feats_imagenet.append(extract_features(p, imagenet_backbone))
    val_color_stats.append(extract_color_stats(p))

val_feats_eurosat = np.array(val_feats_eurosat)
val_feats_imagenet = np.array(val_feats_imagenet)
val_color_stats = np.array(val_color_stats)

print("Extracting features on OOD fixtures...")
ood_feats_eurosat = []
ood_feats_imagenet = []
ood_color_stats = []
for p in ood_fixtures:
    ood_feats_eurosat.append(extract_features(p, eurosat_backbone))
    ood_feats_imagenet.append(extract_features(p, imagenet_backbone))
    ood_color_stats.append(extract_color_stats(p))

ood_feats_eurosat = np.array(ood_feats_eurosat)
ood_feats_imagenet = np.array(ood_feats_imagenet)
ood_color_stats = np.array(ood_color_stats)

print("\n--- Testing Approaches ---")

# Method 1: Centroid / Cosine distance in EuroSAT embedding space
# Normalize embeddings
norm_train_e = train_feats_eurosat / (np.linalg.norm(train_feats_eurosat, axis=1, keepdims=True) + 1e-8)
norm_val_e = val_feats_eurosat / (np.linalg.norm(val_feats_eurosat, axis=1, keepdims=True) + 1e-8)
norm_ood_e = ood_feats_eurosat / (np.linalg.norm(ood_feats_eurosat, axis=1, keepdims=True) + 1e-8)

centroid_e = norm_train_e.mean(axis=0)
centroid_e /= np.linalg.norm(centroid_e)

val_cos_e = norm_val_e @ centroid_e
ood_cos_e = norm_ood_e @ centroid_e

print("EuroSAT Embedding Cosine Similarity to centroid:")
print(f"Val min: {val_cos_e.min():.4f}, mean: {val_cos_e.mean():.4f}, max: {val_cos_e.max():.4f}")
for f, sc in zip(ood_fixtures, ood_cos_e):
    print(f"  {f.name:25s}: {sc:.4f}")

# Method 2: k-NN distance in EuroSAT embedding space
nn_e = NearestNeighbors(n_neighbors=5, metric='cosine').fit(norm_train_e)
val_knn_dist_e, _ = nn_e.kneighbors(norm_val_e)
val_knn_score_e = val_knn_dist_e.mean(axis=1)
ood_knn_dist_e, _ = nn_e.kneighbors(norm_ood_e)
ood_knn_score_e = ood_knn_dist_e.mean(axis=1)
print("\nEuroSAT Embedding k-NN Cosine Distance (lower is in-distribution):")
print(f"Val min: {val_knn_score_e.min():.4f}, mean: {val_knn_score_e.mean():.4f}, max: {val_knn_score_e.max():.4f}")
for f, sc in zip(ood_fixtures, ood_knn_score_e):
    print(f"  {f.name:25s}: {sc:.4f}")

# Method 3: k-NN distance in ImageNet feature space
norm_train_img = train_feats_imagenet / (np.linalg.norm(train_feats_imagenet, axis=1, keepdims=True) + 1e-8)
norm_val_img = val_feats_imagenet / (np.linalg.norm(val_feats_imagenet, axis=1, keepdims=True) + 1e-8)
norm_ood_img = ood_feats_imagenet / (np.linalg.norm(ood_feats_imagenet, axis=1, keepdims=True) + 1e-8)

nn_img = NearestNeighbors(n_neighbors=5, metric='cosine').fit(norm_train_img)
val_knn_dist_img, _ = nn_img.kneighbors(norm_val_img)
val_knn_score_img = val_knn_dist_img.mean(axis=1)
ood_knn_dist_img, _ = nn_img.kneighbors(norm_ood_img)
ood_knn_score_img = ood_knn_dist_img.mean(axis=1)
print("\nImageNet Embedding k-NN Cosine Distance (lower is in-distribution):")
print(f"Val min: {val_knn_score_img.min():.4f}, mean: {val_knn_score_img.mean():.4f}, 99th p: {np.percentile(val_knn_score_img, 99):.4f}, max: {val_knn_score_img.max():.4f}")
for f, sc in zip(ood_fixtures, ood_knn_score_img):
    print(f"  {f.name:25s}: {sc:.4f}")

# Method 4: Centroid / Cosine distance in ImageNet feature space
centroid_img = norm_train_img.mean(axis=0)
centroid_img /= np.linalg.norm(centroid_img)

val_cos_img = norm_val_img @ centroid_img
ood_cos_img = norm_ood_img @ centroid_img

print("\nImageNet Embedding Cosine Similarity to centroid (higher is in-distribution):")
print(f"Val min: {val_cos_img.min():.4f}, 1st p: {np.percentile(val_cos_img, 1):.4f}, mean: {val_cos_img.mean():.4f}, max: {val_cos_img.max():.4f}")
for f, sc in zip(ood_fixtures, ood_cos_img):
    print(f"  {f.name:25s}: {sc:.4f}")
