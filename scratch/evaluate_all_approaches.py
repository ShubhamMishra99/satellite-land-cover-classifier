import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.models as tv_models
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

from src.constants import RAW_DATA_DIR
from src.inference.predictor import load_trained_model

# 1. Load splits
with open('data/processed/splits.json', 'r') as f:
    splits = json.load(f)

train_items = splits['train']
val_items = splits['val']

# Sample 1000 train for reference/fitting, 1000 val for validation evaluation
np.random.seed(42)
sampled_train = [train_items[i] for i in np.random.choice(len(train_items), size=1000, replace=False)]
sampled_val = [val_items[i] for i in np.random.choice(len(val_items), size=1000, replace=False)]

ood_files = sorted(Path("scratch/ood_eval_set").glob("*.jpg"))
print(f"Loaded {len(sampled_train)} train, {len(sampled_val)} val, {len(ood_files)} OOD files.")

# Load models
eurosat_model = load_trained_model('resnet18')
eurosat_backbone = torch.nn.Sequential(*(list(eurosat_model.backbone.children())[:-1]))
eurosat_backbone.eval()

imagenet_resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
imagenet_backbone = torch.nn.Sequential(*(list(imagenet_resnet.children())[:-1]))
imagenet_backbone.eval()

norm_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_image_tensor(p):
    im = Image.open(p).convert('RGB')
    return norm_transform(im).unsqueeze(0)

def compute_image_stats(p):
    im = Image.open(p).convert('RGB').resize((64, 64))
    arr = np.array(im, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    
    # Variance per channel and total
    var_total = float(np.var(arr))
    var_r = float(np.var(r))
    var_g = float(np.var(g))
    var_b = float(np.var(b))
    
    # Gray Laplacian variance (sharpness / edge density)
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    lap_var = float(np.var(lap))
    
    # Color channel mean & std
    mean_rgb = arr.mean(axis=(0, 1))
    
    # Entropy (8-bit grayscale)
    gray_uint8 = (gray * 255).astype(np.uint8)
    hist, _ = np.histogram(gray_uint8, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))
    
    return {
        'var_total': var_total,
        'lap_var': lap_var,
        'entropy': entropy,
        'mean_rgb': mean_rgb,
    }

# Extract features
print("Extracting features on train (1000)...")
train_img_embs = []
train_stats = []
with torch.no_grad():
    for item in sampled_train:
        p = RAW_DATA_DIR / 'EuroSAT_RGB' / item['path']
        t = get_image_tensor(p)
        feat = imagenet_backbone(t).squeeze().cpu().numpy()
        train_img_embs.append(feat)
        train_stats.append(compute_image_stats(p))

train_img_embs = np.array(train_img_embs)
# Normalize
train_img_norm = train_img_embs / (np.linalg.norm(train_img_embs, axis=1, keepdims=True) + 1e-8)

print("Extracting features on val (1000)...")
val_img_embs = []
val_stats = []
with torch.no_grad():
    for item in sampled_val:
        p = RAW_DATA_DIR / 'EuroSAT_RGB' / item['path']
        t = get_image_tensor(p)
        feat = imagenet_backbone(t).squeeze().cpu().numpy()
        val_img_embs.append(feat)
        val_stats.append(compute_image_stats(p))

val_img_embs = np.array(val_img_embs)
val_img_norm = val_img_embs / (np.linalg.norm(val_img_embs, axis=1, keepdims=True) + 1e-8)

print("Extracting features on OOD suite (15)...")
ood_img_embs = []
ood_stats = []
with torch.no_grad():
    for p in ood_files:
        t = get_image_tensor(p)
        feat = imagenet_backbone(t).squeeze().cpu().numpy()
        ood_img_embs.append(feat)
        ood_stats.append(compute_image_stats(p))

ood_img_embs = np.array(ood_img_embs)
ood_img_norm = ood_img_embs / (np.linalg.norm(ood_img_embs, axis=1, keepdims=True) + 1e-8)

print("\n--- 1. Statistical Check Bounds (from Train Sample) ---")
train_vars = [s['var_total'] for s in train_stats]
train_entropies = [s['entropy'] for s in train_stats]
train_lap_vars = [s['lap_var'] for s in train_stats]

print(f"Train Variance:   min={np.min(train_vars):.5f}, 1st%={np.percentile(train_vars, 1):.5f}, mean={np.mean(train_vars):.5f}")
print(f"Train Entropy:    min={np.min(train_entropies):.3f}, 1st%={np.percentile(train_entropies, 1):.3f}, max={np.max(train_entropies):.3f}")
print(f"Train Lap Var:    min={np.min(train_lap_vars):.6f}, 1st%={np.percentile(train_lap_vars, 1):.6f}")

print("\nOOD Stats:")
for p, s in zip(ood_files, ood_stats):
    print(f"  {p.name:22s} | var={s['var_total']:.5f} | entropy={s['entropy']:.2f} | lap_var={s['lap_var']:.6f}")

print("\n--- 2. Prototype / Centroid Distance in Feature Space ---")
# Build class centroids or multi-prototypes
# Let's compute k-means or nearest neighbors
nn_model = NearestNeighbors(n_neighbors=5, metric='cosine').fit(train_img_norm)
val_dist, _ = nn_model.kneighbors(val_img_norm)
val_scores = val_dist.mean(axis=1)

ood_dist, _ = nn_model.kneighbors(ood_img_norm)
ood_scores = ood_dist.mean(axis=1)

print(f"Val k-NN Cosine Distance: min={np.min(val_scores):.4f}, mean={np.mean(val_scores):.4f}, 95th%={np.percentile(val_scores, 95):.4f}, 99th%={np.percentile(val_scores, 99):.4f}, max={np.max(val_scores):.4f}")

print("\nOOD k-NN Cosine Distances:")
for p, sc in zip(ood_files, ood_scores):
    print(f"  {p.name:22s} : {sc:.4f}")

# Threshold analysis:
# If we set threshold at 99th percentile of train/val (e.g. 0.38 or 0.40):
for th in [0.35, 0.38, 0.40, 0.42, 0.44]:
    frr = np.mean(val_scores > th)
    far = np.mean(ood_scores <= th)
    print(f"Threshold {th:.2f} -> False Rejection Rate (val): {frr*100:.2f}%, False Acceptance Rate (ood): {far*100:.2f}%")

# Also test Nearest Centroid to 10 class centroids:
print("\n--- 3. Class-Prototype Centroids ---")
class_prototypes = []
for c_idx in range(10):
    c_items = [it for it in sampled_train if it['label'] == c_idx]
    c_indices = [i for i, it in enumerate(sampled_train) if it['label'] == c_idx]
    c_proto = train_img_norm[c_indices].mean(axis=0)
    c_proto /= np.linalg.norm(c_proto)
    class_prototypes.append(c_proto)
class_prototypes = np.array(class_prototypes) # (10, 512)

# Distance to nearest class prototype = 1 - max(cosine_sim)
val_proto_sim = val_img_norm @ class_prototypes.T # (1000, 10)
val_min_proto_dist = 1.0 - val_proto_sim.max(axis=1)

ood_proto_sim = ood_img_norm @ class_prototypes.T
ood_min_proto_dist = 1.0 - ood_proto_sim.max(axis=1)

print(f"Val Min Prototype Distance: min={np.min(val_min_proto_dist):.4f}, mean={np.mean(val_min_proto_dist):.4f}, 95th%={np.percentile(val_min_proto_dist, 95):.4f}, 99th%={np.percentile(val_min_proto_dist, 99):.4f}, max={np.max(val_min_proto_dist):.4f}")

print("\nOOD Min Prototype Distances:")
for p, sc in zip(ood_files, ood_min_proto_dist):
    print(f"  {p.name:22s} : {sc:.4f}")

for th in [0.30, 0.32, 0.35, 0.37, 0.40]:
    frr = np.mean(val_min_proto_dist > th)
    far = np.mean(ood_min_proto_dist <= th)
    print(f"Prototype Thresh {th:.2f} -> FRR (val): {frr*100:.2f}%, FAR (ood): {far*100:.2f}%")
