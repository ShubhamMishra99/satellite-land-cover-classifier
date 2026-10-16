import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.models as tv_models
from torchvision import transforms
from sklearn.cluster import KMeans

from src.constants import RAW_DATA_DIR

with open('data/processed/splits.json', 'r') as f:
    splits = json.load(f)

train_items = splits['train']
val_items = splits['val']

print(f"Total train: {len(train_items)}, val: {len(val_items)}")

# Pretrained ImageNet ResNet-18 backbone
imagenet_resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
backbone = torch.nn.Sequential(*(list(imagenet_resnet.children())[:-1]))
backbone.eval()

norm_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Let's extract on all train items or a stratified 3,000 train items to compute prototypes
np.random.seed(42)
sampled_train = [train_items[i] for i in np.random.choice(len(train_items), size=3000, replace=False)]

print("Extracting 3000 train embeddings for prototypes...")
train_embs = []
with torch.no_grad():
    for idx, item in enumerate(sampled_train):
        p = RAW_DATA_DIR / 'EuroSAT_RGB' / item['path']
        im = Image.open(p).convert('RGB')
        t = norm_transform(im).unsqueeze(0)
        feat = backbone(t).squeeze().cpu().numpy()
        train_embs.append(feat)

train_embs = np.array(train_embs)
train_norm = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8)

# Compute 10 class prototypes
class_prototypes = []
for c_idx in range(10):
    c_indices = [i for i, it in enumerate(sampled_train) if it['label'] == c_idx]
    c_proto = train_norm[c_indices].mean(axis=0)
    c_proto /= np.linalg.norm(c_proto)
    class_prototypes.append(c_proto)
class_prototypes = np.array(class_prototypes) # (10, 512)

# Also compute multi-prototypes (e.g. 30 clusters via KMeans to capture intra-class diversity)
kmeans = KMeans(n_clusters=30, random_state=42, n_init=10).fit(train_norm)
cluster_prototypes = kmeans.cluster_centers_
cluster_prototypes /= np.linalg.norm(cluster_prototypes, axis=1, keepdims=True) # (30, 512)

# Load OOD suite
ood_files = sorted(Path("scratch/ood_eval_set").glob("*.jpg"))
ood_embs = []
with torch.no_grad():
    for p in ood_files:
        im = Image.open(p).convert('RGB')
        t = norm_transform(im).unsqueeze(0)
        ood_embs.append(backbone(t).squeeze().cpu().numpy())
ood_embs = np.array(ood_embs)
ood_norm = ood_embs / (np.linalg.norm(ood_embs, axis=1, keepdims=True) + 1e-8)

# Evaluate on full validation set (4050 samples)
print("Evaluating on all 4,050 validation samples...")
val_dists_class = []
val_dists_cluster = []

batch_size = 64
for start_idx in range(0, len(val_items), batch_size):
    batch_items = val_items[start_idx:start_idx+batch_size]
    batch_tensors = []
    for it in batch_items:
        p = RAW_DATA_DIR / 'EuroSAT_RGB' / it['path']
        im = Image.open(p).convert('RGB')
        batch_tensors.append(norm_transform(im))
    batch_tensor = torch.stack(batch_tensors)
    with torch.no_grad():
        feats = backbone(batch_tensor).squeeze().cpu().numpy()
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    f_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    
    # Class proto dist: 1 - max(cos_sim)
    d_class = 1.0 - (f_norm @ class_prototypes.T).max(axis=1)
    val_dists_class.extend(d_class.tolist())
    
    # Cluster proto dist
    d_cluster = 1.0 - (f_norm @ cluster_prototypes.T).max(axis=1)
    val_dists_cluster.extend(d_cluster.tolist())

val_dists_class = np.array(val_dists_class)
val_dists_cluster = np.array(val_dists_cluster)

ood_dists_class = 1.0 - (ood_norm @ class_prototypes.T).max(axis=1)
ood_dists_cluster = 1.0 - (ood_norm @ cluster_prototypes.T).max(axis=1)

print("\n--- 10-Class Prototype Performance across 4,050 Val Samples ---")
print(f"Val min={val_dists_class.min():.4f}, mean={val_dists_class.mean():.4f}, 90th%={np.percentile(val_dists_class, 90):.4f}, 95th%={np.percentile(val_dists_class, 95):.4f}, 98th%={np.percentile(val_dists_class, 98):.4f}, 99th%={np.percentile(val_dists_class, 99):.4f}, max={val_dists_class.max():.4f}")

print("\nOOD 10-Class Distances:")
for p, d in zip(ood_files, ood_dists_class):
    print(f"  {p.name:24s}: {d:.4f}")

for th in [0.32, 0.35, 0.37, 0.38, 0.40, 0.42]:
    frr = np.mean(val_dists_class > th) * 100
    far = np.mean(ood_dists_class <= th) * 100
    print(f"Thresh {th:.2f} -> False Rejection Rate (Val): {frr:5.2f}%, False Acceptance Rate (OOD): {far:5.2f}%")

print("\n--- 30-Cluster Multi-Prototype Performance across 4,050 Val Samples ---")
print(f"Val min={val_dists_cluster.min():.4f}, mean={val_dists_cluster.mean():.4f}, 95th%={np.percentile(val_dists_cluster, 95):.4f}, 98th%={np.percentile(val_dists_cluster, 98):.4f}, 99th%={np.percentile(val_dists_cluster, 99):.4f}, max={val_dists_cluster.max():.4f}")

print("\nOOD 30-Cluster Distances:")
for p, d in zip(ood_files, ood_dists_cluster):
    print(f"  {p.name:24s}: {d:.4f}")

for th in [0.30, 0.32, 0.34, 0.36, 0.38, 0.40]:
    frr = np.mean(val_dists_cluster > th) * 100
    far = np.mean(ood_dists_cluster <= th) * 100
    print(f"Cluster Thresh {th:.2f} -> False Rejection Rate (Val): {frr:5.2f}%, False Acceptance Rate (OOD): {far:5.2f}%")
