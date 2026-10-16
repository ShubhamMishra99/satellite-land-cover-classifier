"""Compute and save EuroSAT class and cluster prototypes from training split."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.models as tv_models
from torchvision import transforms
from sklearn.cluster import KMeans

from src.constants import MODELS_DIR, RAW_DATA_DIR

def build_and_save_prototypes():
    with open('data/processed/splits.json', 'r') as f:
        splits = json.load(f)

    train_items = splits['train']
    print(f"Loading {len(train_items)} EuroSAT training images...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imagenet_resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
    backbone = torch.nn.Sequential(*(list(imagenet_resnet.children())[:-1])).to(device)
    backbone.eval()

    norm_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Extract embeddings in batches for all training samples
    batch_size = 128
    all_embeddings = []
    all_labels = []

    print("Extracting training feature embeddings...")
    for start_idx in range(0, len(train_items), batch_size):
        batch = train_items[start_idx : start_idx + batch_size]
        tensors = []
        for it in batch:
            p = RAW_DATA_DIR / 'EuroSAT_RGB' / it['path']
            im = Image.open(p).convert('RGB')
            tensors.append(norm_transform(im))
            all_labels.append(it['label'])
        batch_t = torch.stack(tensors).to(device)
        with torch.no_grad():
            f = backbone(batch_t).squeeze().cpu().numpy()
        if f.ndim == 1:
            f = f.reshape(1, -1)
        all_embeddings.append(f)

    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)
    print(f"Extracted {all_embeddings.shape[0]} embeddings of dimension {all_embeddings.shape[1]}")

    # L2 normalize
    norm_embeddings = all_embeddings / (np.linalg.norm(all_embeddings, axis=1, keepdims=True) + 1e-8)

    # 1. Compute 10 canonical class centroids
    class_prototypes = np.zeros((10, 512), dtype=np.float32)
    for c_idx in range(10):
        c_embs = norm_embeddings[all_labels == c_idx]
        proto = c_embs.mean(axis=0)
        proto /= (np.linalg.norm(proto) + 1e-8)
        class_prototypes[c_idx] = proto

    # 2. Compute 30 sub-cluster centroids via KMeans (capturing intra-class terrain variations)
    print("Computing 30 cluster prototypes via KMeans...")
    kmeans = KMeans(n_clusters=30, random_state=42, n_init=10).fit(norm_embeddings)
    cluster_prototypes = kmeans.cluster_centers_.astype(np.float32)
    cluster_prototypes /= (np.linalg.norm(cluster_prototypes, axis=1, keepdims=True) + 1e-8)

    # Save to models/suitability_prototypes.pt
    save_path = MODELS_DIR / "suitability_prototypes.pt"
    prototype_data = {
        "class_prototypes": torch.from_numpy(class_prototypes),
        "cluster_prototypes": torch.from_numpy(cluster_prototypes),
        "embedding_dim": 512,
        "num_classes": 10,
        "num_clusters": 30,
        "backbone": "resnet18_imagenet1k_v1",
        "threshold_class": 0.38,
        "threshold_cluster": 0.36,
        "metadata": {
            "source_split": "train (18,900 images)",
            "classes": [
                "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
            ]
        }
    }
    torch.save(prototype_data, save_path)
    print(f"Saved suitability prototypes to {save_path} ({save_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    build_and_save_prototypes()
