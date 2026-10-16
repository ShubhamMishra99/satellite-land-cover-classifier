# System Architecture & Technical Specifications

## 1. High-Level System Architecture

```text
+-----------------------------------------------------------------------------------+
|                                 USER INTERFACE                                    |
|              (React 18 + TypeScript + Vite + Tailwind CSS + Lucide)               |
+------------------------------------+----------------------------------------------+
                                     |
                                REST API (HTTP/JSON + Multipart)
                                     |
+------------------------------------v----------------------------------------------+
|                              FASTAPI BACKEND                                      |
|                                                                                   |
|  +----------------+  +-------------------+  +------------------+  +-------------+ |
|  | /health        |  | /predict          |  | /predict/explain |  | /predict/   | |
|  | (Status/Model) |  | (Single Tile)     |  | (Grad-CAM Map)   |  |  batch      | |
|  +----------------+  +-------------------+  +------------------+  +-------------+ |
|                                    |                                              |
|                    +---------------v---------------+                              |
|                    |     Pydantic Validation Layer |                              |
|                    | (MIME, Size, Integrity, Shape)|                              |
|                    +---------------+---------------+                              |
|                                    |                                              |
|                    +---------------v---------------+                              |
|                    |      Inference Service        |                              |
|                    |   (Singleton Model Runner)    |                              |
|                    +---------------+---------------+                              |
+------------------------------------+----------------------------------------------+
                                     |
+------------------------------------v----------------------------------------------+
|                            MACHINE LEARNING LAYER                                 |
|                                                                                   |
|  +---------------------------+             +------------------------------------+ |
|  |     Model Zoo (PyTorch)   |             |       Explainability Engine        | |
|  | - Custom CNN Baseline     |             | - Grad-CAM Target Layer Gradients  | |
|  | - ResNet-18 Fine-Tuned    |             | - Colormap Overlay Generator       | |
|  +---------------------------+             +------------------------------------+ |
|               ^                                                                   |
|  +------------+-----------------------------------------------------------------+ |
|  | Preprocessing Pipeline (torchvision.transforms)                              | |
|  | - Eval / Inference: Resize(64,64) -> ToTensor() -> Normalize(mean, std)      | |
|  | - Training Augment: RandomHorizontalFlip, RandomVerticalFlip, Rotations, Jitter| |
|  +------------------------------------------------------------------------------+ |
+------------------------------------+----------------------------------------------+
                                     |
+------------------------------------v----------------------------------------------+
|                              DATA & STORAGE LAYER                                 |
|  - Raw: EuroSAT RGB (27,000 Sentinel-2 64x64 tiles, 10 classes)                   |
|  - Splits: Stratified Train (70%) / Val (15%) / Test (15%) with fixed seed 42     |
|  - Checkpoints: Versioned PyTorch .pt state_dicts + metadata JSON                 |
+-----------------------------------------------------------------------------------+
```

---

## 2. Core Machine Learning Design Rationale

### 2.1 Why EuroSAT RGB?
- 27,000 geo-referenced image patches covering 13 European countries acquired by ESA's Sentinel-2 satellite.
- 10 distinct Land-Use / Land-Cover (LULC) classes (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake).
- $64 \times 64$ pixels per tile provides an ideal spatial footprint for evaluating satellite land classification across lightweight and deep convolutional models.

### 2.2 Baseline Model: Custom CNN Architecture (Milestone 3)
- **Design Philosophy**: Establish an empirical baseline performance trained strictly from scratch without external knowledge transfer.
- **Topology**:
  1. $3 \times 64 \times 64 \xrightarrow{\text{Conv3x3(32) + BN + ReLU + MaxPool2x2}} 32 \times 32 \times 32$
  2. $32 \times 32 \times 32 \xrightarrow{\text{Conv3x3(64) + BN + ReLU + MaxPool2x2}} 64 \times 16 \times 16$
  3. $64 \times 16 \times 16 \xrightarrow{\text{Conv3x3(128) + BN + ReLU + MaxPool2x2}} 128 \times 8 \times 8$
  4. $\text{Global Average Pooling} \rightarrow 128$
  5. $\text{Dropout}(p=0.3) \rightarrow \text{Linear}(128, 10)$
- **Total Parameters**: 94,762 (all trainable).

### 2.3 Transfer Learning: ResNet-18 with Staged Fine-Tuning (Milestone 4)

#### Why ResNet-18?
1. **Residual Learning**: Deep convolutional networks suffer from degradation and vanishing gradients. ResNet introduces identity shortcut connections ($F(x) + x$), enabling efficient gradient flow through all 18 layers.
2. **Computational Efficiency**: With 11.18M parameters, ResNet-18 achieves high representational capacity while remaining lightweight and suitable for fast inference and deployment compared to heavy backbones (e.g. ResNet-50 with 25.6M or VGG-16 with 138M).
3. **ImageNet-1K Pretrained Representations**: Pretrained on 1.28 million natural images across 1,000 classes, early layers have already learned generalized visual primitives (Gabor edge filters, color transitions, textures) that transfer directly to aerial satellite scenes.

#### Staged Fine-Tuning Strategy
Training a deep network on a smaller dataset (18,900 images) risks **catastrophic forgetting** if all weights are updated with large learning rates from the start. We employ a two-stage strategy:

| Feature / Phase | Phase 1: Feature Extraction | Phase 2: Staged Fine-Tuning |
| :--- | :--- | :--- |
| **Objective** | Train new 10-class linear classification head without disturbing pretrained feature representations | Adapt high-level semantic representations to satellite domain specifics |
| **Frozen Layers** | Backbone: `conv1`, `bn1`, `layer1`, `layer2`, `layer3`, `layer4` | Early backbone: `conv1`, `bn1`, `layer1`, `layer2` (683,072 params) |
| **Trainable Layers** | `fc` classification head only | `layer3`, `layer4`, and `fc` classification head |
| **Trainable Params** | **5,130** (0.05% of model) | **10,498,570** (93.89% of model) |
| **Learning Rate** | $\eta = 10^{-3}$ | $\eta = 5 \times 10^{-5}$ (reduced 20x) |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $10^{-4}$) | AdamW (weight decay $10^{-4}$) |
| **LR Scheduler** | ReduceLROnPlateau (factor 0.5, patience 2) | CosineAnnealingLR ($\eta_{\min} = 10^{-6}$) |
| **Checkpoint** | `models/resnet18_phase1_best.pt` | `models/resnet18_best.pt` |

```text
ResNet-18 Staged Fine-Tuning Topology:
===================================================================
Input [B, 3, 64, 64]
   │
   ▼
[conv1 + bn1 + relu + maxpool] ──► FROZEN in Phase 1 & Phase 2 (Low-level edge filters)
   │
   ▼
[layer1 (2 BasicBlocks, 64-ch)] ──► FROZEN in Phase 1 & Phase 2 (Basic texture primitives)
   │
   ▼
[layer2 (2 BasicBlocks, 128-ch)] ─► FROZEN in Phase 1 & Phase 2 (Intermediate shape motifs)
   │
   ▼
[layer3 (2 BasicBlocks, 256-ch)] ─► FROZEN in Phase 1 │ UNFROZEN in Phase 2 (Semantic patterns)
   │
   ▼
[layer4 (2 BasicBlocks, 512-ch)] ─► FROZEN in Phase 1 │ UNFROZEN in Phase 2 (High-level context)
   │
   ▼
[avgpool (AdaptiveAvgPool2d)]
   │
   ▼
[fc (Linear: 512 -> 10)] ────────► UNFROZEN in Phase 1 & Phase 2 (EuroSAT Classification)
   │
   ▼
Output Logits [B, 10]
===================================================================
```

### 2.4 Model Comparison Methodology (Controlled Experimentation)
To establish a rigorous, scientifically valid comparison between learning from scratch vs. transfer learning:
1. **Identical Partitions**: Exactly the same stratified 70/15/15 dataset splits (fixed seed 42) are used for both Custom CNN and ResNet-18.
2. **Identical Preprocessing**: Same deterministic evaluation transformations and empirical dataset statistics.
3. **Primary Selection Metric**: Macro-averaged F1 score on the validation split ($N=4,050$).
4. **Data Leakage Elimination**: The test partition ($N=4,050$) is completely quarantined and never accessed during training, early stopping, or hyperparameter selection. Test evaluation is reserved exclusively for Milestone 5.

### 2.5 Evaluation Metric: Macro F1
$$\text{Macro } F_1 = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
Where $C=10$. Macro F1 weights every class equally, preventing dominant classes from masking degradation on difficult minority classes (e.g. `PermanentCrop` vs `HerbaceousVegetation`).

---

## 3. Explainability: Grad-CAM Formulation

Grad-CAM computes the gradient of the score for target class $c$, $y^c$, with respect to feature activation maps $A^k$ of the final convolutional layer:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
The class-discriminative localization map $L_{\text{Grad-CAM}}^c$ is computed as:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
- Normal inference operates under `torch.no_grad()`.
- Grad-CAM path enables gradient computation on-demand only when explainability is requested.
