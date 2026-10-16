# Satellite Land-Use Classifier

An end-to-end deep learning application for classifying Sentinel-2 satellite image tiles into 10 land-use / land-cover (LULC) categories using the **EuroSAT RGB** dataset. Built with PyTorch, FastAPI, React + TypeScript + Tailwind CSS, and Grad-CAM visual explainability.

---

## 1. Project Overview

Land-Use and Land-Cover (LULC) mapping from Earth Observation (EO) satellite imagery is essential for urban planning, environmental monitoring, agricultural yield estimation, and natural resource management.

This project implements:
1. **Automated EuroSAT Acquisition & Validation**: Robust dataset integrity and corrupt image detection.
2. **Reproducible Preprocessing Pipeline**: Deterministic evaluation preprocessing and domain-appropriate training augmentations (rotations, flips).
3. **Custom CNN Baseline**: Lightweight 3-block convolutional network with Batch Normalization, ReLU, Global Average Pooling, trained from scratch.
4. **ResNet-18 Transfer Learning**: ImageNet-pretrained backbone with staged fine-tuning (head training → selective layer unfreezing).
5. **Rigorous Evaluation & Model Selection**: Evaluation against validation Macro F1 score, confusion matrices, per-class metrics, and error analysis.
6. **Grad-CAM Visual Explainability**: Class-activation maps indicating tile regions influencing predictions.
7. **FastAPI Backend**: Low-latency REST API with strict Pydantic schema validation, image decoding validation, and batch inference.
8. **React + TypeScript + Tailwind Frontend**: Interactive dashboard with single tile inference, confidence scores, probability distributions, Grad-CAM overlays, and multi-tile land-cover grid visualization.
9. **Engineering Best Practices**: Docker & Docker Compose deployment, pytest automated testing suite, and GitHub Actions CI.

---

## 2. Dataset: EuroSAT RGB

- **Sensor**: Sentinel-2 optical satellite imagery (European Space Agency).
- **Format**: RGB 3-channel tiles, $64 \times 64$ pixels.
- **Classes (10 total)**:
  1. `AnnualCrop`
  2. `Forest`
  3. `HerbaceousVegetation`
  4. `Highway`
  5. `Industrial`
  6. `Pasture`
  7. `PermanentCrop`
  8. `Residential`
  9. `River`
  10. `SeaLake`

---

## 3. Project Architecture & Repository Layout

```text
satellite-land-use-classifier/
├── data/
│   ├── raw/                 # Original EuroSAT image files
│   └── processed/           # Processed datasets / split metadata
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory Data Analysis & visual inspections
│   └── 02_error_analysis.ipynb # Systematic misclassification analysis
├── src/
│   ├── constants.py         # Single source of truth (classes, colors, paths)
│   ├── config.py            # YAML configuration loader
│   ├── data/                # Data download, integrity checking, split manager
│   ├── preprocessing/       # PyTorch datasets and torchvision transforms
│   ├── models/              # Custom CNN & ResNet-18 architectures
│   ├── training/            # Training engine, early stopping, LR schedulers
│   ├── evaluation/          # Macro F1, confusion matrices, per-class metrics
│   ├── explainability/      # Grad-CAM implementation & heatmap generation
│   └── inference/           # Unified model loader & inference engine
├── app/
│   ├── main.py              # FastAPI application factory
│   ├── routes/              # /health, /predict, /predict/explain, /predict/batch, /classes
│   ├── schemas/             # Pydantic input/output schemas
│   └── services/            # Inference service wrapper
├── frontend/                # React + TypeScript + Tailwind CSS dashboard
├── tests/
│   ├── unit/                # Unit tests (preprocessing, model shapes, transforms)
│   ├── api/                 # FastAPI endpoint test client suites
│   └── integration/         # End-to-end inference flow tests
├── models/                  # Versioned checkpoints and artifact metadata
├── configs/
│   ├── base_config.yaml     # Global dataset & server settings
│   ├── custom_cnn.yaml      # Baseline model hyperparameters
│   └── resnet18.yaml        # Transfer learning & staged fine-tuning settings
├── scripts/
│   └── check_env.py         # Environment & configuration audit script
├── docs/
│   └── architecture.md      # Architectural design document & Viva notes
├── requirements.txt         # Pinned Python dependencies
├── Dockerfile               # Container build definition
├── docker-compose.yml       # Local deployment composition
└── README.md
```

---

## 4. Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd satellite-land-use-classifier
   ```

2. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Python dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify environment setup:**
   ```bash
   python scripts/check_env.py
   ```

---

## 5. Single Image Inference

Use `scripts/predict_image.py` to run inference on individual satellite tiles using the trained checkpoints:

```bash
# Predict with fine-tuned ResNet-18
python scripts/predict_image.py --image "data/raw/EuroSAT_RGB/Forest/Forest_1.jpg" --model resnet18

# Predict with Custom CNN baseline
python scripts/predict_image.py --image "data/raw/EuroSAT_RGB/Forest/Forest_1.jpg" --model custom_cnn

# Side-by-side model comparison with Top-3 probability ranking and optional label check
python scripts/predict_image.py --image "data/raw/EuroSAT_RGB/Forest/Forest_1.jpg" --model both --label Forest

# Export structured JSON prediction results
python scripts/predict_image.py --image "data/raw/EuroSAT_RGB/Forest/Forest_1.jpg" --model both --save-json prediction.json
```

### Expected Output Example

```text
============================================================
                 MODEL COMPARISON
============================================================
Image               : data/raw/EuroSAT_RGB/Forest/Forest_1.jpg
Device              : CPU
Original Image Size : 64x64
Model Input Size    : 64x64 (Converted to RGB)
------------------------------------------------------------

Custom CNN:
  Prediction : Forest
  Confidence : 63.98%
  Top 3:
  1. Forest                    63.98%
  2. River                     16.90%
  3. SeaLake                    7.79%

ResNet-18:
  Prediction : Forest
  Confidence : 99.96%
  Top 3:
  1. Forest                    99.96%
  2. Pasture                    0.02%
  3. HerbaceousVegetation       0.02%
------------------------------------------------------------
Agreement : YES
============================================================
```

> **Note on Domain Generalization**: The models were trained on Sentinel-2 EuroSAT RGB imagery. Predictions on external satellite images or non-orthorectified photos may be unreliable if their spectral/spatial characteristics differ substantially from EuroSAT.

---

## 6. Final Model Evaluation & Benchmark Comparison (Milestone 5)

The final evaluation assesses both trained models on the **strictly untouched test partition** (4,050 satellite images across all 10 canonical classes, representing 15% stratified EuroSAT data).

### Running Final Evaluation

```bash
# Evaluate both Custom CNN and ResNet-18 head-to-head (default)
python scripts/evaluate_models.py

# Evaluate ResNet-18 only
python scripts/evaluate_models.py --model resnet18

# Evaluate Custom CNN only
python scripts/evaluate_models.py --model custom_cnn

# Custom batch size or destination directory
python scripts/evaluate_models.py --batch-size 128 --output-dir reports/evaluation

# Run the evaluation smoke test
python scripts/smoke_test_evaluation.py
```

### Verified Test Partition Results

| Evaluation Metric / Dimension | Custom CNN (Baseline) | ResNet-18 (Transfer Learning) | Delta (ResNet-18 - CNN) | Top Performer |
|---|---|---|---|---|
| **Test Accuracy** | **75.31%** (0.7531) | **94.99%** (0.9499) | **+19.68%** (+0.1968) | **ResNet-18** |
| **Macro Precision** | 0.7481 | 0.9484 | +0.2003 | **ResNet-18** |
| **Macro Recall** | 0.7485 | 0.9472 | +0.1987 | **ResNet-18** |
| **Macro F1-Score (Primary)** | **0.7406** | **0.9477** | **+0.2071** (+20.71%) | **ResNet-18** |
| **Weighted F1-Score** | 0.7503 | 0.9497 | +0.1994 | **ResNet-18** |
| **Total Parameter Count** | **94,762** | 11,181,642 | +11,086,880 | **Custom CNN (118x lighter)** |
| **Inference Time (Full Test Set)** | **6.71 s** | 10.35 s | +3.63 s | **Custom CNN** |
| **Average Latency per Image** | **1.66 ms** | 2.56 ms | +0.90 ms | **Custom CNN** |
| **Total Test Misclassifications** | 1,000 / 4,050 | **203 / 4,050** | **-797 errors (-79.7%)** | **ResNet-18** |

### Evaluated Metrics

1. **Accuracy**: Fraction of all test samples correctly predicted: $\frac{\text{Correct}}{\text{Total}}$.
2. **Macro Precision & Recall**: Arithmetic unweighted mean of per-class precision and recall.
3. **Macro F1-Score (Primary Metric)**: Harmonic mean of precision and recall averaged equally across all 10 classes:
   $$\text{Macro } F_1 = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
4. **Weighted F1-Score**: Support-weighted mean of per-class F1-scores.
5. **Per-Class Metrics**: Precision, Recall, F1-Score, and Support for each of the 10 canonical EuroSAT classes.

### Generated Artifacts Location

All evaluation artifacts are saved in `reports/evaluation/`:
- `reports/evaluation/custom_cnn_metrics.json`: Custom CNN overall and per-class metrics.
- `reports/evaluation/resnet18_metrics.json`: ResNet-18 overall and per-class metrics.
- `reports/evaluation/model_comparison.json`: Side-by-side JSON comparison.
- `reports/evaluation/model_comparison.csv`: Side-by-side tabular comparison.
- `reports/evaluation/per_class_metrics.csv`: Per-class support, precision, recall, and F1 with deltas.
- `reports/evaluation/custom_cnn_confusion_matrix.png`: High-res (300 DPI) Custom CNN confusion heatmap.
- `reports/evaluation/custom_cnn_confusion_matrix.csv`: Raw and normalized counts.
- `reports/evaluation/resnet18_confusion_matrix.png`: High-res (300 DPI) ResNet-18 confusion heatmap.
- `reports/evaluation/resnet18_confusion_matrix.csv`: Raw and normalized counts.
- `reports/evaluation/per_class_f1_comparison.png`: Side-by-side bar chart of per-class F1-scores.
- `reports/evaluation/error_analysis.csv`: Sample-level diagnostics with image paths, predictions, confidences.
- `reports/evaluation/evaluation_report.md`: Complete evaluation report.

### Confusion Matrix Interpretation Guide

- **Rows** correspond to ground-truth labels (**True Class**).
- **Columns** correspond to model predictions (**Predicted Class**).
- **Diagonal cells** indicate correctly identified land-cover tiles.
- **Off-diagonal cells** indicate confusion/misclassification. For example:
  - Row `Highway`, Column `River`: High count indicates highways misclassified as rivers (linear spectral overlap).
  - Row `Forest`, Column `Pasture`: High count in baseline CNN indicates green vegetation texture confusion, largely eliminated by ResNet-18.
- Normalized confusion matrices display each cell as a percentage of true class support (row sum = 100%).

### Key Methodological & Academic Insights

#### 1. Why Macro F1 Differs from Simple Accuracy
Accuracy weights every image equally. If a dataset has class imbalance or certain classes are disproportionately easy (e.g. `Residential` or `SeaLake`), high accuracy can mask complete failure on subtle classes (e.g. `Highway` or `River`). **Macro F1** computes unweighted class averages, penalizing any model that sacrifices difficult classes.

#### 2. Why the Test Set Must Never Be Used for Model Selection
Using test data to choose hyperparameters, learning rates, epochs, or model architectures induces **information leakage** and **optimistic selection bias**. The test partition must remain sealed until final evaluation to guarantee an unbiased estimate of real-world generalization performance.

#### 3. Why Validation Performance May Differ from Test Performance
- **Checkpoint Selection Bias**: Checkpoints are chosen at the epoch where validation Macro F1 peaks (e.g. ResNet-18 achieved 0.9309 val Macro F1). The validation score reflects peak performance on that specific partition.
- **Stochastic Sampling Variance**: While both partitions are stratified 15% splits, slight variance in difficult tiles exists across partitions. Here, ResNet-18 generalized exceptionally well on the test set, achieving **0.9477 Macro F1** and **94.99% Accuracy**.

---

## 7. Earth Observer Web Application (Milestone 6: Premium MVP with 3D Earth)

The **Earth Observer** web application is a full-stack Earth Observation platform featuring a cinematic deep-space aesthetic, an interactive rotating 3D Earth visualizer, and a real deep learning satellite land-use inference pipeline.

### Web Application Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                   React 18 + TypeScript Frontend             │
│  - Three.js 3D Rotating Earth (Atmosphere, Stars, Orbit)     │
│  - Satellite Image Intelligence Workspace (Drag & Drop)     │
│  - Interactive Model Selector (ResNet-18 vs Custom CNN)      │
│  - Top-3 Probability Bars & Full 10-Class Distribution      │
│  - Deep-Space Glassmorphism Design System (Vanilla CSS)      │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP REST API (/predict, /health)
┌──────────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend Engine                   │
│  - GET  /health         (Live model availability & device)  │
│  - GET  /classes        (10 EuroSAT classes & color palettes)│
│  - GET  /samples        (Curated demo satellite tiles)      │
│  - POST /predict        (Real-time PyTorch model inference) │
│  - Static Mount         (Serves compiled frontend at /)     │
└──────────────────────────────┬──────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────┐
│                  PyTorch Inference Pipeline                 │
│  - ResNet-18 Transfer Learning (models/resnet18_best.pt)    │
│  - Custom CNN Baseline         (models/custom_cnn_best.pt)  │
│  - Deterministic Preprocessing (Resize, ToTensor, Normalize)│
└─────────────────────────────────────────────────────────────┘
```

### Running the Application (Windows Commands)

#### Option 1: Unified Single-Server Mode (Recommended)
FastAPI automatically serves the production-compiled frontend bundle at `http://127.0.0.1:8000/`:

```powershell
# 1. Build the frontend bundle (one-time build)
cd frontend
npm run build
cd ..

# 2. Launch the FastAPI server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser.

#### Option 2: Live Development Mode (Hot-Reloading)
Run backend and frontend concurrently in two separate terminal windows:

**Terminal 1 (FastAPI Backend):**
```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (Vite Frontend):**
```powershell
cd frontend
npm run dev
```
Open **`http://localhost:5173`** in your browser (Vite automatically proxies API requests to `http://127.0.0.1:8000`).

### User Workflow: Upload & Classify Satellite Tiles

1. **Explore the Hero Landing Page**: View the rotating 3D Earth, starfield, and orbital ring with live telemetry badges. The navigation bar displays **`LOCAL MODEL READY (CPU)`**.
2. **Navigate to the Workspace**: Click **"Explore Classification"** to smooth-scroll down to the **Satellite Image Intelligence** panel.
3. **Select a Target Model**:
   - **ResNet-18 Transfer Learning** (Default: 94.99% test accuracy, 11.2M parameters).
   - **Custom CNN Baseline** (75.31% test accuracy, 94.8K parameters).
4. **Choose an Input Image**:
   - **Quick Test Demo Tiles**: Click any of the 10 curated EuroSAT sample chips (`Forest`, `SeaLake`, `Residential`, `Highway`, etc.) for immediate one-click testing.
   - **Upload Local Image**: Drag and drop any satellite tile (`.jpg`, `.jpeg`, `.png`) or click **"Browse Files"**.
5. **Execute AI Inference**: Click **"Analyze Image"**. The backend performs real deterministic preprocessing, runs forward inference, and calculates Softmax probabilities.
6. **Inspect Results**:
   - Primary Predicted Land-Cover Class with assigned hex color.
   - Exact Confidence percentage.
   - Top-3 candidate predictions with animated horizontal probability bars.
   - Forward pass execution latency in milliseconds.
   - Optional expandable view showing all 10 class probabilities.

### 3D Earth Visualizer & Texture Architecture

- **Rendering Engine**: Three.js (`three` v0.160.0) rendering a 64×64 segment `SphereGeometry`.
- **Texture Generation**: High-resolution (2048×1024) equirectangular procedural canvas texture generated via multi-frequency Fractal Brownian Motion (fBm) noise with realistic ocean bathymetry, continental landmasses, arid regions, lush vegetation, and polar ice caps.
- **Self-Contained & 100% Offline**: Requires zero external network downloads, eliminating CORS errors and broken asset links during offline viva demonstrations.
- **Lighting & Atmosphere**: Sun key light + ambient space light + dual-mesh additive atmosphere rim glow (`#00E5FF`).
- **Orbital Ring & Satellite**: Elliptical orbit tilted at Earth's 23.5° axial obliquity with an orbiting satellite beacon.
- **Graceful WebGL Fallback**: Automatically displays an animated celestial CSS/SVG card if WebGL context creation fails.
- **Performance & Battery Saving**: Integrated `IntersectionObserver` pauses Three.js render loop when scrolled off-screen; respects `prefers-reduced-motion`.

### Known Limitations

1. **Closed-Set Classification**: Models are closed-set classifiers over the 10 EuroSAT land-use classes. Non-satellite photos (people, vehicles, domestic animals) will be projected onto one of the 10 classes.
2. **Sensor Spatial Resolution**: Optimized for Sentinel-2 optical tiles ($64 \times 64$ pixels, 10 m/pixel ground sample distance). Very high-resolution drone imagery or non-overhead oblique photos may exhibit domain shift.

---

## 8. Incremental Development Roadmap

- [x] **Milestone 0: Project Audit & Setup** (Repository structure, configs, constants, env validation)
- [x] **Milestone 1: Dataset Acquisition & Validation** (Download, corruption check, split management, EDA)
- [x] **Milestone 2: Preprocessing & DataLoaders** (Deterministic eval transforms, training augmentations, unit tests)
- [x] **Milestone 3: Custom CNN Baseline** (Architecture, training loop, early stopping, validation curves)
- [x] **Milestone 4: ResNet-18 Transfer Learning** (Staged fine-tuning, validation model comparison)
- [x] **Milestone 5: Final Evaluation & Error Analysis** (Test partition evaluation, confusion matrices, error analysis reports)
- [x] **Milestone 6: Premium MVP with 3D Earth Visualization** (FastAPI backend, React 18, Three.js 3D Earth, full-stack integration)
- [ ] **Milestone 7: Grad-CAM Visual Explainability** (Heatmap generator, overlay API endpoints, visual explanation cards)
- [ ] **Milestone 8: Tile-Level Land-Cover Grid Analysis** (Batch upload, spatial grid display, regional land-cover breakdown)
- [ ] **Milestone 9: Docker Containerization & CI/CD** (Docker Compose multi-stage build, GitHub Actions automated CI)

---

## 9. Academic Defense & Viva Alignment

- **Why EuroSAT?** Standard benchmark with 10 distinct spectral/spatial patterns in Sentinel-2 optical imagery.
- **Why Macro F1 over simple Accuracy?** Macro F1 treats each land-cover class equally regardless of class imbalance, penalizing models that perform poorly on rarer or harder classes.
- **Why 3D Earth Visualization?** Provides spatial context for satellite remote sensing and elevates the portfolio presentation from a basic model script into a functional Earth Observation intelligence platform.
- **Why Uncertainty Thresholding over Naive Closed-Set Prediction?** Standard closed-set softmax models will unconditionally assign 100% of probability mass across the 10 training classes even when presented with a dog photograph, human portrait, or random noise. Incorporating empirical uncertainty thresholds prevents blind misclassification of unsupported inputs.

---

## 10. Uncertainty Detection & Out-of-Distribution (OOD) Awareness

### Empirical Validation Threshold Selection (Held-Out Validation Partition, N=4,050)
The maximum Softmax probability (MSP) operating thresholds were calibrated exclusively on the **EuroSAT validation split** (preserving the test set untouched):

| Model | Operating Threshold ($\tau$) | EuroSAT Validation Acceptance | EuroSAT False Rejection Rate (FRR) | Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **ResNet-18 Transfer** | `0.60` | **96.42%** | **3.58%** | Sharp softmax calibration; filters ambiguous inputs while retaining >96% of valid tiles. |
| **Custom CNN Baseline** | `0.40` | **88.35%** | **11.65%** | Flatter logits due to training from scratch; balanced to avoid excessive false rejection. |

### API Schema Integration & Backward Compatibility
The API response schema introduces dedicated uncertainty fields while preserving all existing prediction keys:
- `status`: `"accepted"` vs `"potentially_unsupported"`
- `is_confident`: `true` if confidence $\ge \tau$, `false` otherwise
- `threshold_used`: Float threshold evaluated (default or client-specified)
- `uncertainty_reason`: Detailed explanation provided when prediction is uncertain

### Scientific Limitations of Maximum Softmax Probability (MSP)
1. **Texture Triggering**: When an OOD image contains dominant textures matching in-domain spectral signatures (e.g., green grass background in a dog photo), deep networks may still produce high confidence (e.g., predicting `HerbaceousVegetation`).
2. **High-Frequency Noise**: Deep ReLU networks can yield high confidence on uniform noise due to corner/edge filter activations.
3. **Future Roadmap for Production OOD Detection**:
   - Penultimate Layer Mahalanobis Distance (Lee et al., 2018)
   - Free Energy Scoring (Liu et al., 2020)
   - Autoencoder Reconstruction Error Validation

