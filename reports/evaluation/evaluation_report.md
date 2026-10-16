# Satellite Land-Use Classifier: Milestone 5 Final Evaluation & Model Comparison

> **EuroSAT RGB Land-Use / Land-Cover Classification Benchmark**

---

## 1. Executive Summary

This report presents the rigorous, reproducible final evaluation of the **Custom CNN baseline** and the fine-tuned **ResNet-18 transfer learning** model on the untouched EuroSAT RGB test partition. Both models were evaluated on the exact same **4,050 test images** using identical deterministic preprocessing pipelines and zero gradient computation.

- **Primary Model Selection Metric (Macro F1)**: **resnet18 achieved 0.9477** vs **custom_cnn achieved 0.7406** (an absolute improvement of **+0.2071** / **20.71%**).
- **Test Accuracy**: **resnet18 achieved 94.99%** vs **custom_cnn at 75.31%**.
- **Total Errors on Test Set**: resnet18 committed **203 errors** (5.01% error rate), compared to **1000 errors** (24.69% error rate) for custom_cnn.

---

## 2. Evaluation Metadata & Strict Protocol

| Parameter | Specification / Value |
|---|---|
| **Evaluation Date / Timestamp** | `2026-10-15T23:51:39.619585` |
| **Dataset Partition** | `test` (Partition manifest: `data/processed/splits.json`) |
| **Evaluated Sample Count** | **4,050 images** (15% stratified test partition) |
| **Hardware Execution Device** | `cpu` |
| **Image Input Dimensions** | `64 x 64` pixels, 3 RGB channels |
| **Deterministic Preprocessing** | Resize(64,64) -> ToTensor() -> EuroSAT Dataset Normalization |
| **Normalization Statistics** | EuroSAT Mean `[0.3444, 0.3803, 0.4078]`, Std `[0.2026, 0.1368, 0.1156]` |
| **Gradient Computation** | Strictly disabled (`torch.inference_mode()`) |
| **Weight Integrity Verification** | SHA-256 parameter hash verified before and after evaluation (Zero mutation) |
| **Zero Division Policy** | `zero_division=0.0` (Safely handles zero precision/recall cases) |
| **Custom CNN Checkpoint** | `C:\Users\DELL\OneDrive\Desktop\College\OJT\models\custom_cnn_best.pt` |
| **ResNet-18 Checkpoint** | `C:\Users\DELL\OneDrive\Desktop\College\OJT\models\resnet18_best.pt` |

---

## 3. Overall Model Comparison

| Metric / Dimension | Custom CNN (Baseline) | ResNet-18 (Transfer Learning) | Delta (ResNet - CNN) | Top Performer |
|---|---|---|---|---|
| **Accuracy** | 0.7531 | 0.9499 | 0.1968 | **resnet18** |
| **Macro Precision** | 0.7481 | 0.9484 | 0.2003 | **resnet18** |
| **Macro Recall** | 0.7485 | 0.9472 | 0.1987 | **resnet18** |
| **Macro F1-Score (Primary)** | 0.7406 | 0.9477 | 0.2071 | **resnet18** |
| **Weighted F1-Score** | 0.7503 | 0.9497 | 0.1994 | **resnet18** |
| **Total Parameters** | 94762 | 11181642 | 11086880 | **custom_cnn (lighter)** |
| **Inference Time Full Test Set (s)** | 6.7126 | 10.3457 | 3.6331 | **custom_cnn** |
| **Avg Inference Time per Image (ms)** | 1.657 | 2.555 | 0.898 | **custom_cnn** |

---

## 4. Per-Class Performance Breakdown

Canonical EuroSAT alphabetical class ordering preserved:

| # | Land-Cover Class | Support | Custom CNN Prec | Custom CNN Rec | Custom CNN F1 | ResNet-18 Prec | ResNet-18 Rec | ResNet-18 F1 | F1 Delta |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **AnnualCrop** | 450 | 0.8456 | 0.8400 | 0.8428 | 0.9281 | 0.9467 | **0.9373** | +0.0945 |
| 2 | **Forest** | 450 | 0.8923 | 0.7733 | 0.8286 | 0.9757 | 0.9800 | **0.9778** | +0.1492 |
| 3 | **HerbaceousVegetation** | 450 | 0.6978 | 0.6978 | 0.6978 | 0.9349 | 0.9578 | **0.9462** | +0.2484 |
| 4 | **Highway** | 375 | 0.6129 | 0.4053 | 0.4880 | 0.9306 | 0.8933 | **0.9116** | +0.4236 |
| 5 | **Industrial** | 375 | 0.8858 | 0.9307 | 0.9077 | 0.9683 | 0.9760 | **0.9721** | +0.0644 |
| 6 | **Pasture** | 300 | 0.5049 | 0.8567 | 0.6354 | 0.9360 | 0.9267 | **0.9313** | +0.2959 |
| 7 | **PermanentCrop** | 375 | 0.6530 | 0.6373 | 0.6451 | 0.9296 | 0.8800 | **0.9041** | +0.2590 |
| 8 | **Residential** | 450 | 0.8852 | 0.9422 | 0.9128 | 0.9822 | 0.9822 | **0.9822** | +0.0694 |
| 9 | **River** | 375 | 0.6624 | 0.5547 | 0.6038 | 0.9141 | 0.9360 | **0.9249** | +0.3211 |
| 10 | **SeaLake** | 450 | 0.8411 | 0.8467 | 0.8439 | 0.9846 | 0.9933 | **0.9889** | +0.1450 |

---

## 5. Error Analysis & Confusion Matrix Diagnosis

### 5.1 Most Difficult and Easiest Classes

#### Custom CNN:
- **Top 3 Highest F1 Classes**: Residential (0.9128), Industrial (0.9077), SeaLake (0.8439)
- **Top 3 Lowest F1 Classes**: Pasture (0.6354), River (0.6038), Highway (0.4880)

#### ResNet-18:
- **Top 3 Highest F1 Classes**: SeaLake (0.9889), Residential (0.9822), Forest (0.9778)
- **Top 3 Lowest F1 Classes**: River (0.9249), Highway (0.9116), PermanentCrop (0.9041)

### 5.2 Top Confusion Pairs

#### Custom CNN Top Confusion Pairs:
| True Class | Predicted Class | Misclassified Count | Share of Total Errors |
|---|---|---|---|
| Forest | Pasture | 84 | 8.4% |
| PermanentCrop | HerbaceousVegetation | 63 | 6.3% |
| River | Highway | 58 | 5.8% |
| Highway | River | 54 | 5.4% |
| Highway | PermanentCrop | 49 | 4.9% |

#### ResNet-18 Top Confusion Pairs:
| True Class | Predicted Class | Misclassified Count | Share of Total Errors |
|---|---|---|---|
| Highway | River | 22 | 10.8% |
| PermanentCrop | HerbaceousVegetation | 18 | 8.9% |
| PermanentCrop | AnnualCrop | 15 | 7.4% |
| River | Highway | 11 | 5.4% |
| AnnualCrop | PermanentCrop | 10 | 4.9% |

### 5.3 Remote Sensing Domain Explanations
- **Highway vs River**: Both feature narrow, linear spectral signatures threading across rural or urban terrain. At 64x64 resolution, asphalt and dark water bodies share low reflectance patterns that confuse basic convolutional feature detectors.
- **Pasture vs HerbaceousVegetation**: Both represent non-forest green vegetation. Distinguishing managed grazing fields from natural grasslands requires deep hierarchical texture cues learned effectively by pretrained ResNet-18.
- **AnnualCrop vs PermanentCrop**: Crop cycles produce seasonal soil exposure and regular furrow textures that occasionally overlap in spectral space.

---

## 6. Inference Latency & Benchmarking Methodology

- **Hardware Platform**: `cpu`
- **Methodology**: Evaluated using warm-up passes (5 batches) prior to timing. CUDA synchronization invoked before and after batch executions when running on GPU. Checkpoint loading and dataset initialization overhead were strictly excluded from the timing window.
- **Custom CNN Latency**: 6.71 s total (1.66 ms/image).
- **ResNet-18 Latency**: 10.35 s total (2.56 ms/image).
- **Generalization Note**: Latency results depend heavily on host CPU architecture, thread allocation, batch size, and thermal throttling; timing measurements reported here reflect this specific host setup.

---

## 7. Academic Insights & Viva Alignment

### Why Macro F1 is the Decisive Metric
Accuracy treats all samples equally, meaning a model can achieve high accuracy by mastering majority classes while failing on difficult or rarer classes. **Macro F1** computes unweighted per-class harmonic means: $$\text{Macro } F_1 = \frac{1}{C} \sum_{c=1}^C F_{1, c}$$ This penalizes models that fail on subtle classes (e.g., HerbaceousVegetation, River) and guarantees balanced domain mapping.

### Why Test Performance Differs from Validation Performance
- **Untouched Partition**: The test partition was never exposed to training, learning-rate scheduling, or early-stopping decisions.
- **Validation Selection Bias**: The validation set was used to select the best checkpoint epoch; test evaluation gives an unbiased generalization estimate.

---

## 8. Generated Output Artifacts

- `reports/evaluation/custom_cnn_metrics.json`
- `reports/evaluation/resnet18_metrics.json`
- `reports/evaluation/model_comparison.json`
- `reports/evaluation/model_comparison.csv`
- `reports/evaluation/per_class_metrics.csv`
- `reports/evaluation/custom_cnn_confusion_matrix.png`
- `reports/evaluation/custom_cnn_confusion_matrix.csv`
- `reports/evaluation/resnet18_confusion_matrix.png`
- `reports/evaluation/resnet18_confusion_matrix.csv`
- `reports/evaluation/per_class_f1_comparison.png`
- `reports/evaluation/error_analysis.csv`
- `reports/evaluation/evaluation_report.md`
