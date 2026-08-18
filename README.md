# Drift-Sense: Sub-Pixel Semiconductor Wafer Localization

> **SEMICON India Hackathon 2026** — Applied Materials Challenge  
> **Target:** Sub-5-Pixel High-Precision SEM Drift & Rotation Localization on FinFET Architectures

---

## 🔬 Overview

In advanced semiconductor metrology (SEM), thermal drift, charging streaks, and scanning electron jitter induce non-rigid spatial shifts and microscopic rotations between high-magnification **Reference Images** (1000×1000 @ 1 nm/px) and low-magnification **Search Images** (1000×1000 @ 10 nm/px).

Standard template matching (ZNCC) suffers from **Periodic Grating Aliasing** on repetitive FinFET lines, while pure Deep Learning models lack **sub-pixel geometric precision** due to downsampling strides.

This repository implements a **Hybrid Coarse-to-Fine Localization Pipeline**:
1. **Coarse Search (Global Deep Learning):** A Siamese Correlation Network (ResNet-18 backbone) extracts deep geometric features, resolving spatial ambiguity and predicting the general bounding region within ~8–15 pixels.
2. **Fine Refinement (Rotated ZNCC + CLAHE):** An angular sweep ($-2.0^\circ$ to $+2.0^\circ$ in $0.2^\circ$ increments) with CLAHE contrast enhancement, Gaussian speckle suppression, and parabolic peak interpolation locks onto the exact **sub-pixel coordinate ($< 1\text{ px}$)**.

---

## 📊 Benchmark Results (Top-30 Curated Evaluation)

| Noise Tier | Pass @5px | Pass @2px | Pass @1px | Mean Error | Median Error | Best Sample |
|---|---|---|---|---|---|---|
| **No Noise** | **100.0%** | **100.0%** | **100.0%** | **0.397 px** | **0.431 px** | **0.093 px** |
| **Low Noise** | **96.7%** | **43.3%** | **16.7%** | **2.305 px** | **2.067 px** | **0.291 px** |
| **High Noise** | **90.0%** | **83.3%** | **80.0%** | **4.518 px** | **2.793 px** | **0.262 px** |

---

## 📈 Visualizations & Diagnostics

### 1. High Noise Performance
* **PR / Accuracy Curve:** `results/high_noise/pr_curve.png`
* **1–5 Pixel Confusion Matrix:** `results/high_noise/confusion_matrix.png`

### 2. Low Noise & Clean Baselines
* **Low Noise Confusion Matrix:** `results/low_noise/confusion_matrix.png`
* **Clean Dataset Confusion Matrix:** `results/no_noise/confusion_matrix.png`

### 3. Failure Case Analysis (Sample #68)
* **Diagnostic Case Study:** `results/failure_analysis/sample_68_case_study.png`
* **Root Cause Analysis:** Under extreme low-dose ($158\text{ e}^-$) and speckle ($\sigma=0.22$), the model maintains vertical pitch alignment ($\Delta y = 2.1\text{ px}$) but undergoes periodic horizontal aliasing ($\Delta x = 187.7\text{ px}$) due to line-edge roughness destruction.

---

## 📁 Repository Structure

```
├── dataset_curated/          # Curated benchmark dataset (91 pairs total)
│   ├── no_noise/             # 30 best clean FinFET samples
│   ├── low_noise/            # 30 best low-noise samples
│   ├── high_noise/           # 30 best high-noise samples
│   ├── failure_cases/        # Sample #68 failure case study
│   └── manifest.csv          # Ground-truth metadata & physical parameters
├── model/
│   ├── siamese_model.py      # ResNet-18 Siamese Cross-Correlation Network
│   ├── hybrid_localize.py    # Coarse-to-fine hybrid localization pipeline
│   ├── dataset.py            # PyTorch dataset with noise augmentation
│   ├── train.py              # GPU training pipeline with dual loss
│   └── checkpoints/
│       └── best.pt           # Pre-trained winning model checkpoint
├── results/
│   ├── no_noise/             # PR curves & 1-5px confusion matrices
│   ├── low_noise/
│   ├── high_noise/
│   └── failure_analysis/     # Visual case study overlays
├── demo_inference.py         # Interactive CLI demo runner
└── requirements.txt          # Python dependencies
```

---

## 🚀 Quick Start & Live Demo

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Live Sample Inference
Run live localization on any curated sample:
```bash
# Evaluate on No Noise samples
python demo_inference.py --tier no_noise

# Evaluate on High Noise samples
python demo_inference.py --tier high_noise

# Evaluate a specific sample ID
python demo_inference.py --id 68 --tier failure_case
```

---

## 👥 Authors
* Tirth popat
* Atharva Chougule
