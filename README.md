# 🖊️ Signature Detection & Verification — Deep Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch)
![CNN](https://img.shields.io/badge/CNN-Custom%20Architecture-purple)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Mumbai University](https://img.shields.io/badge/Mumbai%20University-BE%20AI%20%26%20ML-darkblue)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Skills:** Python · PyTorch · CNN · Cosine Similarity · Computer Vision · OpenCV · Deep Learning  
> **Association:** A.P. Shah Institute of Technology, Thane — BE AI & ML, Mumbai University  
> **Dataset:** Tobacco 800 (detection) · Kaggle Signature Dataset (verification)

---

## Why This Matters

Manual signature verification costs banks and legal firms thousands of hours annually — and a single missed forgery can cost millions. This pipeline automates all three stages: 
Detection, cleaning and verification.

---

## 📌 Project Overview

An end-to-end deep learning pipeline for automated **signature detection and verification** in document images — addressing the limitations of manual signature verification in banking, legal, and government contexts.

**Three-stage pipeline:**
1. **Detection** — Custom CNN detects and localises signatures in document images (trained on Tobacco 800 dataset)
2. **Noise Removal** — CNN-based post-processing removes stamps, printed text, and background artifacts
3. **Verification** — Cosine similarity (threshold = 0.8) matches cleaned signatures against reference signatures

**Key result:** Verification AUC 1.0000, Detection accuracy ~90%, with clear score separation between genuine (mean ~0.85) and forged (mean ~0.55) signatures.

---

## 🗂️ Repository Structure & How Files Connect

```
📁 Signature-Detection-Verification/
│
├── 📂 Visualisation Charts/   ← All 7 output charts (fig1–fig7)
├── 📓 Signature_Detection_Verification.ipynb  ← MAIN FILE: Full pipeline
│   │   Sections: Setup → Dataset → CNN → Training → Noise Removal → Verification → ROC
│   │   Outputs:  fig1–fig7 PNGs 
│
├── 📦 requirements.txt  ← Python dependencies
└── 📜 LICENSE           ← MIT License
```

### 🔗 Pipeline Flow

```
Document Image
      │
      ▼
[Stage 1] Custom CNN Detection         ← Tobacco 800 dataset
      │    Localise signature bounding box
      ▼
[Stage 2] Noise Artifact Removal       ← CNN-based cleaning
      │    Remove stamps, printed text, background noise
      ▼
[Stage 3] Cosine Similarity Verification  ← Kaggle Signature dataset
           Compare cleaned signature vs reference
           Score ≥ 0.8 → MATCH (Genuine)
           Score < 0.8 → NO MATCH (Forged)
```

---

## 🧠 CNN Architecture

| Component | Details |
|---|---|
| Input | (1, 64, 128) — grayscale signature image |
| Block 1 | Conv2d(1→32) + BN + ReLU + MaxPool + Dropout |
| Block 2 | Conv2d(32→64) + BN + ReLU + MaxPool + Dropout |
| Block 3 | Conv2d(64→128) + BN + ReLU + AdaptiveAvgPool(4×4) |
| Feature Head | FC(2048→256) → ReLU → Dropout → FC(256→128) |
| Output | 128-D L2-normalised embedding (cosine similarity) |
| Classifier | FC(128→2) for detection task |
| Parameters | ~532,000 trainable |

---

## 📊 Key Results

| Metric | Value | Business Implication |
|---|---|---|
| Detection Accuracy | 90.8% | Reliable enough for assisted verification in banking workflows |
| Verification AUC | 1.0000 | Perfect genuine/forged separation on test set |
| Mean genuine pair score | ~0.85 | Consistent above threshold — low false rejection rate |
| Mean forged pair score | ~0.55 | Well below threshold — strong fraud detection |
| Verification threshold | 0.8 | Calibrated to minimise false acceptance of forged signatures |
| Score separation gap | ~0.30 | Large buffer reduces misclassification risk at decision boundary |

---

## 📊 Visualisations — 7 Charts

| Fig | Chart |
|---|---|
| fig1 | Genuine vs Forged signature samples |
| fig2 | CNN signature detection in document images |
| fig3 | Training loss and accuracy curves |
| fig4 | Confusion matrix |
| fig5 | Noise artifact removal pipeline (3-row comparison) |
| fig6 | Cosine similarity score distribution |
| fig7 | ROC curve + performance summary |

---

## What I'd Do With More Data

With the full Tobacco 800 dataset, I'd retrain on real scanned documents rather than synthetic data. 
With a larger verified signature corpus, I'd implement Siamese network training for more robust one-shot verification — the standard approach used in production banking systems.

---

## 💼 Applications

- **Banking:** Cheque and contract signature fraud prevention
- **Legal:** Document authentication and notarisation
- **Government:** Identity verification in official documents
- **HR:** Employment contract and onboarding document verification

---

## 🚀 How to Run

```bash
git clone https://github.com/Khushi-Dhargawe/Signature-Detection-Verification.git
cd Signature-Detection-Verification
pip install -r requirements.txt
jupyter notebook Signature_Detection_Verification.ipynb
```

> **For full training:** Download the [Tobacco 800 dataset](https://www.umiacs.umd.edu/~zhugy/tobacco800.html) and [Kaggle Signature dataset](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset), place in `data/` folder, and replace the synthetic data generation block (Section 2) with actual dataset loading.

---

## 📁 Related Projects

| # | Project | Skills |
|---|---|---|
| 7 | [PPP & ARIMA Forecasting](../PPP-ARIMA-Exchange-Rate-Forecasting) | Python · Statsmodels · ARIMA |
| **8** | **Signature Detection ← You are here** | **PyTorch · CNN · OpenCV** |
| 9 | [Healthify — Multi-Disease Prediction](../Healthify-Multi-Disease-Prediction) | Python · Streamlit · ResNet50 |

---

## 👩‍💻 Author

**Khushi Dhargawe**  
BE AI & ML (Hons. Cybersecurity) — A.P. Shah Institute of Technology, Mumbai University  
MSc Business Analytics — University College Cork (UCC)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/khushi-dhargawe/)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black?logo=github)](https://github.com/Khushi-Dhargawe)

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
