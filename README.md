# UNIP – AI-Based Wafer Defect Detection (IESA Deep Tech Hackathon)

UNIP is a deep learning–based system designed to automatically detect and classify semiconductor wafer defects using grayscale image analysis. The solution is built for the **IESA Deep Tech Hackathon** and focuses on accuracy, explainability, and industrial relevance.

---

## 🔍 Problem Statement

Manual inspection of semiconductor wafers is:

* Time-consuming
* Error-prone
* Expensive at scale

Defects like cracks, contamination, or pattern irregularities can significantly reduce yield. UNIP automates this inspection using computer vision and deep learning.

---

## 🚀 Solution Overview

UNIP uses a **ResNet18 CNN model**, modified for grayscale wafer images, to classify multiple defect types. The pipeline includes:

* Automated training with Train / Validation / Test split
* Robust evaluation using accuracy and confusion matrix
* Explainability using **Grad-CAM** for defect localization

---

## 🧠 Model Architecture

* Backbone: **ResNet18**
* Input: Grayscale wafer images (1-channel)
* Image size: 256 × 256
* Output: Multi-class defect classification

---

## 📂 Project Structure

```
unip/
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── weights/
│   └── unip_resnet18.pth
│
├── train.py
├── evaluate.py
├── predict.py
├── gradcam.py
├── requirements.txt
└── README.md
```

> ⚠️ Note: `data/` and `weights/` are excluded from GitHub using `.gitignore`.

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

```bash
python train.py
```

* Uses AdamW optimizer
* CrossEntropyLoss with label smoothing
* Best model saved automatically

---

## 📊 Model Evaluation

```bash
python evaluate.py
```

Outputs:

* Test accuracy
* Classification report
* Confusion matrix

---

## 🔮 Prediction on New Images

```bash
python predict.py
```

Predicts defect class for a single wafer image.

---

## 🏭 Industrial Impact

* Reduces manual inspection effort
* Improves defect detection accuracy
* Scalable for high-volume semiconductor fabs
* Supports Industry 4.0 initiatives

---

## 🔮 Future Improvements

* Real-time inference on edge devices
* Transformer-based vision models
* Defect severity estimation
* Integration with fab inspection pipelines

---

## 👥 Team

**UNIP**
IESA Deep Tech Hackathon 2025

---

## 📜 License

This project is intended for academic and hackathon use.
