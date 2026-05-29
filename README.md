# CinnamonVision 

> **Automated grading and quality estimation of cinnamon quills using geometric and color texture analysis**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green?logo=opencv)](https://opencv.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

CinnamonVision is a computer vision project developed as part of the **EE7204 / EC7205 — Image Processing and Computer Vision** mini project at the **Department of Electrical and Information Engineering, University of Ruhuna**.

Ceylon cinnamon (Sri Lanka's signature export) is currently graded manually by experienced collectors, which introduces subjectivity and inconsistency. This project builds a fully automated, explainable grading pipeline that classifies cinnamon quills into four commercial grades — **Alba, C4, C5, and C5 Special** — using only image processing and classical machine learning, with no end-to-end deep learning.

> More than **75% of the pipeline** is computer vision — meeting the course requirement of ≥65%.

---

## Group Members

| Name | Index Number |
|------|-------------|
| Disanayaka D.M.S.P | EG/2021/4488 |
| Rasanjali A.J.U. | EG/2021/4747 |
| Rashmi E.D.K | EG/2021/4748 |
| Wijegurusinghe O.W.D.K.M. | EG/2021/4890 |

**Course:** EE7204 / EC7205 — Computer Vision and Image Processing  
**Department:** Electrical and Information Engineering, University of Ruhuna  
**Date:** January 2026

---

## Project Structure

```
CinnamonVision/
│
├── main.py              ← Run this to execute the full pipeline
├── preprocess.py        ← Image resizing, color conversion, noise reduction
├── segment.py           ← Otsu thresholding, contour detection, masking
├── features.py          ← Geometric, color (HSV), and texture (GLCM + LBP) features
├── classify.py          ← Dataset builder, KNN and SVM classifiers
├── evaluate.py          ← Confusion matrix, classification report, visualizations
│
├── data/                ← ⚠️ Not included (see Dataset Setup below)
│   ├── Alba/
│   ├── C4/
│   ├── C5/
│   └── C5_Special/
│
├── outputs/             ← Generated automatically when you run the project
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── sample_predictions.png
│   ├── features.csv
│   └── model.pkl
│
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Input Images
     │
     ▼
1. Image Acquisition     — Load from Kaggle dataset (Alba, C4, C5, C5 Special)
     │
     ▼
2. Preprocessing         — Resize to 256×256, RGB→Grayscale + HSV, Gaussian blur,
                           histogram equalization
     │
     ▼
3. Segmentation          — Otsu thresholding, Canny edge detection,
                           morphological cleanup, largest contour extraction
     │
     ▼
4. Feature Extraction
   ├── Geometric (7)     — Area, perimeter, aspect ratio, extent,
   │                       circularity, solidity, width
   ├── Color (12)        — Mean, std, min, max of H, S, V channels
   └── Texture (14)      — GLCM: contrast, homogeneity, energy, correlation
                           LBP: 10-bin histogram
     │
     ▼
5. Feature Vector        — Concatenate + StandardScaler normalization (33 features)
     │
     ▼
6. Classification        — KNN (k=5) and SVM (RBF kernel) with 5-fold cross-validation
     │
     ▼
7. Evaluation            — Accuracy, precision, recall, F1, confusion matrix
```

---

## Dataset Setup

This project uses the **Cinnamon Quill Grades Classification Image Dataset** from Kaggle.

**Download link:** https://www.kaggle.com/datasets/kaveendahelitha40/cinnamon-quill-grades-classification-image-dataset

### Dataset statistics

| Grade | Images |
|-------|--------|
| Alba | 173 |
| C4 | 215 |
| C5 | 199 |
| C5 Special | 334 |
| **Total** | **921** |

### After downloading

1. Extract the ZIP file
2. Place the folders inside the `data/` directory so your structure matches:

```
data/
  Alba/       ← 173 images
  C4/         ← 215 images
  C5/         ← 199 images
  C5_Special/ ← 334 images
```

> ⚠️ The `data/` folder is excluded from this repository via `.gitignore` due to file size. You must download the dataset separately.

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone this repository
git clone https://github.com/YOUR_USERNAME/CinnamonVision.git
cd CinnamonVision

# 2. (Optional but recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Project

### Run the full pipeline (recommended)

```bash
python main.py
```

This will:
1. Build the feature dataset from all images
2. Train KNN and SVM classifiers
3. Run 5-fold cross-validation
4. Save the best model to `outputs/model.pkl`
5. Generate confusion matrix, feature importance chart, and sample prediction images

All results are saved inside the `outputs/` folder.

---

### Run individual modules

```bash
# Test preprocessing on a single image
python preprocess.py

# Test segmentation on a single image
python segment.py

# Train and evaluate classifiers only
python classify.py

# Generate all evaluation plots
python evaluate.py
```

> Before running `preprocess.py` or `segment.py` standalone, edit the `image_path` at the bottom of each file to point to an actual image in your `data/` folder.

---

## Expected Results

After running `main.py`, check your `outputs/` folder for:

| Output file | Description |
|-------------|-------------|
| `confusion_matrix.png` | Per-class prediction accuracy heatmap |
| `feature_importance.png` | Top 15 most discriminative features by variance |
| `sample_predictions.png` | 8 sample images with true vs predicted grade |
| `features.csv` | Full extracted feature matrix (33 features × 921 samples) |
| `model.pkl` | Saved trained classifier |

**Target accuracy (from proposal):**
- Overall: ≥ 90%
- Diameter measurement: ≥ 95%
- Foxing/color detection: ≥ 90%
- Texture: ≥ 85%

---

## Feature Description

### Geometric features (7)

Extracted from contour detection on the binary mask:

| Feature | Description |
|---------|-------------|
| Area | Total pixel area of the quill |
| Perimeter | Length of the quill boundary |
| Aspect ratio | Width / height of bounding box |
| Extent | Contour area / bounding box area |
| Circularity | 4π × area / perimeter² |
| Solidity | Contour area / convex hull area |
| Width | Pixel width of bounding box |

### Color features (12)

Extracted from HSV color space on the masked region:

- Mean, standard deviation, min, and max of **Hue**, **Saturation**, and **Value** channels
- Captures discoloration, foxing, and surface unevenness

### Texture features (14)

- **GLCM** (Gray Level Co-occurrence Matrix): contrast, homogeneity, energy, correlation
- **LBP** (Local Binary Patterns): 10-bin histogram capturing surface microstructure

---

## Troubleshooting

**`ModuleNotFoundError`**  
Run `pip install -r requirements.txt` again. Make sure your virtual environment is activated.

**`FileNotFoundError` on image path**  
Check that your `data/` folder structure exactly matches the layout above. Folder names are case-sensitive.

**Memory error during feature extraction**  
Reduce image resize resolution in `preprocess.py`:  
Change `cv2.resize(img, (256, 256))` → `cv2.resize(img, (128, 128))`

**Low accuracy (below 60%)**  
Try increasing SVM regularization in `classify.py`:  
Change `SVC(kernel='rbf', C=10)` → `SVC(kernel='rbf', C=100)`

**Segmentation failing (no contour found)**  
Some images have complex backgrounds. The pipeline uses the full image as fallback when no valid contour is detected.

---

## References

1. Sandamali, H.A.C. & Liyanage, C.R. (2021). An image processing approach to detect the quality of cinnamon sticks. *ICAPS 2021*.
2. Samarajeewa et al. (2021). Machine learning-based quality detection of cinnamon from outer bark images.
3. Pulasinghe et al. (2023). AgroX: Uplift Ceylon Cinnamon Industry. *IRJIET*, vol. 7, pp. 27–34.
4. Wickramaarachchi et al. (2023). Data-driven solutions for cinnamon agriculturists. *SLIIT*.
5. Kebapci, H., Yanikoglu, B., & Unal, G. (2010). Plant image retrieval using color, shape and texture features. *The Computer Journal*, 53(9).


---

## 🏫 Academic Context

**Course:** EE7204 / EC7205 — Computer Vision and Image Processing  
**Institution:** Department of Electrical and Information Engineering, Faculty of Engineering, University of Ruhuna  
**Submitted:** January 2026