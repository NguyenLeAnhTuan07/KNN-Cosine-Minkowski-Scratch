# KNN Cosine & Minkowski Framework

A from-scratch implementation of the **K-Nearest Neighbors (KNN)** classifier supporting two distinct geometric approaches — Cosine Similarity and Minkowski Distance — with professional-grade preprocessing pipelines. No high-level machine learning libraries are used in the core algorithm.

**Author:** Nguyen Le Anh Tuan

---

## Overview

This project goes beyond a textbook KNN implementation by incorporating robust data-cleaning techniques and automated hyperparameter tuning. The result is a classifier that handles real-world data — including outliers, skewed distributions, and varying feature scales — without requiring external ML frameworks.

---

## Two Geometric Approaches

### 1. Cosine Similarity

This approach measures the **angle** between feature vectors rather than the distance between them, making it well-suited for cases where the pattern of the data matters more than its absolute magnitude.

**Preprocessing:** L2 Normalization — maps all samples onto a unit sphere so that only the direction of each vector is compared.

**Best used when:**
- Analyzing behavioral or usage patterns (e.g., telecom customer segmentation)
- Feature scales vary significantly across columns (e.g., income vs. age)
- The relative proportion of features is more meaningful than raw values

### 2. Minkowski Distance (Euclidean & Manhattan) with Triple-Filtering

This approach uses the Minkowski metric with configurable `p`, allowing a switch between **Euclidean** (`p=2`) and **Manhattan** (`p=1`) distance. A three-stage preprocessing pipeline is applied to ensure robustness on noisy, real-world data.

**Triple-Filtering pipeline:**
- **Clipping** — removes extreme outliers by capping values at the 1st and 99th percentiles
- **Robust Scaling** — normalizes features using the median and interquartile range (IQR), resistant to skewed distributions
- **Distance Weighting** — closer neighbors receive proportionally higher influence during voting via inverse-distance weights (`1/d`)

**Best used when:**
- Working with physical or clinical measurements (e.g., biomechanical orthopedic features)
- The dataset contains noise or significant outliers
- The physical distance between data points carries direct interpretive meaning

---

## Project Structure

```
├── data/
│   ├── data.csv              # Training dataset
│   └── feature_names.txt     # Feature column names (last entry = target label)
├── encoding/
│   └── encoding.py           # Categorical encoding for string labels
├── modelpre/
│   ├── model.py              # KNN core algorithm
│   └── preprocessing.py      # Scaling and normalization logic
├── dudoan.csv                # Input file for new predictions
├── predict.py                # Run predictions on dudoan.csv
├── validation.py             # K-Fold cross-validation to find optimal K
└── README.md
```

---

## How to Use

### 1. Prepare your dataset

- Place your dataset in CSV format into the `data/` directory and rename it `data.csv`
- Update `feature_names.txt` to match your column names, one per line
- The **last name** in the list must be the target label (the column to predict)

### 2. Encode categorical labels (if needed)

If your dataset contains string labels such as `"Normal"` or `"Abnormal"`, use `encoding.py` to map them to integer values (`0`, `1`, `2`, ...) before running the model.

### 3. Add samples to predict

Place the new samples you want to classify in `dudoan.csv`, using the same columns as `data.csv` but **without** the target label column.

---

## Installation & Execution

**Create a virtual environment (recommended)**

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**Install dependencies**

```bash
pip install numpy pandas
```

**Find the optimal K via cross-validation**

```bash
python validation.py
```

This runs K-Fold Cross Validation (5-Fold by default) across all odd values of K up to √N, printing the accuracy at each K and returning the best value. The Cosine version also tests even values of K, since distance weighting eliminates the risk of tied votes.

**Run predictions**

```bash
python predict.py
```

Classifies each row in `dudoan.csv` using the optimal K found above and prints the predicted class for each sample.

---

## Datasets

| Dataset | Description | Source |
|---|---|---|
| Telecom Customer Category | Multi-class customer segmentation | Included in repository |
| Orthopedic Patients | Biomechanical features for clinical classification | [Kaggle](https://www.kaggle.com/datasets/uciml/biomechanical-features-of-orthopedic-patients/data) |

---

## Key Design Decisions

**Why no sklearn for the core algorithm?**
The KNN logic — distance calculation, neighbor selection, and weighted voting — is implemented entirely with NumPy. This makes every step transparent and easy to trace, which is the primary goal of this project.

**Why two separate preprocessing pipelines?**
Cosine and Minkowski distance measure fundamentally different things. Cosine cares only about vector direction, so L2 Normalization is the correct and sufficient preprocessing step. Minkowski cares about magnitude, so outlier removal and robust scaling are necessary first. Mixing the pipelines would undermine the geometric guarantees of each metric.

**Why use distance weighting for Minkowski but not Cosine?**
After L2 Normalization, Cosine distances between neighboring points collapse to near zero, causing inverse-distance weights to become extremely large for the nearest neighbor and negligible for all others — effectively reducing the vote to K=1. Majority voting is therefore more stable for the Cosine variant.

Thank you for checking out this project. Have a great day! ☀️

Author: Nguyễn Lê Anh Tuấn, Nguyễn Đức Huy
