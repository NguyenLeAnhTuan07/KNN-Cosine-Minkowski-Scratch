# 🤖 K-Nearest Neighbors (KNN) — From Theory to Practice

> **ye-PHD Lab** — A comprehensive research document covering the KNN algorithm from mathematical foundations to hands-on experimentation.

---

## 📌 Overview

**K-Nearest Neighbors (KNN)** is a supervised learning and non-parametric algorithm. It operates on a simple principle: **similar data points tend to cluster together in feature space**.

KNN supports two types of tasks:
- 🏷️ **Classification** — assigns labels via majority voting among neighbors
- 📈 **Regression** — predicts continuous values by averaging neighbor outputs

---

## 📂 Table of Contents

| Chapter | Topic |
|---------|-------|
| 1 | What is the KNN Algorithm? |
| 2 | Feature Scaling |
| 3 | Distance Metrics in KNN |
| 4 | The `p` Parameter in Minkowski Distance |
| 5 | Choosing the Parameter `K` |
| 6 | Mathematical Model: KNN + Cosine Similarity |
| 7 | KNN Regression |
| 8 | KNN Workflow |
| 9 | Experiments on a 30-Sample Dataset |

---

## 🔧 Feature Scaling

Because KNN relies entirely on distance calculations, **feature scaling is mandatory** before running the model. Any feature with a larger value range will dominate the others.

### Robust Scaling
Uses **Median** and **Interquartile Range (IQR)** — both resistant to outliers — instead of mean and standard deviation:

$$x_{\text{scaled}} = \frac{x - \text{Median}(x)}{\text{IQR}(x)}$$

### 2-Layer Clipping (Winsorization)
Caps extreme values at safe boundaries without removing samples:
- **Lower Bound:** $x_{\min} = Q1 - 1.5 \times \text{IQR}$
- **Upper Bound:** $x_{\max} = Q3 + 1.5 \times \text{IQR}$

### L2 Normalization (Unit Length Scaling)
Used for Cosine Similarity tasks — transforms each sample vector to unit length:

$$x_{\text{unit}} = \frac{x}{\sqrt{\sum_{i=1}^{n} x_i^2}}$$

---

## 📐 Distance Metrics

| Metric | Formula | Best Used When |
|--------|---------|----------------|
| **Euclidean** | $\sqrt{\sum(x_i - y_i)^2}$ | General-purpose, data is normalized |
| **Manhattan** | $\sum\|x_i - y_i\|$ | Features are independent (e.g., Age vs. Income) |
| **Chebyshev** | $\max(\|x_i - y_i\|)$ | Only the largest deviation matters |
| **Minkowski** | $\left(\sum\|x_i - y_i\|^p\right)^{1/p}$ | Generalization of all three above |

> 💡 **Note:** `p=1` → Manhattan, `p=2` → Euclidean, `p→∞` → Chebyshev

---

## ⚙️ Choosing the Parameter K

### k-Fold Cross Validation
A 5-step process for finding the optimal `K` objectively:

1. **Initialize** candidate K values (prefer odd numbers: `{1, 3, 5, 7, 9, 11}`)
2. **Split** the dataset into `k` folds of equal size
3. **Cross-evaluate** each candidate K across all folds
4. **Select** the K with the highest `CV_Accuracy`
5. **Retrain** the final model on the full dataset

$$\text{CV\_Accuracy} = \frac{1}{k}\sum_{i=1}^{k} \text{Accuracy}_i$$

> **Convention:** `K` (uppercase) = number of neighbors; `k` (lowercase) = number of folds (typically 5 or 10).

### Distance Weighting
Closer neighbors are given higher influence during voting:

$$w_i = \frac{1}{d(x, x_i) + \varepsilon}$$

where $\varepsilon$ is a small constant to avoid division by zero.

### Weighted Majority Vote
The predicted class is the one with the highest total weight among K neighbors:

$$\hat{y}_q = \arg\max_{c \in C} \sum_{z_j \in N_K(z_q)} \mathbf{1}[y_j = c]$$

---

## 📐 Cosine Similarity

Measures the directional similarity between two vectors via the angle $\theta$:

$$S_C(z_q, z_i) = \cos\theta = \frac{z_q \cdot z_i}{\|z_q\| \cdot \|z_i\|}$$

Converted to a distance metric (smaller = more similar):

$$D_C(z_q, z_i) = 1 - S_C(z_q, z_i), \quad D_C \in [0, 2]$$

---

## 📊 KNN Regression

Instead of voting, KNN Regression averages the target values of K neighbors:

$$\hat{y} = \frac{1}{K}\sum_{i=1}^{K} y_i$$

With distance weighting:

$$\hat{y} = \frac{\sum_{i=1}^{K} w_i y_i}{\sum_{i=1}^{K} w_i}$$

### KNN Regression vs. Linear Regression

| Criterion | KNN Regression | Linear Regression |
|-----------|---------------|-------------------|
| Training time | ✅ Zero | ❌ Requires optimization |
| Prediction time | ❌ Slow (scans full dataset) | ✅ Near-instant |
| Non-linear patterns | ✅ Handles well | ❌ Constrained to linear |
| Extrapolation | ❌ Cannot extrapolate | ✅ Can extend beyond training range |
| Large-scale data | ❌ Expensive | ✅ Efficient |

**Rule of thumb:** Simple, linear data → **Linear Regression**. Complex, non-linear, small-scale data → **KNN Regression**.

---

## 🔄 KNN Workflow

```
1. Choose K
      ↓
2. Compute distances from the query point to all training points
      ↓
3. Sort distances and select the K nearest neighbors
      ↓
4. Make prediction
   ├── Classification → Weighted Majority Vote
   └── Regression    → Weighted Average of neighbor values
```

---

## 🧪 Experiments

A dataset of **30 customer records** with 3 features — `Age`, `Income`, `Tenure` — and 4 class labels (`Custcat`). A test set of 4 samples was used for prediction.

### Experiment 1 — Euclidean Distance + 5-Fold Cross Validation

**Robust Scaling parameters:**
- $\text{Age}_{\text{scaled}} = \frac{\text{Age} - 5.5}{5}$
- $\text{Income}_{\text{scaled}} = \frac{\text{Income} - 5}{4.25}$
- $\text{Tenure}_{\text{scaled}} = \frac{\text{Tenure} - 3}{2}$

**Cross Validation Results:**

| K | Correct Predictions | Accuracy |
|---|---------------------|----------|
| K=1 | 26/30 | 86.67% |
| K=3 | 27/30 | **90.00%** ✅ |
| K=5 | 26/30 | 86.67% |

**Test Set Predictions (K=3, Weighted Euclidean):**

| Test Sample | Predicted Class |
|-------------|-----------------|
| (Age=3, Income=4, Tenure=2) | Class 1 |
| (Age=5, Income=5, Tenure=3) | Class 2 |
| (Age=7, Income=8, Tenure=4) | Class 3 |
| (Age=9, Income=9, Tenure=5) | Class 4 |

> 🏆 **K=3 yields the best accuracy** with Euclidean distance.

---

### Experiment 2 — Cosine Similarity + 5-Fold Cross Validation

Data was L2-normalized before computing Cosine similarity.

**Cross Validation Results:**

| K | Correct Predictions | Accuracy |
|---|---------------------|----------|
| K=1 | 3/30 | 10.00% |
| K=3 | 2/30 | 6.67% |
| K=5 | 2/30 | 6.67% |

**Test Set Predictions (K=3, Cosine Similarity):**

| Test Sample | Predicted Class |
|-------------|-----------------|
| (Age=3, Income=4, Tenure=2) | Class 1 |
| (Age=5, Income=5, Tenure=3) | Class 2 |
| (Age=7, Income=8, Tenure=4) | Class 3 |
| (Age=9, Income=9, Tenure=5) | Class 4 |

> ⚠️ **Cosine Similarity performed poorly on this dataset.** It measures directional similarity, which is more meaningful for text/NLP data than for tabular numeric features. However, the final label predictions still matched expectations due to the strong angular separation between classes.

---

## 📚 References

- VinBigData — KNN Algorithm Illustration
- Wikipedia — Euclidean Distance
- Viblo — Manhattan & Chebyshev Distance

---

## 👨‍🔬 Author

**ye-PHD Lab**

---

*Made with ❤️ for learning Machine Learning from scratch.*
