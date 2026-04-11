"""
validation.py
-------------
K-Fold Cross Validation cho KNN Cosine Distance.

Luồng:
  1. Nếu chưa có scaler_config.pkl → gọi preprocessing để scale data train
  2. Đọc data_scaled.csv để cross-validate
"""

import pandas as pd
import numpy as np
import os
from modelpre.preprocessing import run_preprocessing
from modelpre.model import knn_core

TRAIN_CSV   = 'data/data.csv'
FEATURE_TXT = 'data/feature_names.txt'
SCALE_DIR   = 'scale'
K_FOLDS     = 5


def run_cosine_validation(k_folds: int = K_FOLDS):
    # Bước 1: Nếu chưa scale thì scale trước
    if not os.path.exists(os.path.join(SCALE_DIR, "scaler_config.pkl")):
        print("[validation] Chưa có scale, đang scale dữ liệu train...")
        run_preprocessing(TRAIN_CSV, FEATURE_TXT, training=True, scale_dir=SCALE_DIR)

    # Bước 2: Đọc data_scaled.csv
    scaled_csv = os.path.join(SCALE_DIR, "data_scaled.csv")
    df = pd.read_csv(scaled_csv)

    with open(FEATURE_TXT, 'r') as f:
        all_names = [line.strip() for line in f if line.strip()]
    feature_names = all_names[:-1]
    label_name    = all_names[-1]

    X_norm = df[feature_names].values.astype(float)
    y      = df[label_name].values

    # Bước 3: K-Fold Cross Validation
    indices = np.arange(len(X_norm))
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)

    print(f"{'='*50}")
    print(f"VALIDATION: COSINE + L2 NORM ({k_folds}-Fold)")
    print(f"{'='*50}")

    max_k    = int(np.sqrt(len(X_norm)))
    best_k   = 1
    best_acc = 0.0

    for k in range(1, max_k + 1, 2):
        fold_accs = []
        for i in range(k_folds):
            test_idx  = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])

            X_train_fold, X_test_fold = X_norm[train_idx], X_norm[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            correct = sum(
                1 for idx, x_test in enumerate(X_test_fold)
                if knn_core(X_train_fold, y_train_fold, x_test, k=k) == y_test_fold[idx]
            )
            fold_accs.append(correct / len(y_test_fold))

        avg_acc = np.mean(fold_accs)
        print(f"K = {k:2d} | Accuracy: {avg_acc:.4f}")

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_k   = k

    print(f"{'='*50}")
    print(f"KẾT QUẢ: K tốt nhất là {best_k} với Accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    run_cosine_validation()