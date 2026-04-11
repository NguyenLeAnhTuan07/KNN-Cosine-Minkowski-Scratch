"""
predict.py
----------
Dự đoán nhãn cho dữ liệu trong dudoan.csv.

Luồng:
  1. Nếu chưa có scaler_config.pkl → gọi preprocessing để scale data train
  2. Gọi preprocessing để scale dudoan.csv
  3. Dự đoán từng dòng bằng knn_core (Cosine Distance)
"""

import os
import pickle
from modelpre.preprocessing import run_preprocessing
from modelpre.model import knn_core

TRAIN_CSV   = 'data/data.csv'
FEATURE_TXT = 'data/feature_names.txt'
PREDICT_CSV = 'predict/dudoan.csv'
SCALE_DIR   = 'scale'
K           = 5


def main():
    # Bước 1: Nếu chưa scale train thì scale trước
    if not os.path.exists(os.path.join(SCALE_DIR, "scaler_config.pkl")):
        print("[predict] Chưa có scale, đang scale dữ liệu train...")
        run_preprocessing(TRAIN_CSV, FEATURE_TXT, training=True, scale_dir=SCALE_DIR)

    # Bước 2: Scale dudoan.csv và lấy dữ liệu
    if not os.path.exists(PREDICT_CSV):
        print(f"Lỗi: Không tìm thấy file '{PREDICT_CSV}'")
        return

    X_test_scaled, params = run_preprocessing(
        PREDICT_CSV, FEATURE_TXT, training=False, scale_dir=SCALE_DIR
    )
    X_train = params['X_train_scaled']
    y_train = params['y_train']

    # Bước 3: Dự đoán
    print(f"\nKết quả dự đoán (K={K}):")
    print("-" * 30)
    for i, row in enumerate(X_test_scaled):
        result = knn_core(X_train, y_train, row, k=K)
        print(f"({result})")


if __name__ == "__main__":
    main()