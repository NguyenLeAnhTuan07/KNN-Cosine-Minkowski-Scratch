import pandas as pd
import numpy as np
import pickle
import os
from encoding.encoding import encode_categorical

def l2_normalize(X):
    # Tính độ dài (norm) của từng vector (mỗi dòng)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Tránh chia cho 0 nếu có vector rỗng
    norms[norms == 0] = 1e-9
    # Chia từng vector cho độ dài của nó → vector đơn vị (length = 1)
    return X / norms

def run_preprocessing(csv_path, feature_path, training=True):
    if not os.path.exists('scale'):
        os.makedirs('scale')

    with open(feature_path, 'r', encoding='utf-8') as f:
        all_names = [line.strip() for line in f.readlines() if line.strip()]

    df = pd.read_csv(csv_path)
    df = encode_categorical(df)

    if training:
        X = df[all_names[:-1]].values.astype(float)
        y = df[all_names[-1]].values

        # L2 Normalization: đưa mỗi vector về độ dài = 1
        X_normalized = l2_normalize(X)

        params = {
            'feature_names': all_names[:-1],
            'X_train_scaled': X_normalized,
            'y_train': y
        }
        with open('scale/scaler_config.pkl', 'wb') as f:
            pickle.dump(params, f)
        return params

    else:
        with open('scale/scaler_config.pkl', 'rb') as f:
            params = pickle.load(f)

        X_new = df[params['feature_names']].values.astype(float)

        # Áp dụng L2 Normalization cho dữ liệu mới
        X_new_normalized = l2_normalize(X_new)
        return X_new_normalized, params