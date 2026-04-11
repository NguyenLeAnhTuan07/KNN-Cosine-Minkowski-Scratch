"""
preprocessing.py
----------------
Tiền xử lý dữ liệu: đọc CSV, encode, gọi scale.py để scale,
lưu kết quả ra file và trả về dữ liệu đã scale.
"""

import pandas as pd
import numpy as np
import pickle
import os
from encoding.encoding import encode_categorical
from modelpre.l2_normalize import l2_normalize


def run_preprocessing(csv_path: str, feature_path: str,
                      training: bool = True,
                      scale_dir: str = "scale") -> dict:
    """
    Đọc dữ liệu, gọi scale.py để scale, lưu file, trả về params.

    training=True  → scale data train, lưu data_scaled.csv + scaler_config.pkl
    training=False → scale data dự đoán, lưu dudoan_scaled.csv, trả về (X, params)
    """
    os.makedirs(scale_dir, exist_ok=True)

    with open(feature_path, 'r', encoding='utf-8') as f:
        all_names = [line.strip() for line in f if line.strip()]
    feature_names = all_names[:-1]
    label_name    = all_names[-1]

    df = pd.read_csv(csv_path)
    df = encode_categorical(df) 

    if training:
        X = df[feature_names].values.astype(float)
        y = df[label_name].values

        X_scaled = l2_normalize(X) 

        # Lưu CSV đã scale
        df_scaled = pd.DataFrame(X_scaled, columns=feature_names)
        df_scaled[label_name] = y
        df_scaled.to_csv(os.path.join(scale_dir, "data_scaled.csv"), index=False)

        # Lưu thông số
        params = {
            'feature_names':  feature_names,
            'label_name':     label_name,
            'X_train_scaled': X_scaled,
            'y_train':        y
        }
        with open(os.path.join(scale_dir, "scaler_config.pkl"), 'wb') as f:
            pickle.dump(params, f)

        return params

    else:
        pkl_path = os.path.join(scale_dir, "scaler_config.pkl")
        with open(pkl_path, 'rb') as f:
            params = pickle.load(f)

        X = df[params['feature_names']].values.astype(float)
        X_scaled = l2_normalize(X) 

        # Lưu CSV đã scale
        df_scaled = pd.DataFrame(X_scaled, columns=params['feature_names'])
        df_scaled.to_csv(os.path.join(scale_dir, "dudoan_scaled.csv"), index=False)

        return X_scaled, params