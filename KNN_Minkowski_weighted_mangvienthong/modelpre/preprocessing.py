import pandas as pd
import numpy as np
from encoding.encoding import encode_categorical
from modelpre.robust_clipping import fit_scaler, transform_scaler, save_scaled_csv, is_fitted


def run_preprocessing(csv_path, feature_path, training=True,
                      scaled_output_path=None):
    """
    Đọc CSV, encode, rồi scale qua scale.py.

    - training=True  : fit scaler trên data, lưu params + CSV scaled.
    - training=False : load params đã fit, transform data mới + CSV scaled.

    scaled_output_path: nếu truyền vào thì lưu file CSV đã scale tại đường dẫn đó.
    """
    with open(feature_path, 'r', encoding='utf-8') as f:
        all_names = [line.strip() for line in f.readlines() if line.strip()]

    feature_names = all_names[:-1]
    label_name    = all_names[-1]

    df = pd.read_csv(csv_path)
    df = encode_categorical(df)

    if training:
        X = df[feature_names].values.astype(float)
        y = df[label_name].values

        X_scaled, params = fit_scaler(X, feature_names, y=y)

        # Lưu CSV training đã scale (mặc định nếu không chỉ định)
        out_path = scaled_output_path or 'scale/data_scaled.csv'
        save_scaled_csv(X_scaled, feature_names, out_path,
                        y=y, label_name=label_name)
        return params

    else:
        X_new = df[feature_names].values.astype(float)

        X_new_scaled, params = transform_scaler(X_new)

        # Lưu CSV predict đã scale (nếu có chỉ định)
        if scaled_output_path:
            save_scaled_csv(X_new_scaled, feature_names, scaled_output_path)

        return X_new_scaled, params