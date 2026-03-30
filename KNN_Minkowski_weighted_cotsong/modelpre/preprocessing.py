import pandas as pd
import numpy as np
import pickle
import os
from encoding.encoding import encode_categorical

def run_preprocessing(csv_path, feature_path, training=True):
    if not os.path.exists('scale'):
        os.makedirs('scale')

    with open(feature_path, 'r', encoding='utf-8') as f:
        all_names = [line.strip() for line in f.readlines() if line.strip()]
    
    # BƯỚC QUAN TRỌNG: Phải đọc file trước để có biến 'df'
    df = pd.read_csv(csv_path)
    df = encode_categorical(df)

    if training:
        X = df[all_names[:-1]].values.astype(float)
        y = df[all_names[-1]].values
        
        # --- BỘ LỌC 1: CLIPPING (Chặn ngưỡng để khử Outliers cực đại) ---
        lower_perc = np.percentile(X, 1, axis=0)
        upper_perc = np.percentile(X, 99, axis=0)
        X = np.clip(X, lower_perc, upper_perc)

        # --- BỘ LỌC 2: ROBUST SCALING (Dùng Median và IQR thay cho Mean/Std) ---
        median = np.median(X, axis=0)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        iqr[iqr == 0] = 1 # Tránh chia cho 0
        
        X_scaled = (X - median) / iqr
        
        params = {
            'feature_names': all_names[:-1],
            'median': median,
            'iqr': iqr,
            'lower_perc': lower_perc,
            'upper_perc': upper_perc,
            'X_train_scaled': X_scaled,
            'y_train': y
        }
        with open('scale/scaler_config.pkl', 'wb') as f:
            pickle.dump(params, f)
        return params
    
    else:
        with open('scale/scaler_config.pkl', 'rb') as f:
            params = pickle.load(f)
            
        X_new = df[params['feature_names']].values.astype(float)
        
        # Áp dụng Clipping và Robust Scale cũ cho dữ liệu dự đoán
        X_new = np.clip(X_new, params['lower_perc'], params['upper_perc'])
        X_new_scaled = (X_new - params['median']) / params['iqr']
        return X_new_scaled, params