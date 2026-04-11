import numpy as np
import pickle
import os

SCALE_DIR = 'scale'
SCALER_PATH = os.path.join(SCALE_DIR, 'scaler_config.pkl')


def fit_scaler(X, feature_names, y=None):
    """
    Fit scaler trên tập training:
      - Bộ lọc 1: Clipping (chặn 1% - 99% để khử Outliers cực đại)
      - Bộ lọc 2: Robust Scaling (dùng Median và IQR thay cho Mean/Std)

    Lưu params vào scale/scaler_config.pkl.
    Trả về X đã được scale và dict params.
    """
    if not os.path.exists(SCALE_DIR):
        os.makedirs(SCALE_DIR)

    # --- Bộ lọc 1: Clipping ---
    lower_perc = np.percentile(X, 1, axis=0)
    upper_perc = np.percentile(X, 99, axis=0)
    X_clipped = np.clip(X, lower_perc, upper_perc)

    # --- Bộ lọc 2: Robust Scaling ---
    median = np.median(X_clipped, axis=0)
    q1 = np.percentile(X_clipped, 25, axis=0)
    q3 = np.percentile(X_clipped, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1  # tránh chia cho 0

    X_scaled = (X_clipped - median) / iqr

    params = {
        'feature_names': feature_names,
        'lower_perc':    lower_perc,
        'upper_perc':    upper_perc,
        'median':        median,
        'iqr':           iqr,
    }
    if y is not None:
        params['X_train_scaled'] = X_scaled
        params['y_train']        = y

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(params, f)

    return X_scaled, params


def transform_scaler(X):
    """
    Áp dụng params đã fit lên tập mới (predict / validation fold).
    Trả về X đã được scale và dict params.
    """
    with open(SCALER_PATH, 'rb') as f:
        params = pickle.load(f)

    X_clipped = np.clip(X, params['lower_perc'], params['upper_perc'])
    X_scaled  = (X_clipped - params['median']) / params['iqr']
    return X_scaled, params


def save_scaled_csv(X_scaled, feature_names, output_path, y=None, label_name=None):
    """
    Lưu dữ liệu đã scale thành file CSV.
    Nếu truyền y và label_name thì cột nhãn cũng được ghi vào cuối.
    """
    import pandas as pd
    df = pd.DataFrame(X_scaled, columns=feature_names)
    if y is not None and label_name is not None:
        df[label_name] = y
    df.to_csv(output_path, index=False)
    print(f"[scale] Đã lưu dữ liệu scaled -> {output_path}")


def is_fitted():
    """Kiểm tra xem scaler đã được fit chưa."""
    return os.path.exists(SCALER_PATH)


def load_params():
    """Tải params scaler đã lưu."""
    with open(SCALER_PATH, 'rb') as f:
        return pickle.load(f)