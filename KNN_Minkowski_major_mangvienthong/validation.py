import numpy as np
import pandas as pd
from encoding.encoding import encode_categorical
from modelpre.robust_clipping import fit_scaler, is_fitted, load_params, save_scaled_csv


# Hàm tính khoảng cách Minkowski phục vụ Validation
def minkowski_dist(X_train, x_test, p):
    return np.power(
        np.sum(np.power(np.abs(X_train - x_test), p), axis=1),
        1 / p
    )


def run_minkowski_validation(file_path, feature_names_path, p_val=2, k_folds=5):
    df = pd.read_csv(file_path)
    df = encode_categorical(df)

    with open(feature_names_path, 'r') as f:
        features = [line.strip() for line in f.readlines() if line.strip()]

    feature_names = features[:-1]
    label_name    = features[-1]

    X = df[feature_names].values.astype(float)
    y = df[label_name].values

    # --- Scale toàn bộ data qua scale.py ---
    # Fit mới (không ghi đè scaler_config đang dùng bởi predict) vì validation
    # cần kiểm soát từng fold nên tự scale inline theo cùng logic.
    # Nếu scaler chưa tồn tại, fit và lưu luôn để predict có thể dùng sau.
    if not is_fitted():
        print("[validation] Chưa có scaler, đang fit và lưu...")
        X_scaled, _ = fit_scaler(X, feature_names, y=y)
        save_scaled_csv(X_scaled, feature_names,
                        'scale/data_scaled.csv', y=y, label_name=label_name)
    else:
        # Dùng params đã lưu để scale nhất quán với predict
        params = load_params()
        X_clipped = np.clip(X, params['lower_perc'], params['upper_perc'])
        X_scaled  = (X_clipped - params['median']) / params['iqr']
        print("[validation] Đã dùng scaler hiện có để scale data.")

    # --- K-Fold Cross Validation ---
    indices = np.arange(len(X_scaled))
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)

    dist_name = "EUCLIDEAN" if p_val == 2 else "MANHATTAN"
    print(f"{'='*50}")
    print(f"VALIDATION: {dist_name} + ROBUST SCALE")
    print(f"{'='*50}")

    max_k = int(np.sqrt(len(X_scaled)))
    results = []

    for k in range(1, max_k + 1, 2):
        fold_accs = []
        for i in range(k_folds):
            test_idx  = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            correct = 0
            for idx, x_te_point in enumerate(X_te):
                dists   = minkowski_dist(X_tr, x_te_point, p_val)
                k_idx   = np.argsort(dists)[:k]
                labels  = y_tr[k_idx]
                vals, counts = np.unique(labels, return_counts=True)
                pred    = vals[np.argmax(counts)]
                if pred == y_te[idx]:
                    correct += 1
            fold_accs.append(correct / len(y_te))

        avg_acc = np.mean(fold_accs)
        print(f"K = {k:2d} | Accuracy: {avg_acc:.4f}")
        results.append((k, avg_acc))

    best = max(results, key=lambda x: x[1])
    print(f"{'='*50}")
    print(f"==> K tốt nhất: {best[0]} với Acc: {best[1]:.4f}")


if __name__ == "__main__":
    # Để kiểm tra Manhattan, đổi p_val=1
    run_minkowski_validation('data/data.csv', 'data/feature_names.txt', p_val=2)