import pandas as pd
import numpy as np
from modelpre.preprocessing import encode_categorical

# Hàm tính khoảng cách Minkowski nội bộ để phục vụ Validation
def minkowski_dist(X_train, x_test, p):
    return np.power(np.sum(np.power(np.abs(X_train - x_test), p), axis=1), 1/p)

def run_minkowski_validation(file_path, feature_names_path, p_val=2, k_folds=5):
    df = pd.read_csv(file_path)
    df = encode_categorical(df)
    
    with open(feature_names_path, 'r') as f:
        features = [line.strip() for line in f.readlines() if line.strip()]
    
    X = df[features[:-1]].values.astype(float)
    y = df[features[-1]].values

    # --- MÀNG LỌC 1: CLIPPING (Chặn 1% - 99%) ---
    lower = np.percentile(X, 1, axis=0)
    upper = np.percentile(X, 99, axis=0)
    X = np.clip(X, lower, upper)

    # --- MÀNG LỌC 2: ROBUST SCALING (Median & IQR) ---
    median = np.median(X, axis=0)
    q1, q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0] = 1
    X_scaled = (X - median) / iqr

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
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
            X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            correct = 0
            for idx, x_te_point in enumerate(X_te):
                # Tính khoảng cách
                dists = minkowski_dist(X_tr, x_te_point, p_val)
                k_idx = np.argsort(dists)[:k]
                # Majority Voting
                labels = y_tr[k_idx]
                vals, counts = np.unique(labels, return_counts=True)
                pred = vals[np.argmax(counts)]
                
                if pred == y_te[idx]: correct += 1
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