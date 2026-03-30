import pandas as pd
import numpy as np
import os
from modelpre.preprocessing import l2_normalize, encode_categorical
from modelpre.model import knn_core # Đảm bảo model.py của bạn đang dùng Cosine

def run_cosine_validation(file_path, feature_names_path, k_folds=5):
    # 1. Đọc và tiền xử lý dữ liệu
    df = pd.read_csv(file_path)
    df = encode_categorical(df)
    
    with open(feature_names_path, 'r') as f:
        features = [line.strip() for line in f.readlines() if line.strip()]
    
    X = df[features[:-1]].values.astype(float)
    y = df[features[-1]].values
    
    # 2. L2 Normalize (Quan trọng cho Cosine)
    X_norm = l2_normalize(X)
    
    # 3. K-Fold Cross Validation
    indices = np.arange(len(X_norm))
    np.random.seed(42) # Cố định để kết quả có thể tái lập
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)
    
    print(f"{'='*50}")
    print(f"VALIDATION: COSINE + L2 NORM ({k_folds}-Fold)")
    print(f"{'='*50}")
    
    max_k = int(np.sqrt(len(X_norm)))
    best_k = 1
    best_acc = 0

    for k in range(1, max_k + 1, 2):
        fold_accs = []
        for i in range(k_folds):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
            
            # Chia train/test cho fold hiện tại
            X_train_fold, X_test_fold = X_norm[train_idx], X_norm[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            correct = 0
            for idx, x_test in enumerate(X_test_fold):
                pred = knn_core(X_train_fold, y_train_fold, x_test, k=k)
                if pred == y_test_fold[idx]:
                    correct += 1
            fold_accs.append(correct / len(y_test_fold))
            
        avg_acc = np.mean(fold_accs)
        print(f"K = {k:2d} | Accuracy: {avg_acc:.4f}")
        
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_k = k

    print(f"{'='*50}")
    print(f"KẾT QUẢ: K tốt nhất là {best_k} với Accuracy: {best_acc:.4f}")

# Chạy chương trình
if __name__ == "__main__":
    run_cosine_validation('data/data.csv', 'data/feature_names.txt')