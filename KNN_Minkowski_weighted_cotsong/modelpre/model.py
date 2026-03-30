import numpy as np

def calculate_distance(X_train, x_test_point, p=2):
    # Tính khoảng cách Manhattan (p=1) hoặc Euclidean (p=2)
    return np.power(np.sum(np.power(np.abs(X_train - x_test_point), p), axis=1), 1/p)

def knn_core(X_train, y_train, x_test_point, k=3, p=2):
    distances = calculate_distance(X_train, x_test_point, p)
    
    # Tìm K láng giềng gần nhất
    k_indices = np.argsort(distances)[:k]
    k_distances = distances[k_indices]
    k_nearest_labels = y_train[k_indices]
    
    # --- BỘ LỌC 3: WEIGHTED KNN (Trọng số nghịch đảo khoảng cách) ---
    # Những điểm càng gần thì phiếu bầu càng nặng
    weights = 1 / (k_distances + 1e-5)
    
    unique_labels = np.unique(k_nearest_labels)
    label_weights = {}
    
    for label in unique_labels:
        # Tính tổng trọng số cho từng nhãn
        label_weights[label] = np.sum(weights[k_nearest_labels == label])
    
    # Trả về nhãn có tổng trọng số cao nhất (Majority Voting có trọng số)
    return max(label_weights, key=label_weights.get)