import numpy as np

def cosine_distance_core(X_train, x_test_point):
    """
    Tính khoảng cách Cosine giữa một điểm test và toàn bộ tập train
    D_c = 1 - (A.B / (||A||*||B||))
    """
    # 1. Tính tích vô hướng
    dot_product = np.dot(X_train, x_test_point)

    # 2. Tính độ dài Vector
    norm_train = np.linalg.norm(X_train, axis=1)
    norm_test = np.linalg.norm(x_test_point)

    # Tránh lỗi chia cho 0 nếu có vector rỗng
    norm_train[norm_train == 0] = 1e-9
    if norm_test == 0: norm_test = 1e-9

    # 3. Tính độ tương đồng
    similarity = dot_product / (norm_train * norm_test)

    # 4. Trả về khoảng cách Dc
    return 1 - similarity

def knn_core(X_train, y_train, x_test_point, k=1):
    distances = cosine_distance_core(X_train, x_test_point)

    # Tìm K láng giềng có Dc nhỏ nhất
    k_indices = np.argsort(distances)[:k]
    k_distances = distances[k_indices]
    k_nearest_labels = y_train[k_indices]

    # --- WEIGHTED KNN (Trọng số nghịch đảo khoảng cách) ---
    weights = 1 / (k_distances + 1e-5)

    unique_labels = np.unique(k_nearest_labels)
    label_weights = {}

    for label in unique_labels:
        label_weights[label] = np.sum(weights[k_nearest_labels == label])

    return max(label_weights, key=label_weights.get)