import numpy as np

def cosine_distance_core(X_train, x_test_point):
    """
    Tính khoảng cách Cosine giữa một điểm test và toàn bộ tập train
    D_c = 1 - (A.B / (||A||*||B||))
    """
    # 1. Tính tích vô hướng (Tử số công thức 9)
    # dot_product là một mảng chứa tích của x_test với từng dòng trong X_train
    dot_product = np.dot(X_train, x_test_point)
    
    # 2. Tính độ dài Vector (Mẫu số công thức 9)
    norm_train = np.linalg.norm(X_train, axis=1)
    norm_test = np.linalg.norm(x_test_point)
    
    # Tránh lỗi chia cho 0 nếu có vector rỗng
    norm_train[norm_train == 0] = 1e-9
    if norm_test == 0: norm_test = 1e-9
    
    # 3. Tính độ tương đồng Sc
    similarity = dot_product / (norm_train * norm_test)
    
    # 4. Trả về khoảng cách Dc (Công thức 10)
    return 1 - similarity

def knn_core(X_train, y_train, x_test_point, k=3):
    # Tính khoảng cách Cosine
    distances = cosine_distance_core(X_train, x_test_point)
    
    # Tìm K láng giềng có Dc nhỏ nhất
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = y_train[k_indices]
    
    # Bầu chọn đa số (Công thức 11)
    values, counts = np.unique(k_nearest_labels, return_counts=True)
    return values[np.argmax(counts)]