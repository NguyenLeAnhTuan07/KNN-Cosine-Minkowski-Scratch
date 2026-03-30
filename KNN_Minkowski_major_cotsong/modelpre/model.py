import numpy as np

def calculate_distance(X_train, x_test_point, p=2):
    # Tính khoảng cách Manhattan (p=1) hoặc Euclidean (p=2)
    return np.power(np.sum(np.power(np.abs(X_train - x_test_point), p), axis=1), 1/p)

def knn_core(X_train, y_train, x_test_point, k=3, p=2):
    distances = calculate_distance(X_train, x_test_point, p)

    # Tìm K láng giềng gần nhất
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = y_train[k_indices]

    # Majority Voting: mỗi láng giềng có phiếu bầu bằng nhau
    values, counts = np.unique(k_nearest_labels, return_counts=True)
    return values[np.argmax(counts)]