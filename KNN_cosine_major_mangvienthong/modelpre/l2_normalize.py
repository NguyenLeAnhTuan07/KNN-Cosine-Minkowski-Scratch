"""
scale.py
--------
Thuật toán Scale: L2 Normalization

    x_norm = x / ||x||_2

Đưa mỗi vector về độ dài = 1 (vector đơn vị).
Phù hợp với Cosine Distance vì Cosine chỉ quan tâm đến hướng vector.
"""

import numpy as np
import pandas as pd


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """
    L2 Normalization: chia mỗi vector cho độ dài Euclidean của nó.
        x_norm = x / ||x||_2
    Kết quả: ||x||_2 = 1 với mọi vector.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9  # tránh chia cho 0
    return X / norms
