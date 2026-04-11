import os
from modelpre.preprocessing import run_preprocessing
from modelpre.model import knn_core
from modelpre.robust_clipping import is_fitted

def main():
    train_file   = 'data/data.csv'
    feature_txt  = 'data/feature_names.txt'
    predict_file = 'predict/dudoan.csv'

    # 1. Fit scaler nếu chưa có (đồng thời tạo data_scaled.csv)
    if not is_fitted():
        print("[predict] Chưa có scaler, đang fit trên data training...")
        run_preprocessing(train_file, feature_txt, training=True,
                          scaled_output_path='scale/data_scaled.csv')

    # 2. Dự đoán từ dudoan.csv
    if not os.path.exists(predict_file):
        print(f"Lỗi: Không tìm thấy file {predict_file}")
        return

    X_test_scaled, params = run_preprocessing(
        predict_file, feature_txt,
        training=False,
        scaled_output_path='scale/dudoan_scaled.csv'   # lưu file predict đã scale
    )

    X_train = params['X_train_scaled']
    y_train = params['y_train']

    for i, row in enumerate(X_test_scaled):
        res = knn_core(X_train, y_train, row)   # mặc định K=3, p=2
        print(f"({res})")


if __name__ == "__main__":
    main()