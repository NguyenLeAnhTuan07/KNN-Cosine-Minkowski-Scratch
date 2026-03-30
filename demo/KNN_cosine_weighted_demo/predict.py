import pandas as pd
import os
import pickle
from modelpre.preprocessing import run_preprocessing
from modelpre.model import knn_core

def main():
    # Khai báo file
    train_file = 'data/data.csv'
    feature_txt = 'data/feature_names.txt'
    predict_file = 'dudoan.csv'
    
    # 1. Tạo bộ Scale nếu chưa có
    if not os.path.exists('scale/scaler_config.pkl'):
        run_preprocessing(train_file, feature_txt, training=True)

    # 2. Dự đoán dữ liệu từ file dudoan.csv
    if os.path.exists(predict_file):
        X_test_scaled, params = run_preprocessing(predict_file, feature_txt, training=False)
        
        X_train = params['X_train_scaled']
        y_train = params['y_train']
        

        for i, row in enumerate(X_test_scaled):

            res = knn_core(X_train, y_train, row)
            
            print(f"({res})")
    else:
        print(f"Lỗi: Không tìm thấy file {predict_file}")

if __name__ == "__main__":
    main()