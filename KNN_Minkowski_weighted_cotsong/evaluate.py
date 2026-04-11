import pandas as pd
import numpy as np
import os
from modelpre.model import knn_core
from modelpre.robust_clipping import is_fitted, load_params
from modelpre.preprocessing import run_preprocessing


# ═══════════════════════════════════════════════════════════
#  HÀM DỰ ĐOÁN KNN CHO TỪNG FOLD
# ═══════════════════════════════════════════════════════════

def knn_predict_batch(X_train, y_train, X_test, k=3, p=2):
    """Dự đoán toàn bộ tập test bằng KNN."""
    return np.array([knn_core(X_train, y_train, row, k=k, p=p) for row in X_test])


# ═══════════════════════════════════════════════════════════
#  MA TRẬN NHẦM LẪN ĐA NHÃN
# ═══════════════════════════════════════════════════════════

def build_confusion_matrix(y_true, y_pred, classes):
    """
    Xây dựng ma trận nhầm lẫn (confusion matrix).
    Hàng = nhãn thực tế, Cột = nhãn dự đoán.
    """
    n = len(classes)
    idx_map = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx_map[t]][idx_map[p]] += 1
    return cm


def print_confusion_matrix(cm, classes):
    col_w = max(10, max(len(str(c)) for c in classes) + 2)
    header = f"{'':>{col_w}}" + "".join(f"{str(c):>{col_w}}" for c in classes)
    print(header)
    print("-" * len(header))
    for i, c in enumerate(classes):
        row = f"{str(c):>{col_w}}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(len(classes)))
        print(row)


# ═══════════════════════════════════════════════════════════
#  CHỈ SỐ ĐÁNH GIÁ TỪNG NHÃN (TP/TN/FP/FN per class)
# ═══════════════════════════════════════════════════════════

def per_class_metrics(cm, classes):
    """
    Tính Precision, Recall, F1 cho từng nhãn theo chiến lược One-vs-Rest.
    Trả về dict: {class_label: {precision, recall, f1, support}}
    """
    eps = 1e-8
    metrics = {}
    n = len(classes)
    for i, c in enumerate(classes):
        tp = cm[i][i]
        fp = np.sum(cm[:, i]) - tp   # cột i trừ diagonal
        fn = np.sum(cm[i, :]) - tp   # hàng i trừ diagonal
        tn = np.sum(cm) - tp - fp - fn

        precision = tp / (tp + fp + eps)
        recall    = tp / (tp + fn + eps)
        f1        = 2 * precision * recall / (precision + recall + eps)
        support   = int(np.sum(cm[i, :]))

        metrics[c] = {
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "Precision": precision, "Recall": recall, "F1": f1,
            "Support": support
        }
    return metrics


# ═══════════════════════════════════════════════════════════
#  AUC — 2 NHÃN
# ═══════════════════════════════════════════════════════════

def auc_binary(y_true, y_scores):
    """
    Tính AUC bằng phương pháp trapezoid thủ công cho bài toán 2 nhãn.
    y_scores: xác suất / điểm tin cậy (ở đây dùng khoảng cách nghịch đảo).
    """
    desc_idx   = np.argsort(y_scores)[::-1]
    y_sorted   = y_true[desc_idx]
    num_pos    = np.sum(y_true == 1)
    num_neg    = np.sum(y_true == 0)
    if num_pos == 0 or num_neg == 0:
        return 0.5

    tp_c, fp_c, tp_p, fp_p, auc = 0, 0, 0, 0, 0.0
    for label in y_sorted:
        if label == 1:
            tp_c += 1
        else:
            fp_c += 1
            auc  += (fp_c - fp_p) * (tp_c + tp_p) / 2
            fp_p, tp_p = fp_c, tp_c
    return auc / (num_pos * num_neg)


# ═══════════════════════════════════════════════════════════
#  TỔNG HỢP TOÀN BỘ MÔ HÌNH — 3 CÔNG THỨC ĐA NHÃN
# ═══════════════════════════════════════════════════════════

def aggregate_metrics(per_cls, method, classes):
    """
    Tổng hợp Precision, Recall, F1 theo method:
      - 'accuracy'  : Accuracy tổng thể (tỉ lệ dự đoán đúng)
      - 'macro'     : Trung bình vị mô (Macro-Average)
      - 'weighted'  : Trung bình có trọng số (Weighted-Average)
    """
    total_samples = sum(per_cls[c]["Support"] for c in classes)

    if method == "accuracy":
        tp_total = sum(per_cls[c]["TP"] for c in classes)
        return tp_total / total_samples, None, None

    prec_list = np.array([per_cls[c]["Precision"] for c in classes])
    rec_list  = np.array([per_cls[c]["Recall"]    for c in classes])
    f1_list   = np.array([per_cls[c]["F1"]        for c in classes])
    sup_list  = np.array([per_cls[c]["Support"]   for c in classes])

    if method == "macro":
        prec = np.mean(prec_list)
        rec  = np.mean(rec_list)
        f1   = np.mean(f1_list)
    else:  # weighted
        weights = sup_list / total_samples
        prec = np.sum(weights * prec_list)
        rec  = np.sum(weights * rec_list)
        f1   = np.sum(weights * f1_list)

    return prec, rec, f1


# ═══════════════════════════════════════════════════════════
#  K-FOLD CROSS VALIDATION KNN
# ═══════════════════════════════════════════════════════════

def run_k_fold_cv(X, y, classes, k_folds=5, k_knn=3, p=2, agg_method=None):
    """
    Chạy K-Fold CV cho KNN.
    agg_method: 'accuracy' | 'macro' | 'weighted' | None (2 nhãn)
    """
    m = X.shape[0]
    indices = np.arange(m)
    np.random.seed(42)
    np.random.shuffle(indices)

    fold_sizes = np.full(k_folds, m // k_folds)
    fold_sizes[: m % k_folds] += 1
    current = 0
    folds = []
    for size in fold_sizes:
        folds.append(indices[current: current + size])
        current += size

    is_multiclass = len(classes) > 2

    history = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "AUC": []}
    all_pred_details = []   # stt, y_true, y_pred

    print(f"\n{'='*65}")
    print(f"  K-FOLD CROSS VALIDATION — KNN (K={k_knn}, p={p}, Folds={k_folds})")
    print(f"{'='*65}")
    print(f"{'Fold':<8} | {'Acc':<8} | {'Prec':<8} | {'Recall':<8} | {'F1':<8} | {'AUC':<8}")
    print("-" * 65)

    for i in range(k_folds):
        test_idx  = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        y_pred = knn_predict_batch(X_tr, y_tr, X_te, k=k_knn, p=p)

        # Ma trận nhầm lẫn fold này
        cm_fold  = build_confusion_matrix(y_te, y_pred, classes)
        per_cls  = per_class_metrics(cm_fold, classes)

        # Accuracy
        acc = np.sum(np.diag(cm_fold)) / len(y_te)

        # Precision, Recall, F1
        if is_multiclass:
            prec, rec, f1 = aggregate_metrics(per_cls, agg_method, classes)
            if agg_method == "accuracy":
                prec = rec = f1 = acc   # accuracy mode chỉ có 1 chỉ số
        else:
            prec = per_cls[classes[1]]["Precision"]
            rec  = per_cls[classes[1]]["Recall"]
            f1   = per_cls[classes[1]]["F1"]

        # AUC
        if not is_multiclass:
            # 2 nhãn: dùng tỉ lệ phiếu bầu làm score (1-NN vote ratio)
            scores = np.array([
                np.mean(knn_core.__module__ and
                        y_tr[np.argsort(
                            np.power(np.sum(np.power(np.abs(X_tr - row), p), axis=1), 1/p)
                        )[:k_knn]] == classes[1])
                for row in X_te
            ], dtype=float)
            y_te_bin = (y_te == classes[1]).astype(int)
            auc = auc_binary(y_te_bin, scores)
        else:
            # Đa nhãn: AUC = trung bình AUC One-vs-Rest từng nhãn
            auc_list = []
            for cls in classes:
                y_bin = (y_te == cls).astype(int)
                scores = np.array([
                    np.mean(y_tr[np.argsort(
                        np.power(np.sum(np.power(np.abs(X_tr - row), p), axis=1), 1/p)
                    )[:k_knn]] == cls)
                    for row in X_te
                ], dtype=float)
                auc_list.append(auc_binary(y_bin, scores))
            auc = np.mean(auc_list)

        history["Accuracy"].append(acc)
        history["Precision"].append(prec)
        history["Recall"].append(rec)
        history["F1"].append(f1)
        history["AUC"].append(auc)

        print(f"Fold {i+1:<3} | {acc:<8.4f} | {prec:<8.4f} | {rec:<8.4f} | {f1:<8.4f} | {auc:<8.4f}")

        for j, idx in enumerate(test_idx):
            all_pred_details.append({"stt": idx, "y_true": y_te[j], "y_pred": y_pred[j]})

    return history, all_pred_details


# ═══════════════════════════════════════════════════════════
#  LƯU KẾT QUẢ & IN MA TRẬN NHẦM LẪN TỔNG
# ═══════════════════════════════════════════════════════════

def save_and_print_results(all_pred_details, classes):
    df = pd.DataFrame(all_pred_details).sort_values(by="stt")
    if not os.path.exists("predict"):
        os.makedirs("predict")
    df.to_csv("predict/knn_eval_details.csv", index=False)

    y_true_all = df["y_true"].values
    y_pred_all = df["y_pred"].values

    # Ma trận nhầm lẫn tổng
    cm_total = build_confusion_matrix(y_true_all, y_pred_all, classes)
    per_cls  = per_class_metrics(cm_total, classes)

    print(f"\n{'='*65}")
    print("MA TRẬN NHẦM LẪN TỔNG (toàn bộ fold)")
    print(f"{'='*65}")
    print("  (Hàng = Thực tế  |  Cột = Dự đoán)\n")
    print_confusion_matrix(cm_total, classes)

    print(f"\n{'='*65}")
    print("CHỈ SỐ TỪNG NHÃN (One-vs-Rest)")
    print(f"{'='*65}")
    print(f"{'Nhãn':<15} | {'TP':>6} | {'TN':>6} | {'FP':>6} | {'FN':>6} | "
          f"{'Prec':>8} | {'Recall':>8} | {'F1':>8} | {'Support':>8}")
    print("-" * 87)
    for c in classes:
        m = per_cls[c]
        print(f"{str(c):<15} | {m['TP']:>6} | {m['TN']:>6} | {m['FP']:>6} | {m['FN']:>6} | "
              f"{m['Precision']:>8.4f} | {m['Recall']:>8.4f} | {m['F1']:>8.4f} | {m['Support']:>8}")

    print(f"\nFile chi tiết: predict/knn_eval_details.csv")
    return per_cls


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    # ── Đọc data đã scale ──────────────────────────────────
    path = "scale/data_scaled.csv"
    if not os.path.exists(path):
        print("Lỗi: Không tìm thấy scale/data_scaled.csv.")
        print("Đang chạy preprocessing để tạo file scale...")
        run_preprocessing("data/data.csv", "data/feature_names.txt",
                          training=True, scaled_output_path=path)

    df = pd.read_csv(path)
    X  = df.iloc[:, :-1].values.astype(float)
    y  = df.iloc[:, -1].values
    classes = sorted(np.unique(y), key=lambda x: str(x))
    is_multiclass = len(classes) > 2

    print(f"\nNhãn phát hiện: {classes}  ({'đa nhãn' if is_multiclass else '2 nhãn'})")

    # ── Chọn phương pháp tổng hợp (chỉ hỏi khi đa nhãn) ───
    agg_method = None
    if is_multiclass:
        print(f"\n{'='*58}")
        print("CHỌN PHƯƠNG PHÁP TỔNG HỢP CHỈ SỐ (cho Precision/Recall/F1)")
        print(f"{'='*58}")
        print("  1. Accuracy       — Tỉ lệ dự đoán đúng tổng thể")
        print("  2. Macro-Average  — Trung bình vị mô (coi các nhãn ngang nhau)")
        print("  3. Weighted-Average — Trung bình có trọng số theo số mẫu mỗi nhãn")
        choice = input("Nhập lựa chọn (1/2/3): ").strip()
        agg_map = {"1": "accuracy", "2": "macro", "3": "weighted"}
        if choice not in agg_map:
            print("Lựa chọn không hợp lệ. Dùng mặc định Macro-Average.")
            choice = "2"
        agg_method = agg_map[choice]
        agg_name   = {"accuracy": "Accuracy", "macro": "Macro-Average",
                      "weighted": "Weighted-Average"}[agg_method]
        print(f"=> Phương pháp đã chọn: {agg_name}")

    # ── Tham số KNN ────────────────────────────────────────
    print(f"\n{'='*58}")
    print("THAM SỐ KNN")
    k_input = input("Nhập K (số láng giềng, mặc định=3): ").strip()
    p_input = input("Nhập p (1=Manhattan, 2=Euclidean, mặc định=2): ").strip()
    k_knn = int(k_input) if k_input.isdigit() and int(k_input) > 0 else 3
    p_val = int(p_input) if p_input in ("1", "2") else 2
    print(f"=> K={k_knn}, p={p_val} ({'Euclidean' if p_val==2 else 'Manhattan'})")

    # ── Chạy K-Fold CV ─────────────────────────────────────
    cv_history, all_pred_details = run_k_fold_cv(
        X, y, classes, k_folds=5, k_knn=k_knn, p=p_val, agg_method=agg_method
    )

    # ── Ma trận nhầm lẫn tổng + chỉ số từng nhãn ──────────
    save_and_print_results(all_pred_details, classes)

    # ── Đánh giá tổng quát K-Fold Mean ─────────────────────
    print(f"\n{'='*58}")
    print("ĐÁNH GIÁ TỔNG QUÁT (K-FOLD MEAN)")
    print(f"{'='*58}")
    for metric in ["Accuracy", "Precision", "Recall", "F1", "AUC"]:
        mean_val = np.mean(cv_history[metric])
        print(f"  {metric:<15}: {mean_val:.4f}  ({mean_val*100:.2f}%)")
    print("-" * 58)

    # ── CV Score ───────────────────────────────────────────
    print(f"\n{'='*58}")
    print("CV SCORE")
    print(f"{'='*58}")
    print("Bạn muốn tính CV Score dựa trên chỉ số nào (Mi)?")
    options = {"1": "Accuracy", "2": "Precision", "3": "Recall", "4": "F1", "5": "AUC"}
    for k_opt, v in options.items():
        print(f"  {k_opt}. {v}")

    cv_choice = input("Nhập lựa chọn (1-5): ").strip()
    if cv_choice not in options:
        print("Lựa chọn không hợp lệ. Kết thúc.")
        return

    chosen_metric = options[cv_choice]
    scores  = np.array(cv_history[chosen_metric])
    cv_mean = np.mean(scores)
    cv_std  = np.std(scores)

    print(f"\n{'-'*58}")
    print(f"KẾT QUẢ CV SCORE  (Mi = {chosen_metric})")
    print(f"  CV_Score_mean  :  {cv_mean:.6f}")
    print(f"  CV_Score_std   :  {cv_std:.6f}")
    print(f"{'='*58}")


if __name__ == "__main__":
    main()