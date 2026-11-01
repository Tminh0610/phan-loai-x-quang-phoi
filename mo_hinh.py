"""Chia dữ liệu, huấn luyện và đánh giá bộ phân loại.

Quy ước nhãn: 0 = NORMAL, 1 = PNEUMONIA."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def chia_du_lieu(nhan):
    """Chia chỉ số ảnh: 80% học, 20% validation; giữ gần đúng tỷ lệ nhãn.

    Danh sách đầu vào đã được loại bản sao chính xác trong main.py.
    random_state cố định giúp chạy lại cùng cách chia.
    """
    chi_so = np.arange(len(nhan))
    return train_test_split(chi_so, test_size=0.2, random_state=42,
                            stratify=nhan)


def huan_luyen(X_hoc, y_hoc):
    """Scaler và Logistic Regression chỉ học từ phần huấn luyện."""
    mo_hinh = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="liblinear",
                           class_weight="balanced", random_state=42),
    )
    # balanced: điều chỉnh trọng số lỗi theo lớp, không làm số ảnh bằng nhau.
    mo_hinh.fit(X_hoc, y_hoc)
    return mo_hinh


def danh_gia(mo_hinh, X_kiem_tra, y_kiem_tra, file_hinh):
    """Tính chỉ số validation và lưu ma trận nhầm lẫn."""
    du_doan = mo_hinh.predict(X_kiem_tra)
    ma_tran = confusion_matrix(y_kiem_tra, du_doan, labels=[0, 1])
    tn, fp, fn, tp = ma_tran.ravel()
    ket_qua = {
        "accuracy": float(accuracy_score(y_kiem_tra, du_doan)),
        "precision_viem_phoi": float(precision_score(y_kiem_tra, du_doan, zero_division=0)),
        "recall_viem_phoi": float(recall_score(y_kiem_tra, du_doan, zero_division=0)),
        "specificity_binh_thuong": float(tn / (tn + fp)),
        "f1_viem_phoi": float(f1_score(y_kiem_tra, du_doan, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(ma_tran, display_labels=["NORMAL", "PNEUMONIA"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set(xlabel="Nhãn dự đoán", ylabel="Nhãn thật",
           title="Kết quả trên validation nội bộ")
    fig.tight_layout()
    fig.savefig(file_hinh, dpi=160)
    plt.close(fig)
    return ket_qua, du_doan
