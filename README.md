# Nhóm 8 — Phân loại ảnh X-quang phổi

Dự án môn Xử lý ảnh sử dụng HOG và Logistic Regression để phân loại hai nhãn NORMAL và PNEUMONIA.

## Quy trình

Ảnh đầu vào → mức xám → resize 128 × 128 → chia 255 → HOG 8.100 đặc trưng → StandardScaler → Logistic Regression.

## Cấu trúc

```text
nhom_8/
├── main.py                  Chạy huấn luyện hoặc dự đoán
├── xu_ly_anh.py              Tiền xử lý, HOG và hình minh họa
├── mo_hinh.py                Chia tập, huấn luyện và đánh giá
├── requirements.txt         Phiên bản thư viện
├── data/chest_xray/train/    Dữ liệu NORMAL và PNEUMONIA
├── ket_qua/                 Mô hình và kết quả thực nghiệm
└── HOC_VA_THUYET_TRINH.md    Giải thích các bước của chương trình
```

## Cài đặt

Dùng Python 3.12. Mở terminal trong thư mục `nhom_8`:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Pillow đọc và resize ảnh; NumPy xử lý mảng; scikit-image tính HOG; scikit-learn huấn luyện và đánh giá; matplotlib vẽ hình; joblib lưu và nạp mô hình.

## Dữ liệu

Nguồn: [Chest X-Ray Images (Pneumonia) trên Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

Đặt ảnh tại:

```text
data/chest_xray/train/NORMAL/
data/chest_xray/train/PNEUMONIA/
```

Thư mục dự án trên máy đã có dữ liệu. Gói `nhom_8.zip` chỉ chứa mã nguồn, tài liệu, mô hình và kết quả; không chứa bộ ảnh. Xem `data/DAT_DU_LIEU_O_DAY.md` khi chạy trên máy khác.

## Dự đoán một ảnh

```powershell
.venv\Scripts\python.exe main.py --anh "C:\du_lieu\anh.jpeg"
```

Thay đường dẫn ví dụ bằng ảnh có thật. Lệnh nạp `ket_qua/mo_hinh.joblib` và trả nhãn NORMAL hoặc PNEUMONIA. Chỉ nạp mô hình từ nguồn tin cậy. Kết quả phục vụ thực nghiệm phân loại, không phải kết luận chẩn đoán.

## Huấn luyện lại

```powershell
.venv\Scripts\python.exe main.py --ket-qua ket_qua_lan_2
```

Dùng thư mục kết quả mới để giữ lần chạy trước. Nếu dữ liệu nằm ngoài dự án:

```powershell
.venv\Scripts\python.exe main.py --data "D:\du_lieu\chest_xray" --ket-qua ket_qua_lan_2
```

Để dự đoán bằng mô hình mới, thêm `--ket-qua ket_qua_lan_2` vào lệnh dự đoán.

## Kết quả thực nghiệm

Sau khi bỏ qua 26 tệp trùng byte trong 5.216 ảnh train gốc, chương trình giữ 5.190 ảnh và chia phân tầng 80/20 với seed 42. Phần học có 4.152 ảnh; validation có 1.038 ảnh.

| Chỉ số validation | Giá trị |
|---|---:|
| Accuracy | 93,93% |
| Precision PNEUMONIA | 98,89% |
| Recall PNEUMONIA | 92,86% |
| F1 PNEUMONIA | 95,78% |
| Specificity NORMAL | 97,01% |

`ket_qua.json` lưu số liệu tổng hợp; `du_doan_validation.csv` lưu từng dự đoán; `chia_du_lieu.csv` lưu cách chia và SHA256. Mô hình lưu chỉ học trên phần train và chính là mô hình được đánh giá trên validation.

## Phạm vi và giới hạn

Chương trình chưa dùng val/test gốc, chưa chia theo bệnh nhân và chưa phát hiện mọi ảnh gần trùng. HOG được tính trên toàn ảnh, chưa tách vùng phổi hoặc định vị tổn thương. Histogram chỉ phục vụ quan sát; chưa dùng CLAHE hay khử nhiễu. Trọng số lớp `balanced` điều chỉnh mức phạt khi học, không thay đổi số lượng ảnh hai lớp.
