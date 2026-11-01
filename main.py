"""Chương trình phân loại ảnh X-quang phổi của Nhóm 8.

Huấn luyện: python main.py --ket-qua ket_qua_lan_2
Dự đoán: python main.py --anh "duong_dan_anh.jpeg"
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import joblib
import numpy as np
from xu_ly_anh import (doc_anh, tien_xu_ly, lay_dac_trung, xu_ly_nhieu_anh,
                      ve_minh_hoa, KICH_THUOC, THAM_SO_HOG)
from mo_hinh import chia_du_lieu, huan_luyen, danh_gia

THU_MUC = Path(__file__).resolve().parent
TEN_NHAN = ["NORMAL", "PNEUMONIA"]


def doc_danh_sach_anh(thu_muc_data):
    """Chỉ đọc thư mục train gốc; bỏ bản sao có cùng nội dung byte.

    Không xóa ảnh trên đĩa. Hash không phát hiện mọi ảnh gần trùng/bệnh nhân.
    """
    danh_sach, nhan, ma_hash, anh_trung = [], [], [], []
    da_gap = {}
    for ma_nhan, ten in enumerate(TEN_NHAN):
        folder = thu_muc_data / "train" / ten
        if not folder.is_dir():
            raise FileNotFoundError(f"Không thấy {folder}. Đọc mục Đặt dữ liệu trong README.md.")
        tep_anh = sorted(p for p in folder.iterdir() if p.suffix.lower() in [".jpeg", ".jpg", ".png"])
        if not tep_anh:
            raise ValueError(f"Thư mục không có ảnh: {folder}")
        for p in tep_anh:
            chu_ky = hashlib.sha256(p.read_bytes()).hexdigest()
            if chu_ky in da_gap:
                anh_cu, nhan_cu = da_gap[chu_ky]
                if nhan_cu != ma_nhan:
                    raise ValueError(f"Ảnh giống nhau nhưng khác nhãn: {p} và {anh_cu}")
                anh_trung.append([p.relative_to(thu_muc_data).as_posix(),
                                  anh_cu.relative_to(thu_muc_data).as_posix()])
                continue
            da_gap[chu_ky] = (p, ma_nhan)
            danh_sach.append(p)
            nhan.append(ma_nhan)
            ma_hash.append(chu_ky)
    return danh_sach, np.asarray(nhan), ma_hash, anh_trung


def ghi_csv(file, tieu_de, cac_dong):
    with file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(tieu_de)
        writer.writerows(cac_dong)


def main():
    parser = argparse.ArgumentParser(description="Nhóm 8 - Phân loại ảnh X-quang phổi bằng HOG và Logistic Regression")
    parser.add_argument("--data", type=Path, default=THU_MUC / "data" / "chest_xray")
    parser.add_argument("--ket-qua", type=Path, default=THU_MUC / "ket_qua")
    parser.add_argument("--anh", type=Path, help="Dự đoán một ảnh bằng mô hình đã lưu")
    args = parser.parse_args()
    out = args.ket_qua
    out.mkdir(parents=True, exist_ok=True)
    file_model = out / "mo_hinh.joblib"

    if args.anh is not None:
        # Chỉ nạp model do nhóm tạo hoặc tin cậy: joblib không phải định dạng an toàn cho file lạ.
        goi = joblib.load(file_model)
        if goi["kich_thuoc"] != KICH_THUOC or goi["hog"] != THAM_SO_HOG:
            raise ValueError("Cấu hình ảnh đã thay đổi; cần huấn luyện lại mô hình.")
        x = lay_dac_trung(tien_xu_ly(doc_anh(args.anh))).reshape(1, -1)
        nhan = int(goi["mo_hinh"].predict(x)[0])
        print("Nhãn dự đoán:", TEN_NHAN[nhan])
        print("Đây là minh họa phân loại ảnh, không phải kết luận chẩn đoán.")
        return

    if file_model.exists():
        raise FileExistsError("Đã có kết quả. Muốn chạy lại, thêm --ket-qua ket_qua_lan_2 để giữ kết quả cũ.")

    print("Bước 1/5: Đọc ảnh train và bỏ bản sao chính xác...", flush=True)
    ds_anh, y, hashes, anh_trung = doc_danh_sach_anh(args.data)
    hoc, kiem_tra = chia_du_lieu(y)
    # Bảo đảm không có cùng ảnh ở hai phía sau khi chia.
    assert not (set(hashes[i] for i in hoc) & set(hashes[i] for i in kiem_tra))
    ghi_csv(out / "anh_trung.csv", ["ban_sao_bo_qua", "anh_giu_lai"], anh_trung)
    tap_hoc = set(hoc.tolist())
    ghi_csv(out / "chia_du_lieu.csv", ["anh", "nhan", "tap", "sha256"],
            [[p.relative_to(args.data).as_posix(), TEN_NHAN[y[i]],
              "hoc" if i in tap_hoc else "validation", hashes[i]] for i, p in enumerate(ds_anh)])
    print(f"Giữ {len(y)} ảnh; bỏ qua {len(anh_trung)} bản sao; học {len(hoc)}, validation {len(kiem_tra)}.", flush=True)

    print("Bước 2/5: Xử lý ảnh và lấy đặc trưng HOG...", flush=True)
    X = xu_ly_nhieu_anh(ds_anh)
    print("Kích thước ma trận đặc trưng:", X.shape, flush=True)

    print("Bước 3/5: Huấn luyện Logistic Regression...", flush=True)
    model = huan_luyen(X[hoc], y[hoc])

    print("Bước 4/5: Kiểm tra trên validation...", flush=True)
    diem, du_doan = danh_gia(model, X[kiem_tra], y[kiem_tra], out / "ma_tran_nham_lan.png")
    ghi_csv(out / "du_doan_validation.csv", ["anh", "nhan_that", "nhan_du_doan"],
            [[ds_anh[i].relative_to(args.data).as_posix(), TEN_NHAN[y[i]], TEN_NHAN[d]]
             for i, d in zip(kiem_tra, du_doan)])
    thong_ke = {"so_anh_goc": len(y) + len(anh_trung), "ban_sao_bo_qua": len(anh_trung),
                "so_anh_hoc": len(hoc), "so_anh_validation": len(kiem_tra),
                "so_anh_hoc_moi_lop": {ten: int(sum(y[hoc] == j)) for j, ten in enumerate(TEN_NHAN)},
                "so_anh_validation_moi_lop": {ten: int(sum(y[kiem_tra] == j)) for j, ten in enumerate(TEN_NHAN)},
                "seed": 42, "test_goc_da_dung": False, "ket_qua_validation": diem}
    (out / "ket_qua.json").write_text(json.dumps(thong_ke, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Bước 5/5: Lưu mô hình và hình minh họa...", flush=True)
    for j, ten in enumerate(TEN_NHAN):
        i = next(i for i in hoc if y[i] == j)  # Hình minh họa lấy từ phần học.
        ve_minh_hoa(ds_anh[i], out / f"xu_ly_{ten}.png")
    joblib.dump({"mo_hinh": model, "kich_thuoc": KICH_THUOC, "hog": THAM_SO_HOG}, file_model)
    print(json.dumps(diem, indent=2, ensure_ascii=False))
    print("Hoàn tất. Xem kết quả tại:", out.resolve())


if __name__ == "__main__":
    main()
