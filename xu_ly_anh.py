"""Tiền xử lý ảnh X-quang và trích xuất đặc trưng HOG."""
import numpy as np
from PIL import Image
from skimage.feature import hog
import matplotlib
matplotlib.use("Agg")  # Lưu hình ra file, không cần mở cửa sổ đồ họa.
import matplotlib.pyplot as plt

# Các ảnh đều được đưa về cùng kích thước và cùng cấu hình HOG.
KICH_THUOC = (128, 128)
THAM_SO_HOG = dict(orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm="L2-Hys")


def doc_anh(duong_dan):
    """Đọc ảnh và chuyển sang mức xám: mỗi điểm ảnh có một mức sáng."""
    with Image.open(duong_dan) as anh:
        return anh.convert("L")


def tien_xu_ly(anh_xam):
    """Resize 128x128, sau đó đổi miền giá trị từ 0..255 sang 0..1."""
    anh_nho = anh_xam.resize(KICH_THUOC, Image.Resampling.BILINEAR)
    return np.asarray(anh_nho, dtype=np.float32) / 255.0


def lay_dac_trung(anh_da_xu_ly):
    """HOG mô tả hướng thay đổi độ sáng; trả về vector 8.100 số."""
    return hog(anh_da_xu_ly, **THAM_SO_HOG).astype(np.float32)


def xu_ly_nhieu_anh(danh_sach_anh):
    """Mỗi dòng X là đặc trưng của một ảnh; mỗi cột là một đặc trưng."""
    cac_dac_trung = []
    for thu_tu, duong_dan in enumerate(danh_sach_anh, start=1):
        anh_xam = doc_anh(duong_dan)
        anh_da_xu_ly = tien_xu_ly(anh_xam)
        cac_dac_trung.append(lay_dac_trung(anh_da_xu_ly))
        if thu_tu % 500 == 0:
            print(f"  Đã xử lý {thu_tu}/{len(danh_sach_anh)} ảnh", flush=True)
    return np.asarray(cac_dac_trung, dtype=np.float32)


def ve_minh_hoa(duong_dan, file_hinh):
    """Xuất ảnh mức xám, ảnh đã xử lý, HOG và histogram."""
    anh_xam = doc_anh(duong_dan)
    anh_da_xu_ly = tien_xu_ly(anh_xam)
    _, anh_hog = hog(anh_da_xu_ly, visualize=True, **THAM_SO_HOG)
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes[0].imshow(anh_xam, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Ảnh mức xám ban đầu\n{anh_xam.width} × {anh_xam.height}")
    axes[1].imshow(anh_da_xu_ly, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Resize 128 × 128\nChuẩn hóa về [0, 1]")
    axes[2].imshow(anh_hog, cmap="gray")
    axes[2].set_title("Minh họa HOG\nKhông phải vùng tổn thương")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(file_hinh, dpi=160, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    # Histogram chỉ dùng để quan sát, không tham gia trích xuất đặc trưng.
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(anh_da_xu_ly.ravel(), bins=64, range=(0, 1))
    ax.set(xlabel="Mức sáng sau chuẩn hóa", ylabel="Số điểm ảnh",
           title="Histogram mức xám của ảnh đã resize")
    fig.tight_layout()
    fig.savefig(file_hinh.with_name(file_hinh.stem + "_histogram.png"), dpi=160)
    plt.close(fig)
