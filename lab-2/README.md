# Lab 2 - Dimensionality Reduction

## Thông tin đồ án

- Môn học: CSC14005 - Introduction to Machine Learning
- Chương sách được chọn: Chapter 12, *Dimensionality Reduction*, trong *Foundations of Machine Learning* (Mohri, Rostamizadeh, Talwalkar, 2nd edition).
- Nội dung: PCA, KPCA, Isomap, Laplacian Eigenmaps, LLE và bổ đề Johnson--Lindenstrauss.
- Mở rộng: t-SNE/UMAP làm baseline trực quan và notebook Neg-t-SNE; phần nghiên cứu thảo luận bài báo PaCMAP của Wang et al.

| MSSV | Họ và tên |
|------|-----------|
| 23120119 | Đỗ Tiến Đạt |
| 23120163 | Lê Nhật Minh Tâm |
| 23120137 | Triệu Tuấn Kiệt |
| 23120134 | Nguyễn Đăng Khoa |
| 23120105 | Huỳnh Mạnh Tường |

## Cài đặt

Yêu cầu Python `>= 3.13`. Cách khuyến nghị dùng `uv`:

```powershell
cd lab-2
uv sync
```

Có thể dùng `pip` để cài đầy đủ dependency, gồm cả notebook Neg-t-SNE:

```powershell
python -m pip install -r requirements.txt
```

Các tính năng tùy chọn được cài riêng khi cần:

```powershell
uv sync --extra data
uv sync --extra contrastive
```

`data` thêm Scanpy/Pillow cho tập dữ liệu mở rộng. `contrastive` thêm
`torch` và `contrastive-ne` để chạy `contrastive_learning.ipynb`.
Trên Windows, `contrastive-ne` phụ thuộc `annoy`, có thể cần MSVC Build
Tools để biên dịch; dependency này không cần cho script tạo số liệu báo cáo
PCA/KPCA/Isomap.

## Dữ liệu

- MNIST: đặt `code/data/raw/mnist.hdf5`, hoặc sử dụng loader tự động trong [`code/data/README.md`](code/data/README.md).
- Tải trước các dữ liệu được hỗ trợ:

```powershell
uv run python code/data/load_data.py
```

## Tái lập kết quả

Tạo lại năm hình và các số liệu PCA, KPCA, Isomap được trích trong báo cáo:

```powershell
uv run python code/experiments/report_pca_kpca_isomap.py
```

Biên dịch báo cáo từ thư mục `report/`:

```powershell
pdflatex -jobname=report -interaction=nonstopmode -halt-on-error main.tex
bibtex report
pdflatex -jobname=report -interaction=nonstopmode -halt-on-error main.tex
pdflatex -jobname=report -interaction=nonstopmode -halt-on-error main.tex
```

| File | Mô tả |
|------|-------|
| [`code/experiments/report_pca_kpca_isomap.py`](code/experiments/report_pca_kpca_isomap.py) | Sinh hình/bảng trong phần thực nghiệm chính với `SEED = 42`. |
| [`code/experiments/baseline.ipynb`](code/experiments/baseline.ipynb) | Baseline PCA, KPCA, Isomap, LLE, Laplacian, t-SNE và UMAP; MNIST 10,000 mẫu, `SEED = 42`. |
| [`code/experiments/contrastive_learning.ipynb`](code/experiments/contrastive_learning.ipynb) | Mở rộng Neg-t-SNE (`cne.CNE`) và wrapper t-SNE/UMAP. |

## Cấu trúc

```text
lab-2/
|-- README.md
|-- requirements.txt
|-- pyproject.toml
|-- report/
|   |-- main.tex
|   |-- ref/
|   |   |-- ref.tex
|   |   `-- ref.bib
|   |-- report.pdf
|   |-- content/
|   `-- figures/
`-- code/
    |-- README.md
    |-- models/
    |-- experiments/
    |-- data/
    `-- utils/
```

Chi tiết về từng tệp mã nguồn và phạm vi from-scratch/wrapper nằm trong
[`code/README.md`](code/README.md).
