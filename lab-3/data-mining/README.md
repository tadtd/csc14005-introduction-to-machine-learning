# Part III – Data Mining: Tensor Decomposition for EEG Analysis

## Tổng quan

Module này triển khai các thuật toán tensor decomposition cho phân tích dữ liệu EEG, phục vụ đồ án "Foundations of Tensor Computations for AI (NeurIPS 2025)".

### Thuật toán đã cài đặt
- **CP-ALS**: Canonical Polyadic decomposition via Alternating Least Squares (từ đầu bằng NumPy)
- **CP-OPT**: CP decomposition via gradient-based optimization (L-BFGS-B, scipy)
- **PARAFAC2**: Parallel Factor Analysis 2 (cho dữ liệu có kích thước không đều)

### Dữ liệu
- **PhysioNet EEG Motor Movement/Imagery Dataset** (109 subjects, 64 channels)
- Tải tự động qua `mne.datasets.eegbci`

---

## Cài đặt

### 1. Tạo môi trường ảo (khuyến nghị)

```bash
# Với venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Hoặc:

```bash
pip install numpy scipy matplotlib mne jupyter tqdm ipykernel
```

### 3. Đăng ký kernel cho Jupyter (nếu dùng venv)

```bash
python -m ipykernel install --user --name=data-mining --display-name="Python (Data Mining)"
```

---

## Cấu trúc thư mục

```
data-mining/
├── src/                          # Source code
│   ├── __init__.py               # Package initialization
│   ├── cp_als.py                 # CP-ALS from scratch (NumPy)
│   ├── cp_opt.py                 # CP-OPT via scipy L-BFGS-B
│   ├── parafac2.py               # PARAFAC2 implementation
│   ├── data.py                   # EEG data loading & tensor construction
│   ├── evaluate.py               # Evaluation metrics
│   └── visualize.py              # Visualization functions
├── notebooks/                    # Jupyter notebooks
│   ├── 01_synthetic.ipynb        # Algorithm validation (synthetic tensor)
│   ├── 02_eeg_cp.ipynb           # CP decomposition on real EEG
│   ├── 03_parafac2.ipynb         # PARAFAC2 on real EEG
│   └── 04_cmtf.ipynb             # CMTF demonstration
├── pyproject.toml                # Project configuration
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

---

## Chạy Notebooks

Chạy theo thứ tự:

```bash
cd data-mining
jupyter notebook
```

1. **01_synthetic.ipynb**: Xác nhận tính đúng đắn của thuật toán trên tensor tổng hợp
2. **02_eeg_cp.ipynb**: CP decomposition trên dữ liệu EEG thật
3. **03_parafac2.ipynb**: PARAFAC2 trên dữ liệu EEG thật
4. **04_cmtf.ipynb**: Minh họa CMTF

### Tải dữ liệu EEG

Dữ liệu sẽ tự động tải khi chạy notebook `02_eeg_cp.ipynb` lần đầu. MNE cache dữ liệu vào `~/mne_data/`.

Nếu muốn tải trước:

```python
from src.data import download_eeg_data
download_eeg_data(subjects=list(range(1, 21)), runs=[4, 8, 12], verbose=True)
```

---

## Lưu ý quan trọng

- **KHÔNG** sử dụng dữ liệu giả cho kết quả chính
- Notebook `01_synthetic.ipynb` dùng dữ liệu tổng hợp CHỈ để kiểm tra tính đúng đắn thuật toán
- Tất cả kết quả thực nghiệm (notebooks 02-04) đều từ dữ liệu PhysioNet thật
- Mọi biểu đồ đều được tạo từ dữ liệu đã tính toán, không giả lập

---

## Xử lý sự cố

### Lỗi `ModuleNotFoundError: No module named 'mne'`
```bash
pip install mne
```

### Lỗi tải dữ liệu EEG
Kiểm tra kết nối internet. Dữ liệu tải từ PhysioNet (~2 GB cho 20 subjects).

### Lỗi `ImportError` khi import `src`
Đảm bảo chạy notebook từ thư mục `data-mining/`:
```bash
cd data-mining
jupyter notebook
```

### Python version
Khuyến nghị Python 3.10–3.12 cho tương thích tốt nhất với MNE.
