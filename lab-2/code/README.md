# Mã nguồn giảm chiều

## Môi trường chạy

Từ thư mục `lab-2/`, cài môi trường cho phần bắt buộc:

```powershell
uv sync
```

Hoặc cài đầy đủ dependency, gồm phần mở rộng Neg-t-SNE:

```powershell
python -m pip install -r requirements.txt
```

Nếu đang dùng `uv`, notebook Neg-t-SNE là phần mở rộng và cần thêm dependency:

```powershell
uv sync --extra contrastive
```

## Phạm vi cài đặt

| Thuật toán | Hiện thực |
|------------|-----------|
| PCA | Cài đặt từ đầu bằng NumPy/SVD trong `models/pca.py`. |
| KPCA | Cài đặt từ đầu, gồm kernel và kernel centering, trong `models/kpca.py`. |
| Isomap | Cài đặt từ đầu, gồm đồ thị k-NN, đường đi ngắn nhất và MDS, trong `models/isomap.py`. |
| LLE | Cài đặt từ đầu trong `models/lle.py`. |
| Laplacian Eigenmaps | Cài đặt từ đầu trong `models/laplacian_eigen.py`. |
| t-SNE | Wrapper cho `sklearn.manifold.TSNE` trong `models/t_sne.py`; chỉ dùng mở rộng. |
| UMAP | Wrapper cho `umap-learn` trong `models/umap.py`; chỉ dùng mở rộng. |
| Neg-t-SNE | Wrapper cho `cne.CNE` trong `models/neg_tsne.py`; chỉ dùng mở rộng. |

## Các tệp chính

| Đường dẫn | Vai trò |
|-----------|---------|
| `models/base.py` | API chung, kiểm tra input, preprocessing và đo thời gian. |
| `models/*.py` | Các mô hình giảm chiều nêu trên. |
| `data/load_data.py` | Nạp hoặc tạo các tập Wine, circles, Swiss roll, COIL-20 và MNIST. |
| `utils/knn_graph.py` | Xây đồ thị k-NN đối xứng cho Neg-t-SNE. |
| `experiments/report_pca_kpca_isomap.py` | Sinh lại hình và số liệu kiểm chứng lý thuyết PCA/KPCA/Isomap. |
| `experiments/export_baseline_figures.py` | Xuất lưới embedding baseline và hình độ nhạy tham số vào `report/figures/`. |
| `experiments/baseline.ipynb` | So sánh baseline đầy đủ (metrics, heatmap) trên bốn tập dữ liệu. |
| `experiments/contrastive_learning.ipynb` | Phần mở rộng Neg-t-SNE. |

## Chạy thực nghiệm chính

```powershell
uv run python code/experiments/report_pca_kpca_isomap.py
uv run python code/experiments/export_baseline_figures.py
```

Script đầu dùng `SEED = 42` và ghi hình kiểm chứng lý thuyết (Wine, Circles
mở rộng, Swiss roll) vào `report/figures/`. Script thứ hai xuất lưới baseline
bốn tập và hình độ nhạy LLE/Laplacian. Các notebook được mở từ thư mục `lab-2/`
và sử dụng kernel của `.venv`.
