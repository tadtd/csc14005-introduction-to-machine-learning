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
| `experiments/report_pca_kpca_isomap.py` | Sinh lại hình và số liệu chính được dùng trong báo cáo. |
| `experiments/baseline.ipynb` | So sánh baseline bổ sung, gồm LLE/Laplacian và wrapper t-SNE/UMAP. |
| `experiments/contrastive_learning.ipynb` | Phần mở rộng Neg-t-SNE. |

## Chạy thực nghiệm chính

```powershell
uv run python code/experiments/report_pca_kpca_isomap.py
```

Script sử dụng `SEED = 42` và ghi hình vào `report/figures/`. Các notebook
được mở từ thư mục `lab-2/` và sử dụng kernel của `.venv`.
