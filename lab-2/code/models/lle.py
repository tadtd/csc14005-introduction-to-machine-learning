import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse
from scipy.sparse.linalg import eigsh

from .base import BaseDR

class MyLLE(BaseDR):
    """
    Locally Linear Embedding (LLE) Implementation from Scratch.
    Bám sát quy trình toán học: Tìm láng giềng -> Giải trọng số W -> Tối ưu hóa Y.
    """
    def __init__(self, n_neighbors: int = 10, n_components: int = 2, reg: float = 1e-3, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.t = n_neighbors      # t nearest neighbors
        self.reg = reg            # Regularization parameter for C'
        self.embedding_ = None
        self._X_fit = None

    def _fit(self, X: np.ndarray) -> None:
        m, n = X.shape
        
        # --- BƯỚC 1: Tìm t láng giềng gần nhất ---
        # Sử dụng Euclidean distance
        dist_matrix = cdist(X, X, metric='euclidean')
        # Lấy t láng giềng (bỏ qua chính nó tại vị trí index 0)
        neighbors_idx = np.argsort(dist_matrix, axis=1)[:, 1:self.t + 1]
        
        # --- BƯỚC 2: Xây dựng ma trận trọng số W (Equation 12.8 & 12.9) ---
        W = self._compute_reconstruction_weights(X, neighbors_idx)
        
        # --- BƯỚC 3: Tối ưu hóa tọa độ nhúng Y (Equation 12.10) ---
        self.embedding_ = self._optimize_embedding(W)
        self._X_fit = X

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if X is self._X_fit:
            return self.embedding_
        if self._X_fit is not None and X.shape == self._X_fit.shape and np.allclose(X, self._X_fit):
            return self.embedding_
        raise NotImplementedError("Out-of-sample extension for LLE is not implemented.")

    def _compute_reconstruction_weights(self, X: np.ndarray, neighbors_idx: np.ndarray) -> sparse.csr_matrix:
        m = X.shape[0]
        rows, cols, data = [], [], []
        
        for i in range(m):
            idx_i = neighbors_idx[i]
            # Tính xi - xj cho tất cả j trong láng giềng của i
            # Xi có kích thước (t, n)
            Xi = X[idx_i] - X[i]
            
            # Tính ma trận covariance cục bộ C' (Equation 12.8)
            # C' = (xi - xj)^T * (xi - xk) -> (t, t) matrix
            C_prime = np.dot(Xi, Xi.T)
            
            # Regularization để đảm bảo C' khả nghịch (Tránh lỗi ma trận suy biến)
            trace = np.trace(C_prime)
            r = self.reg * trace if trace > 0 else self.reg
            C_prime += np.eye(self.t) * r
            
            # Giải hệ phương trình C' * w = 1 để tìm weights chưa chuẩn hóa
            w_i = np.linalg.solve(C_prime, np.ones(self.t))
            
            # Chuẩn hóa để sum(w_i) = 1 (Theo yêu cầu Step 2 trong hình)
            w_i /= np.sum(w_i)
            
            for j_idx, weight in enumerate(w_i):
                rows.append(i)
                cols.append(idx_i[j_idx])
                data.append(weight)
        
        return sparse.csr_matrix((data, (rows, cols)), shape=(m, m))

    def _optimize_embedding(self, W: sparse.csr_matrix) -> np.ndarray:
        m = W.shape[0]
        I = sparse.eye(m, format='csr')
        
        # Xây dựng ma trận M = (I - W)^T * (I - W)
        # Lưu ý: Tài liệu hình 2 ghi M = (I - W^T)(I - W^T) có thể là lỗi đánh máy, 
        # chuẩn thuật toán LLE phải là (I - W).T @ (I - W)
        I_minus_W = I - W
        M = I_minus_W.T @ I_minus_W
        
        # Tìm d+1 vector riêng ứng với trị riêng nhỏ nhất (Smallest Magnitude)
        # Sử dụng eigsh để xử lý ma trận thưa hiệu quả
        eigenvalues, eigenvectors = eigsh(M, k=self.n_components + 1, which='SM')
        
        # Loại bỏ vector riêng cuối cùng ứng với trị riêng = 0 (Constant vector)
        # Lấy d cột từ index 1 đến d+1
        return eigenvectors[:, 1:]