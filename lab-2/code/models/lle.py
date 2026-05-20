import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse

from .base import BaseDR

class MyLLE(BaseDR):
    """
    Locally Linear Embedding (LLE) Implementation from Scratch.
    Sửa đổi chuẩn hóa theo công thức sách: M = (I - W)(I - W) và lọc bỏ chính xác trị riêng 0.
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
        dist_matrix = cdist(X, X, metric='euclidean')
        neighbors_idx = np.argsort(dist_matrix, axis=1)[:, 1:self.t + 1]
        
        # --- BƯỚC 2: Xây dựng ma trận trọng số W ---
        W = self._compute_reconstruction_weights(X, neighbors_idx)
        
        # --- BƯỚC 3: Tối ưu hóa tọa độ nhúng Y ---
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
            Xi = X[idx_i] - X[i]
            
            # Tính ma trận covariance cục bộ C' (Equation 12.8)
            C_prime = np.dot(Xi, Xi.T)
            
            # Regularization để đảm bảo C' khả nghịch
            trace = np.trace(C_prime)
            r = self.reg * trace if trace > 0 else self.reg
            C_prime += np.eye(self.t) * r
            
            # Giải hệ phương trình C' * w = 1 để tìm weights chưa chuẩn hóa
            w_i = np.linalg.solve(C_prime, np.ones(self.t))
            
            # Chuẩn hóa để sum(w_i) = 1 (Equation 12.9)
            w_i /= np.sum(w_i)
            
            for j_idx, weight in enumerate(w_i):
                rows.append(i)
                cols.append(idx_i[j_idx])
                data.append(weight)
        
        return sparse.csr_matrix((data, (rows, cols)), shape=(m, m))

    def _optimize_embedding(self, W: sparse.csr_matrix) -> np.ndarray:
        m = W.shape[0]
        I = sparse.eye(m, format='csr')
        
       
        I_minus_W = I - W
        M = I_minus_W.T @ I_minus_W  
        
        # Chuyển về dạng dense để giải toán trị riêng ổn định tuyệt đối bằng np.linalg.eigh
        M_dense = M.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(M_dense)
        
        # --- XỬ LÝ THEO TÀI LIỆU: Tìm và loại bỏ chính xác trị riêng 0 ---
        # Sách yêu cầu: "excluding the last singular vector corresponding to the singular value 0"
        non_zero_indices = np.where(eigenvalues > 1e-7)[0]
        
        # Lấy k (n_components) trị riêng nhỏ nhất còn lại đứng ngay sau trị riêng 0
        selected_indices = non_zero_indices[:self.n_components]
        
        return eigenvectors[:, selected_indices]