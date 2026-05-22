import numpy as np
from .base import BaseDR

class LaplacianEigenmaps(BaseDR):
    def __init__(self, n_neighbors: int = 5, n_components: int = 2, sigma: float = 1.0, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.k = n_neighbors
        self.sigma = sigma
        self.embedding_ = None
        self._X_fit = None

    def _fit(self, X: np.ndarray) -> None:
        distance_matrix = self._compute_distance_matrix(X)
        
        neighbors = self._find_neighbors(distance_matrix, self.k)
        
        W = self._build_weight_matrix(X, neighbors, sigma=self.sigma) # weight matrix
        D = self._build_degree_matrix(W) # degree matrix
        L = self._build_laplacian(D, W) # laplacian matrix
        
        eigenvalues, eigenvectors = np.linalg.eigh(L) # eigenvalues and eigenvectors
        
        # sort eigenvalues and eigenvectors from smallest to largest
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # exclude the singular vector corresponding to the singular value 0
        non_zero_indices = np.where(eigenvalues > 1e-5)[0]
        
        # get the smallest n_components eigenvalues and eigenvectors
        selected_indices = non_zero_indices[:self.n_components]
        
        self.embedding_ = eigenvectors[:, selected_indices]
        self._X_fit = X

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if X is self._X_fit:
            return self.embedding_
        if self._X_fit is not None and X.shape == self._X_fit.shape and np.allclose(X, self._X_fit):
            return self.embedding_
        raise NotImplementedError("Out-of-sample extension for Laplacian Eigenmaps is not implemented.")

    def _compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        distance_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])
        return distance_matrix

    def _find_neighbors(self, distance_matrix: np.ndarray, k: int) -> np.ndarray:
        n_samples = distance_matrix.shape[0]
        neighbors = np.zeros((n_samples, k), dtype=int)
        for i in range(n_samples):
            sorted_indices = np.argsort(distance_matrix[i])
            neighbors[i] = sorted_indices[1:k+1] # Bỏ chính nó (chỉ số 0)
        return neighbors

    def _build_weight_matrix(self, X: np.ndarray, neighbors: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        n_samples = X.shape[0]
        W = np.zeros((n_samples, n_samples))
        
        # determine the relationship between neighbors (symmetric graph)
        # if i is a neighbor of j OR j is a neighbor of i
        for i in range(n_samples):
            for j in neighbors[i]:
                W[i, j] = 1.0
                W[j, i] = 1.0
        
        # calculate the weight according to the standard formula: exp(-||xi - xj||^2 / (2 * sigma^2))
        for i in range(n_samples):
            for j in range(n_samples):
                if W[i, j] == 1.0: # only calculate if there is a neighbor relationship
                    distance = np.linalg.norm(X[i] - X[j])
                    W[i, j] = np.exp(-(distance ** 2) / (2 * (sigma ** 2)))
                    
        return W

    def _build_degree_matrix(self, W: np.ndarray) -> np.ndarray:
        n_samples = W.shape[0]
        D = np.zeros((n_samples, n_samples))
        degrees = np.sum(W, axis=1)
        np.fill_diagonal(D, degrees)
        return D

    def _build_laplacian(self, D: np.ndarray, W: np.ndarray) -> np.ndarray:
        return D - W