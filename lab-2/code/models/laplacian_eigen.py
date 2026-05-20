"""
Laplacian Eigenmaps: A dimensionality reduction technique based on spectral analysis.

This implementation follows the classical Laplacian Eigenmaps algorithm:
1. Construct a k-nearest neighbor graph
2. Build a weight matrix based on Gaussian kernel
3. Compute the graph Laplacian
4. Solve the generalized eigenvalue problem
5. Use the smallest eigenvectors as the low-dimensional embedding
"""

import numpy as np
from typing import Tuple

from .base import BaseDR

class LaplacianEigenmaps(BaseDR):
    def __init__(self, k: int = 5, n_components: int = 2, sigma: float = 1.0, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.k = k
        self.sigma = sigma
        self.embedding_ = None
        self._X_fit = None

    def _fit(self, X: np.ndarray) -> None:
        print("Step 1: Computing pairwise distance matrix...")
        distance_matrix = self._compute_distance_matrix(X)
        
        print("Step 2: Finding k nearest neighbors...")
        neighbors = self._find_neighbors(distance_matrix, self.k)
        
        print("Step 3: Building weight matrix...")
        W = self._build_weight_matrix(X, neighbors, sigma=self.sigma)
        
        print("Step 4: Building degree matrix...")
        D = self._build_degree_matrix(W)
        
        print("Step 5: Constructing graph Laplacian...")
        L = self._build_laplacian(D, W)
        
        print("Step 6: Computing eigenvectors and eigenvalues...")
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        
        print("Step 7: Selecting embedding dimensions...")
        self.embedding_ = eigenvectors[:, 1:self.n_components+1]
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
            neighbors[i] = sorted_indices[1:k+1]
        return neighbors

    def _build_weight_matrix(self, X: np.ndarray, neighbors: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        n_samples = X.shape[0]
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in neighbors[i]:
                distance = np.linalg.norm(X[i] - X[j])
                weight = np.exp(-(distance ** 2) / (sigma ** 2))
                W[i, j] = weight
        W = np.maximum(W, W.T)
        return W

    def _build_degree_matrix(self, W: np.ndarray) -> np.ndarray:
        n_samples = W.shape[0]
        D = np.zeros((n_samples, n_samples))
        degrees = np.sum(W, axis=1)
        np.fill_diagonal(D, degrees)
        return D

    def _build_laplacian(self, D: np.ndarray, W: np.ndarray) -> np.ndarray:
        return D - W
