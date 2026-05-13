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


def compute_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Euclidean distance matrix for all data points.
    
    Mathematical formula:
    D[i, j] = ||x_i - x_j||_2 = sqrt(sum_k (x_i_k - x_j_k)^2)
    
    Args:
        X: Input data matrix of shape (n_samples, n_features)
    
    Returns:
        Distance matrix of shape (n_samples, n_samples)
    """
    n_samples = X.shape[0]
    distance_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            # Compute Euclidean distance between point i and point j
            distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])
    
    return distance_matrix


def find_neighbors(distance_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Find the k nearest neighbors for each point using the distance matrix.
    
    For each point, we find the indices of its k nearest neighbors
    (excluding itself).
    
    Args:
        distance_matrix: Pairwise distance matrix of shape (n_samples, n_samples)
        k: Number of nearest neighbors to find
    
    Returns:
        Neighbor indices matrix of shape (n_samples, k), where neighbors[i, j]
        contains the index of the j-th nearest neighbor of point i
    """
    n_samples = distance_matrix.shape[0]
    neighbors = np.zeros((n_samples, k), dtype=int)
    
    for i in range(n_samples):
        # Get indices that would sort distances for point i
        # argsort returns indices in ascending order (nearest first)
        sorted_indices = np.argsort(distance_matrix[i])
        
        # Skip the first index (which is the point itself with distance 0)
        # Take the next k indices as nearest neighbors
        neighbors[i] = sorted_indices[1:k+1]
    
    return neighbors


def build_weight_matrix(X: np.ndarray, neighbors: np.ndarray, 
                       sigma: float = 1.0) -> np.ndarray:
    """
    Build the weight matrix W using the Gaussian kernel for neighboring points.
    
    Weight formula for neighbors:
    W[i, j] = exp(-||x_i - x_j||^2 / sigma^2)
    
    Weight for non-neighbors:
    W[i, j] = 0
    
    The matrix is made symmetric: W[i, j] = max(W[i, j], W[j, i])
    
    Args:
        X: Input data matrix of shape (n_samples, n_features)
        neighbors: Neighbor indices matrix from find_neighbors()
        sigma: Bandwidth parameter for the Gaussian kernel (default=1.0)
    
    Returns:
        Sparse symmetric weight matrix of shape (n_samples, n_samples)
    """
    n_samples = X.shape[0]
    W = np.zeros((n_samples, n_samples))
    
    # For each point i
    for i in range(n_samples):
        # For each neighbor j of point i
        for j in neighbors[i]:
            # Compute Euclidean distance
            distance = np.linalg.norm(X[i] - X[j])
            
            # Apply Gaussian kernel: exp(-distance^2 / sigma^2)
            weight = np.exp(-(distance ** 2) / (sigma ** 2))
            
            W[i, j] = weight
    
    # Make the weight matrix symmetric
    # W_symmetric[i, j] = max(W[i, j], W[j, i])
    W = np.maximum(W, W.T)
    
    return W


def build_degree_matrix(W: np.ndarray) -> np.ndarray:
    """
    Build the diagonal degree matrix D.
    
    The degree of point i is the sum of all weights connected to it:
    D[i, i] = sum_j W[i, j]
    
    All off-diagonal elements are zero.
    
    Args:
        W: Weight matrix of shape (n_samples, n_samples)
    
    Returns:
        Diagonal degree matrix of shape (n_samples, n_samples)
    """
    n_samples = W.shape[0]
    D = np.zeros((n_samples, n_samples))
    
    # Compute the degree (sum of weights) for each point
    degrees = np.sum(W, axis=1)
    
    # Place degrees on the diagonal
    np.fill_diagonal(D, degrees)
    
    return D


def build_laplacian(D: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Build the graph Laplacian matrix.
    
    The Laplacian is defined as:
    L = D - W
    
    where D is the degree matrix and W is the weight matrix.
    
    The Laplacian is a symmetric positive semi-definite matrix that
    encodes the local geometry of the data manifold.
    
    Args:
        D: Degree matrix of shape (n_samples, n_samples)
        W: Weight matrix of shape (n_samples, n_samples)
    
    Returns:
        Graph Laplacian matrix of shape (n_samples, n_samples)
    """
    L = D - W
    return L


def laplacian_eigenmaps(X: np.ndarray, k: int, d: int, 
                       sigma: float = 1.0) -> np.ndarray:
    """
    Perform Laplacian Eigenmaps dimensionality reduction.
    
    This function reduces the dimensionality of data X from n_features to d
    by finding the low-dimensional manifold structure using spectral analysis.
    
    Algorithm steps:
    1. Compute pairwise distances between all points
    2. Find k nearest neighbors for each point
    3. Build weight matrix using Gaussian kernel on neighbors
    4. Build degree matrix
    5. Construct graph Laplacian L = D - W
    6. Compute eigenvectors of L
    7. Sort eigenvectors by eigenvalues
    8. Ignore the eigenvector with eigenvalue 0 (constant eigenvector)
    9. Use the next d smallest eigenvectors as the embedding
    
    Args:
        X: Input data of shape (n_samples, n_features)
        k: Number of nearest neighbors (typically 5-30)
        d: Target dimensionality (typically 2 or 3)
        sigma: Bandwidth parameter for Gaussian kernel (default=1.0)
    
    Returns:
        Low-dimensional embedding Y of shape (n_samples, d)
    """
    n_samples = X.shape[0]
    
    print("Step 1: Computing pairwise distance matrix...")
    distance_matrix = compute_distance_matrix(X)
    
    print("Step 2: Finding k nearest neighbors...")
    neighbors = find_neighbors(distance_matrix, k)
    
    print("Step 3: Building weight matrix...")
    W = build_weight_matrix(X, neighbors, sigma=sigma)
    
    print("Step 4: Building degree matrix...")
    D = build_degree_matrix(W)
    
    print("Step 5: Constructing graph Laplacian...")
    L = build_laplacian(D, W)
    
    print("Step 6: Computing eigenvectors and eigenvalues...")
    # Use numpy.linalg.eigh for symmetric matrix eigendecomposition
    # eigh returns (eigenvalues, eigenvectors)
    # eigenvalues are in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    print("Step 7: Selecting embedding dimensions...")
    # The first eigenvalue is 0 (or very close to 0) with eigenvector = constant
    # We skip it and use the next d smallest eigenvectors
    # Eigenvectors are columns of the eigenvectors matrix
    Y = eigenvectors[:, 1:d+1]
    
    return Y

