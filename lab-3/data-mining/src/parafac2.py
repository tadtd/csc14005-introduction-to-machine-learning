"""
PARAFAC2: Parallel Factor Analysis 2
======================================

PARAFAC2 is a generalization of CP decomposition that handles datasets
where the "subjects" (slices along one mode) may have different numbers
of observations along another mode.

Mathematical Background
-----------------------
Standard CP requires a regular tensor X ∈ ℝ^{I×J×K}. In many real
applications (e.g., EEG where trials have different lengths, or
chromatography where elution profiles differ), we have a collection of
matrices {X_k}_{k=1}^{K} where X_k ∈ ℝ^{I_k × J}.

Note: I_k can vary across slices k (different number of time points per
subject/trial), but J must be constant (same channels/variables).

PARAFAC2 Model
--------------
For each slice k:
    X_k ≈ U_k · S_k · Vᵀ

where:
- U_k ∈ ℝ^{I_k × R}: slice-specific loadings (can have different I_k)
- S_k ∈ ℝ^{R × R}: diagonal matrix of component weights for slice k
- V ∈ ℝ^{J × R}: shared loadings across all slices

PARAFAC2 Constraint:
    U_kᵀ · U_k = Φ  for all k

This means the cross-product of U_k is constant across slices, even
though U_k itself varies. This makes the model identifiable.

Procrustes Rotation
-------------------
The constraint is enforced via the orthogonal Procrustes problem:
    U_k = P_k · H

where:
- P_k ∈ ℝ^{I_k × R}: orthogonal matrix (from SVD)
- H ∈ ℝ^{R × R}: constant matrix such that HᵀH = Φ

PARAFAC2-ALS Algorithm
----------------------
1. Initialize V, {S_k}, H
2. For each slice k:
   a. Compute Z_k = X_kᵀ · P_k  (or use SVD to find P_k)
   b. Stack Z_k to form a regular tensor
3. Apply standard CP-ALS to the stacked tensor
4. Recover P_k via SVD of X_k · V · S_k
5. Update H, V, S_k
6. Repeat until convergence

References
----------
- Harshman, R. A. (1972). PARAFAC2: Mathematical and technical notes.
  UCLA Working Papers in Phonetics, 22, 30-44.
- Kiers, H. A., Ten Berge, J. M., & Bro, R. (1999). PARAFAC2—Part I.
  A direct fitting algorithm for the PARAFAC2 model.
  Journal of Chemometrics, 13, 275-294.
"""

import numpy as np
from numpy.linalg import svd, norm
import warnings


# =============================================================================
# 1. PROCRUSTES ROTATION
# =============================================================================

def _procrustes_rotation(X, target):
    """
    Solve the orthogonal Procrustes problem.

    Tìm ma trận trực giao Q sao cho ||X·Q - target||_F là nhỏ nhất.

    Mathematical Solution
    ---------------------
    Q = V · Uᵀ

    where U·Σ·Vᵀ = SVD(Xᵀ · target)

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Source matrix.
    target : np.ndarray, shape (n, p)
        Target matrix.

    Returns
    -------
    Q : np.ndarray, shape (p, p)
        Orthogonal rotation matrix.
    """
    M = X.T @ target
    U, _, Vt = svd(M, full_matrices=False)
    return U @ Vt


# =============================================================================
# 2. PARAFAC2 INITIALIZATION
# =============================================================================

def _initialize_parafac2(slices, rank, random_state=None):
    """
    Initialize PARAFAC2 factors.

    Khởi tạo các thành phần của PARAFAC2:
    - V: shared loading matrix (J × R)
    - weights: component weights for each slice (K × R)  
    - projections: orthogonal projections P_k for each slice
    - H: constraint matrix (R × R)

    Strategy: Use SVD of the "average" cross-product matrix.

    Parameters
    ----------
    slices : list of np.ndarray
        K matrices, each X_k ∈ ℝ^{I_k × J}.
    rank : int
        Number of components R.
    random_state : int or None
        Random seed.

    Returns
    -------
    V : np.ndarray, shape (J, R)
    weights : np.ndarray, shape (K, R)
    projections : list of np.ndarray
    H : np.ndarray, shape (R, R)
    """
    rng = np.random.RandomState(random_state)
    K = len(slices)
    J = slices[0].shape[1]

    # Khởi tạo V bằng SVD của ma trận cross-product trung bình
    # Σ_k X_kᵀ X_k / K ≈ V · Λ · Vᵀ
    cross_product_sum = np.zeros((J, J))
    for X_k in slices:
        cross_product_sum += X_k.T @ X_k
    cross_product_avg = cross_product_sum / K

    U_init, s_init, Vt_init = svd(cross_product_avg, full_matrices=False)
    V = U_init[:, :rank]

    # Khởi tạo H = I (identity, simplest constraint)
    H = np.eye(rank)

    # Khởi tạo weights và projections
    weights = np.ones((K, rank))
    projections = []

    for k in range(K):
        X_k = slices[k]
        I_k = X_k.shape[0]

        # Tính P_k sơ bộ bằng SVD
        if I_k >= rank:
            # P_k = U_k (từ SVD của X_k · V)
            Z_k = X_k @ V
            U_k, s_k, Vt_k = svd(Z_k, full_matrices=False)
            P_k = U_k[:, :rank] @ Vt_k[:rank, :]
            weights[k, :] = s_k[:rank]
        else:
            # Nếu I_k < rank, padding
            P_k = rng.randn(I_k, rank)
            Q, _ = np.linalg.qr(P_k)
            P_k = Q

        projections.append(P_k)

    return V, weights, projections, H


# =============================================================================
# 3. PARAFAC2-ALS MAIN ALGORITHM
# =============================================================================

def parafac2_als(slices, rank, max_iter=200, tol=1e-8, random_state=None,
                 verbose=False):
    """
    PARAFAC2 decomposition via Alternating Least Squares.

    Algorithm (Kiers et al., 1999)
    ------------------------------
    Input: {X_k}_{k=1}^K with X_k ∈ ℝ^{I_k × J}, rank R
    Output: V, {S_k}, {P_k}, H

    1. Initialize V, {S_k}, {P_k}, H
    2. Repeat until convergence:
       a. For each slice k:
          - Compute B_k = X_kᵀ · P_k · H        (projection to common space)
       b. Stack {B_k} into a 3D tensor B ∈ ℝ^{J × R × K}
       c. Update V by fitting the "stacked" model:
          - B_(1) ≈ V · (S ⊙ I_K)              (simplified ALS update)
       d. Update S_k:
          - S_k = diag(B_kᵀ · V)                (for each k)
       e. Update projections P_k:
          - M_k = X_k · V · S_k
          - U·Σ·Wᵀ = SVD(M_k)
          - P_k = U · Wᵀ                         (Procrustes)
       f. Update H (optional, can keep as identity)
    3. Compute fit, return results

    Parameters
    ----------
    slices : list of np.ndarray
        K matrices, each of shape (I_k, J). All must have the same J.
    rank : int
        Number of PARAFAC2 components.
    max_iter : int, default=200
        Maximum iterations.
    tol : float, default=1e-8
        Convergence tolerance.
    random_state : int or None
        Random seed.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    V : np.ndarray, shape (J, R)
        Shared loading matrix (common across slices).
    weights : np.ndarray, shape (K, R)
        Per-slice component weights (diagonal of S_k).
    projections : list of np.ndarray
        Per-slice projection matrices P_k.
    info : dict
        Dictionary with keys:
        - 'fit_history': list of fit values per iteration
        - 'n_iter': iterations performed
        - 'converged': convergence flag
        - 'H': constraint matrix
    """
    # Validate input
    K = len(slices)
    if K < 2:
        raise ValueError(f"PARAFAC2 requires at least 2 slices, got {K}")

    J = slices[0].shape[1]
    for k, X_k in enumerate(slices):
        if X_k.shape[1] != J:
            raise ValueError(
                f"All slices must have the same number of columns. "
                f"Slice 0 has {J} columns, but slice {k} has {X_k.shape[1]}."
            )

    # Tính tổng ||X_k||²_F
    total_norm_sq = sum(np.sum(X_k**2) for X_k in slices)
    total_norm = np.sqrt(total_norm_sq)

    if total_norm == 0:
        warnings.warn("All slices have zero norm.")
        V = np.zeros((J, rank))
        weights = np.zeros((K, rank))
        projections = [np.zeros((X_k.shape[0], rank)) for X_k in slices]
        info = {'fit_history': [0.0], 'n_iter': 0, 'converged': True, 'H': np.eye(rank)}
        return V, weights, projections, info

    # =========================================================================
    # Bước 1: Khởi tạo
    # =========================================================================
    V, weights, projections, H = _initialize_parafac2(slices, rank, random_state)

    fit_history = []
    prev_fit = -np.inf

    if verbose:
        print(f"PARAFAC2: K={K} slices, J={J}, rank={rank}")
        print(f"  Slice sizes: {[X_k.shape[0] for X_k in slices]}")

    # =========================================================================
    # Bước 2: Vòng lặp ALS
    # =========================================================================
    for iteration in range(max_iter):

        # ---------------------------------------------------------------------
        # Bước 2a-b: Project mỗi slice vào common space
        # ---------------------------------------------------------------------
        # B_k = P_kᵀ · X_k ∈ ℝ^{R × J}  (projected data)
        projected = []
        for k in range(K):
            B_k = projections[k].T @ slices[k]  # (R, J)
            projected.append(B_k)

        # Stack thành tensor B ∈ ℝ^{R × J × K}
        B = np.stack(projected, axis=2)  # (R, J, K)

        # ---------------------------------------------------------------------
        # Bước 2c: Update V bằng least squares
        # ---------------------------------------------------------------------
        # Với mỗi component r, ta fit:
        # B[r, :, k] ≈ weights[k, r] · V[:, r]
        # Tức là giải: V[:, r] = Σ_k weights[k,r] · B[r,:,k] / Σ_k weights[k,r]²

        # Hoặc dùng mode-1 unfolding: B_(1) = V · (D_weights)
        # ALS update cho V:
        # Unfold B theo mode-1 (J mode): B_(1) ∈ ℝ^{J × (R·K)}
        # Nhưng ta có thể đơn giản hơn:

        # Phương pháp trực tiếp: V_{new} từ SVD hoặc least squares
        # B[r, j, k] ≈ V[j, r] · weights[k, r]
        # → V[j, r] = Σ_k B[r, j, k] · weights[k, r] / (Σ_k weights[k, r]²)

        for r in range(rank):
            numerator = np.zeros(J)
            denominator = 0.0
            for k in range(K):
                numerator += B[r, :, k] * weights[k, r]
                denominator += weights[k, r] ** 2
            if denominator > 1e-12:
                V[:, r] = numerator / denominator
            else:
                V[:, r] = 0.0

        # Normalize V columns
        for r in range(rank):
            v_norm = np.linalg.norm(V[:, r])
            if v_norm > 1e-12:
                V[:, r] /= v_norm

        # ---------------------------------------------------------------------
        # Bước 2d: Update weights (S_k)
        # ---------------------------------------------------------------------
        for k in range(K):
            # weights[k, r] = B_k[r, :] · V[:, r] = (P_kᵀ X_k)_{r,:} · V_{:,r}
            for r in range(rank):
                weights[k, r] = projected[k][r, :] @ V[:, r]

        # ---------------------------------------------------------------------
        # Bước 2e: Update projections P_k (Procrustes)
        # ---------------------------------------------------------------------
        for k in range(K):
            # M_k = X_k · V · diag(weights[k, :])
            # P_k = nearest orthogonal matrix to M_k (Procrustes)
            M_k = slices[k] @ V * weights[k, :][np.newaxis, :]  # (I_k, R)

            # SVD: M_k = U · Σ · Wᵀ
            U_k, _, Wt_k = svd(M_k, full_matrices=False)

            # P_k = U · Wᵀ (nearest orthogonal)
            projections[k] = U_k @ Wt_k

        # ---------------------------------------------------------------------
        # Bước 2f: Tính fit
        # ---------------------------------------------------------------------
        residual_sq = 0.0
        for k in range(K):
            # X̂_k = P_k · diag(weights[k,:]) · Vᵀ
            X_hat_k = projections[k] * weights[k, :][np.newaxis, :] @ V.T
            residual_sq += np.sum((slices[k] - X_hat_k) ** 2)

        fit = 1.0 - np.sqrt(max(residual_sq, 0.0)) / total_norm
        fit_history.append(fit)

        if verbose and (iteration + 1) % 10 == 0:
            print(f"  Iter {iteration + 1:4d}: fit={fit:.8f}")

        # Kiểm tra hội tụ
        if iteration > 0:
            fit_change = abs(fit - prev_fit)
            if abs(prev_fit) > 0:
                relative_change = fit_change / abs(prev_fit)
            else:
                relative_change = fit_change

            if relative_change < tol:
                if verbose:
                    print(f"  Converged at iteration {iteration + 1}")
                info = {
                    'fit_history': fit_history,
                    'n_iter': iteration + 1,
                    'converged': True,
                    'H': H.copy()
                }
                return V, weights, projections, info

        prev_fit = fit

    if verbose:
        print(f"  Max iterations ({max_iter}) reached")

    info = {
        'fit_history': fit_history,
        'n_iter': max_iter,
        'converged': False,
        'H': H.copy()
    }

    return V, weights, projections, info


# =============================================================================
# 4. PARAFAC2 RECONSTRUCTION
# =============================================================================

def reconstruct_slice(V, weights_k, projection_k):
    """
    Reconstruct a single slice from PARAFAC2 factors.

    X̂_k = P_k · diag(weights_k) · Vᵀ

    Parameters
    ----------
    V : np.ndarray, shape (J, R)
        Shared loading matrix.
    weights_k : np.ndarray, shape (R,)
        Weights for slice k.
    projection_k : np.ndarray, shape (I_k, R)
        Projection matrix for slice k.

    Returns
    -------
    np.ndarray, shape (I_k, J)
        Reconstructed slice.
    """
    return projection_k * weights_k[np.newaxis, :] @ V.T


def reconstruct_all(V, weights, projections):
    """
    Reconstruct all slices from PARAFAC2 factors.

    Parameters
    ----------
    V : np.ndarray, shape (J, R)
    weights : np.ndarray, shape (K, R)
    projections : list of np.ndarray

    Returns
    -------
    list of np.ndarray
        Reconstructed slices.
    """
    return [reconstruct_slice(V, weights[k], projections[k])
            for k in range(len(projections))]
