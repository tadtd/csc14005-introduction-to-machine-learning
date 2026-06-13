"""
CP-ALS: Canonical Polyadic Decomposition via Alternating Least Squares
=======================================================================

Implemented from scratch using NumPy only (no tensorly CP calls).

Mathematical Background
-----------------------
A tensor X ∈ ℝ^{I₁ × I₂ × ... × I_N} is approximated by a sum of
rank-one tensors:

    X ≈ Σ_{r=1}^{R} λ_r · a_r^(1) ∘ a_r^(2) ∘ ... ∘ a_r^(N)

where:
- R is the CP rank (number of components)
- λ_r are scalar weights
- a_r^(n) ∈ ℝ^{I_n} are the factor vectors for mode n
- ∘ denotes the outer product

The factor vectors for each mode are collected into factor matrices:
    A^(n) = [a_1^(n), a_2^(n), ..., a_R^(n)] ∈ ℝ^{I_n × R}

ALS Algorithm
-------------
ALS fixes all factor matrices except one and solves for the optimal
update of that one matrix. Cycling through all modes constitutes one
iteration. For mode n, the update is:

    A^(n) ← X_(n) · (⊙_{k≠n} A^(k)) · (⊛_{k≠n} A^(k)ᵀA^(k))^{-1}

where:
- X_(n) is the mode-n unfolding (matricization) of X
- ⊙ is the Khatri-Rao product (column-wise Kronecker)
- ⊛ is the Hadamard (element-wise) product

Key Operations
--------------
1. unfold(X, n): Mode-n matricization
2. khatri_rao(A, B): Khatri-Rao product
3. mttkrp(X, factors, n): Matricized Tensor Times Khatri-Rao Product
4. reconstruct(factors, weights): Build tensor from factors

References
----------
- Kolda, T. G., & Bader, B. W. (2009). Tensor decompositions and
  applications. SIAM Review, 51(3), 455-500.
- Bro, R. (1997). PARAFAC. Tutorial and applications.
  Chemometrics and Intelligent Laboratory Systems, 38(2), 149-171.
"""

import numpy as np
from numpy.linalg import pinv
import warnings


# =============================================================================
# 1. MODE-N UNFOLDING (MATRICIZATION)
# =============================================================================

def unfold(tensor, mode):
    """
    Mode-n unfolding (matricization) of a tensor.

    Chuyển tensor N chiều thành ma trận 2 chiều bằng cách "mở" theo mode n.

    Mathematical Definition
    -----------------------
    Given X ∈ ℝ^{I₁ × I₂ × ... × I_N}, the mode-n unfolding X_(n) is a
    matrix of size I_n × (I₁·I₂·...·I_{n-1}·I_{n+1}·...·I_N).

    The element X(i₁, i₂, ..., i_N) maps to X_(n)(i_n, j) where:
        j = 1 + Σ_{k=1, k≠n}^{N} (i_k - 1) · J_k
        J_k = Π_{m=1, m≠n}^{k-1} I_m

    Implementation
    --------------
    We use np.moveaxis to bring mode n to the front, then reshape to 2D.
    This is equivalent to the standard mode-n unfolding definition.

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor of arbitrary order (≥ 2).
    mode : int
        The mode along which to unfold (0-indexed).

    Returns
    -------
    np.ndarray
        The mode-n unfolded matrix of shape (I_n, Π_{k≠n} I_k).

    Examples
    --------
    >>> X = np.arange(24).reshape(3, 4, 2)
    >>> X_0 = unfold(X, 0)
    >>> X_0.shape
    (3, 8)
    >>> X_1 = unfold(X, 1)
    >>> X_1.shape
    (4, 6)
    """
    if mode < 0 or mode >= tensor.ndim:
        raise ValueError(
            f"mode={mode} is out of range for tensor with {tensor.ndim} dimensions"
        )

    # Đưa mode n ra phía trước, sau đó reshape thành ma trận 2D
    # moveaxis(tensor, mode, 0): chuyển axis 'mode' đến vị trí 0
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def fold(unfolded, mode, shape):
    """
    Fold (refold) a mode-n unfolded matrix back into a tensor.

    Đây là phép nghịch đảo của unfold: chuyển ma trận 2D trở lại tensor N chiều.

    Parameters
    ----------
    unfolded : np.ndarray
        Mode-n unfolded matrix of shape (I_n, Π_{k≠n} I_k).
    mode : int
        The mode that was unfolded (0-indexed).
    shape : tuple of int
        The original tensor shape.

    Returns
    -------
    np.ndarray
        The refolded tensor with the original shape.

    Examples
    --------
    >>> X = np.arange(24).reshape(3, 4, 2)
    >>> X_unf = unfold(X, 1)
    >>> X_refolded = fold(X_unf, 1, (3, 4, 2))
    >>> np.allclose(X, X_refolded)
    True
    """
    # Xây dựng shape trung gian: mode n ở vị trí đầu
    full_shape = list(shape)
    mode_size = full_shape.pop(mode)
    new_shape = [mode_size] + full_shape

    # Reshape về dạng tensor (mode n ở đầu), rồi moveaxis trả mode n về vị trí cũ
    return np.moveaxis(unfolded.reshape(new_shape), 0, mode)


# =============================================================================
# 2. KHATRI-RAO PRODUCT
# =============================================================================

def khatri_rao(matrices):
    """
    Khatri-Rao product (column-wise Kronecker product) of a list of matrices.

    Mathematical Definition
    -----------------------
    Given matrices A ∈ ℝ^{I×R} and B ∈ ℝ^{J×R}, the Khatri-Rao product
    A ⊙ B ∈ ℝ^{IJ×R} is defined column-wise:

        (A ⊙ B)_{:,r} = a_{:,r} ⊗ b_{:,r}

    where ⊗ is the Kronecker product of column vectors:
        a ⊗ b = [a₁b₁, a₁b₂, ..., a₁b_J, a₂b₁, ..., a_I·b_J]ᵀ

    For multiple matrices [A^(1), A^(2), ..., A^(N)]:
        A^(1) ⊙ A^(2) ⊙ ... ⊙ A^(N)

    is computed by sequential application from right to left (or left to right,
    since the operation is associative).

    Parameters
    ----------
    matrices : list of np.ndarray
        List of matrices, all must have the same number of columns (R).
        Shapes: [(I₁, R), (I₂, R), ..., (I_N, R)]

    Returns
    -------
    np.ndarray
        Khatri-Rao product of shape (I₁·I₂·...·I_N, R).

    Raises
    ------
    ValueError
        If matrices have different numbers of columns.

    Examples
    --------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5, 6], [7, 8]])
    >>> kr = khatri_rao([A, B])
    >>> kr.shape
    (4, 2)
    >>> # Column 0: [1*5, 1*7, 3*5, 3*7] = [5, 7, 15, 21]
    >>> # Column 1: [2*6, 2*8, 4*6, 4*8] = [12, 16, 24, 32]
    """
    if len(matrices) < 2:
        raise ValueError("Khatri-Rao product requires at least 2 matrices")

    # Kiểm tra tất cả ma trận có cùng số cột R
    R = matrices[0].shape[1]
    for i, m in enumerate(matrices):
        if m.shape[1] != R:
            raise ValueError(
                f"All matrices must have the same number of columns. "
                f"Matrix 0 has {R} columns, but matrix {i} has {m.shape[1]} columns."
            )

    # Tính Khatri-Rao product tuần tự từ ma trận cuối
    # Bắt đầu với ma trận cuối cùng
    result = matrices[-1]

    # Nhân ngược từ cuối lên đầu
    for i in range(len(matrices) - 2, -1, -1):
        n_rows_a = matrices[i].shape[0]
        n_rows_b = result.shape[0]

        # Với mỗi cột r, tính Kronecker product của 2 vector cột
        # Sử dụng broadcasting: a[:, np.newaxis, r] * b[np.newaxis, :, r]
        # rồi reshape thành vector
        new_result = np.zeros((n_rows_a * n_rows_b, R))
        for r in range(R):
            # Kronecker product của 2 vector cột
            # a_col ⊗ b_col = a_col[:, None] * b_col[None, :]
            # rồi flatten (row-major = C order)
            new_result[:, r] = np.outer(matrices[i][:, r], result[:, r]).ravel()

        result = new_result

    return result


def khatri_rao_fast(matrices):
    """
    Fast Khatri-Rao product using broadcasting (optimized version).

    Tương tự khatri_rao() nhưng sử dụng broadcasting thay vì vòng lặp
    trên từng cột, cho tốc độ nhanh hơn đáng kể.

    Parameters
    ----------
    matrices : list of np.ndarray
        Same as khatri_rao().

    Returns
    -------
    np.ndarray
        Same as khatri_rao().
    """
    if len(matrices) < 2:
        raise ValueError("Khatri-Rao product requires at least 2 matrices")

    R = matrices[0].shape[1]

    result = matrices[-1]
    for i in range(len(matrices) - 2, -1, -1):
        a = matrices[i]
        # a has shape (I_a, R), result has shape (I_b, R)
        # We want output of shape (I_a * I_b, R)
        # Use einsum: 'ar,br->abr' then reshape to (I_a*I_b, R)
        result = np.einsum('ir,jr->ijr', a, result).reshape(-1, R)

    return result


# =============================================================================
# 3. MTTKRP (Matricized Tensor Times Khatri-Rao Product)
# =============================================================================

def mttkrp(tensor, factors, mode):
    """
    Matricized Tensor Times Khatri-Rao Product.
    For 3-way tensors, uses memory-efficient np.einsum directly.
    """
    N = tensor.ndim
    if N == 3:
        if mode == 0:
            return np.einsum('ijk,jr,kr->ir', tensor, factors[1], factors[2])
        elif mode == 1:
            return np.einsum('ijk,ir,kr->jr', tensor, factors[0], factors[2])
        elif mode == 2:
            return np.einsum('ijk,ir,jr->kr', tensor, factors[0], factors[1])

    R = factors[0].shape[1]
    kr_matrices = []
    for n in range(N):
        if n != mode:
            kr_matrices.append(factors[n])

    kr = khatri_rao_fast(kr_matrices)
    X_unf = unfold(tensor, mode)
    return X_unf @ kr


# =============================================================================
# 4. TENSOR RECONSTRUCTION
# =============================================================================

def reconstruct(factors, weights=None):
    """
    Reconstruct a tensor from CP factor matrices and weights.

    Mathematical Definition
    -----------------------
    X̂ = Σ_{r=1}^{R} λ_r · a_r^(1) ∘ a_r^(2) ∘ ... ∘ a_r^(N)

    Trong đó:
    - λ_r là trọng số của component r (nếu weights=None thì λ_r = 1)
    - a_r^(n) là cột r của factor matrix A^(n)
    - ∘ là outer product

    Implementation
    --------------
    Thay vì tính từng rank-one tensor rồi cộng lại, ta sử dụng
    Khatri-Rao product và fold lại:

    X̂_(n) = A^(n) · diag(λ) · (⊙_{k≠n} A^(k))ᵀ

    Parameters
    ----------
    factors : list of np.ndarray
        List of factor matrices [A^(1), ..., A^(N)].
    weights : np.ndarray or None
        Weight vector λ ∈ ℝ^R. If None, all weights are 1.

    Returns
    -------
    np.ndarray
        Reconstructed tensor with shape (I₁, I₂, ..., I_N).

    Examples
    --------
    >>> A = [np.random.rand(3, 2), np.random.rand(4, 2), np.random.rand(5, 2)]
    >>> X_hat = reconstruct(A)
    >>> X_hat.shape
    (3, 4, 5)
    """
    N = len(factors)
    R = factors[0].shape[1]
    shape = tuple(f.shape[0] for f in factors)

    if weights is None:
        weights = np.ones(R)

    # Sử dụng mode-0 để reconstruct:
    # X̂_(0) = A^(0) · diag(λ) · (A^(N-1) ⊙ ... ⊙ A^(1))ᵀ
    kr_others = khatri_rao_fast([factors[n] for n in range(1, N)])

    # A^(0) · diag(λ) = A^(0) * λ[np.newaxis, :]  (broadcasting)
    A_weighted = factors[0] * weights[np.newaxis, :]

    # Ma trận mode-0 unfolded của X̂
    X_hat_unf = A_weighted @ kr_others.T

    # Fold lại thành tensor
    return fold(X_hat_unf, 0, shape)


# =============================================================================
# 5. CP-ALS ALGORITHM
# =============================================================================

def _normalize_factors(factors):
    """
    Normalize factor matrices to unit columns and extract weights.

    Chuẩn hóa mỗi cột của factor matrix về norm 1, lưu norm vào weights.

    Điều này giúp:
    1. Tránh vấn đề scale ambiguity (A^(n) có thể scale tùy ý nếu
       scale ngược được bù ở mode khác)
    2. Cải thiện ổn định số học

    Parameters
    ----------
    factors : list of np.ndarray
        Factor matrices to normalize.

    Returns
    -------
    weights : np.ndarray
        Column norms (product across all modes).
    factors : list of np.ndarray
        Normalized factor matrices (unit columns).
    """
    R = factors[0].shape[1]
    weights = np.ones(R)

    normalized = []
    for A in factors:
        norms = np.linalg.norm(A, axis=0)  # norm của mỗi cột
        # Tránh chia cho 0
        norms = np.maximum(norms, 1e-12)
        weights *= norms
        normalized.append(A / norms[np.newaxis, :])

    return weights, normalized


def cp_als(tensor, rank, max_iter=200, tol=1e-8, random_state=None,
           normalize_every=1, init='random', verbose=False):
    """
    CP Decomposition via Alternating Least Squares (from scratch).

    Thuật toán CP-ALS tìm xấp xỉ rank-R tốt nhất cho tensor bằng cách
    luân phiên tối ưu từng factor matrix.

    Algorithm
    ---------
    Input: Tensor X ∈ ℝ^{I₁×I₂×...×I_N}, rank R
    Output: weights λ, factor matrices {A^(n)}

    1. Khởi tạo ngẫu nhiên A^(1), ..., A^(N)
    2. Repeat until convergence:
       For n = 1, ..., N:
           a. V = ⊛_{k≠n} (A^(k)ᵀ A^(k))     [Hadamard of Gram matrices]
           b. M = MTTKRP(X, factors, n)         [Right-hand side]
           c. A^(n) = M · V^{-1}                [Least squares update]
       Normalize factors, compute fit
    3. Return λ, {A^(n)}, fit_history

    Convergence Criterion
    ---------------------
    |fit^(t) - fit^(t-1)| / |fit^(t-1)| < tol

    where fit = 1 - ||X - X̂||_F / ||X||_F

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor of order ≥ 2.
    rank : int
        CP rank (number of components R).
    max_iter : int, default=200
        Maximum number of ALS iterations.
    tol : float, default=1e-8
        Convergence tolerance on relative fit change.
    random_state : int or None
        Random seed for reproducibility.
    normalize_every : int, default=1
        Normalize factors every N iterations (1 = every iteration).
    init : str, default='random'
        Initialization method: 'random' or 'svd'.
    verbose : bool, default=False
        If True, print convergence information.

    Returns
    -------
    weights : np.ndarray
        Component weights λ ∈ ℝ^R.
    factors : list of np.ndarray
        Factor matrices [A^(1), ..., A^(N)].
    info : dict
        Dictionary with keys:
        - 'fit_history': list of fit values per iteration
        - 'n_iter': number of iterations performed
        - 'converged': whether convergence was reached

    Examples
    --------
    >>> X = np.random.rand(10, 8, 6)
    >>> weights, factors, info = cp_als(X, rank=3, max_iter=100)
    >>> print(f"Final fit: {info['fit_history'][-1]:.6f}")
    >>> print(f"Converged: {info['converged']}")
    """
    # Kiểm tra đầu vào
    if tensor.ndim < 2:
        raise ValueError(f"Tensor must have at least 2 dimensions, got {tensor.ndim}")
    if rank < 1:
        raise ValueError(f"Rank must be ≥ 1, got {rank}")

    N = tensor.ndim
    shape = tensor.shape
    rng = np.random.RandomState(random_state)

    # Tính ||X||_F một lần (dùng cho fit)
    X_norm = np.linalg.norm(tensor)
    if X_norm == 0:
        warnings.warn("Input tensor has zero norm. Returning zero factors.")
        factors = [np.zeros((s, rank)) for s in shape]
        weights = np.zeros(rank)
        info = {'fit_history': [0.0], 'n_iter': 0, 'converged': True}
        if tensor.ndim == 3:
            return weights, factors[0], factors[1], factors[2]
        return weights, factors, info

    # =========================================================================
    # Bước 1: Khởi tạo
    # =========================================================================
    if init == 'svd':
        factors = []
        for mode in range(N):
            X_unf = unfold(tensor, mode)
            # Dùng SVD để tìm rank thành phần chính của X_unf
            U, _, _ = np.linalg.svd(X_unf, full_matrices=False)
            factors.append(U[:, :rank])
    else:
        # Mỗi factor matrix A^(n) ∈ ℝ^{I_n × R} được khởi tạo từ N(0, 1)
        factors = [rng.randn(s, rank) for s in shape]

    # Chuẩn hóa ban đầu
    weights, factors = _normalize_factors(factors)

    fit_history = []
    prev_fit = 0.0

    if verbose:
        print(f"CP-ALS: tensor shape={shape}, rank={rank}")
        print(f"{'Iter':>6s} {'Fit':>12s} {'Fit Change':>14s}")
        print("-" * 36)

    # =========================================================================
    # Bước 2: Vòng lặp ALS chính
    # =========================================================================
    for iteration in range(max_iter):

        # Cập nhật từng mode
        for mode in range(N):
            # -----------------------------------------------------------------
            # Bước 2a: Tính V = ⊛_{k≠n} (A^(k)ᵀ A^(k))
            # -----------------------------------------------------------------
            # V ∈ ℝ^{R×R} là Hadamard product của các Gram matrices
            # Bắt đầu với ma trận toàn 1 (identity cho Hadamard product)
            V = np.ones((rank, rank))
            for k in range(N):
                if k != mode:
                    # Gram matrix: A^(k)ᵀ A^(k) ∈ ℝ^{R×R}
                    V *= (factors[k].T @ factors[k])

            # -----------------------------------------------------------------
            # Bước 2b: Tính MTTKRP
            # -----------------------------------------------------------------
            M = mttkrp(tensor, factors, mode)

            # -----------------------------------------------------------------
            # Bước 2c: Cập nhật A^(n) = M · V^{-1}
            # -----------------------------------------------------------------
            factors[mode] = M @ np.linalg.pinv(V)

        # =====================================================================
        # Bước 2d: Normalize factors
        # =====================================================================
        if (iteration + 1) % normalize_every == 0:
            weights, factors = _normalize_factors(factors)

        # =====================================================================
        # Bước 2e: Tính fit và kiểm tra hội tụ
        # =====================================================================
        # Fit = 1 - ||X - X̂||_F / ||X||_F
        # Để tránh reconstruct toàn bộ tensor (tốn bộ nhớ),
        # ta tính ||X - X̂||²_F = ||X||²_F - 2⟨X, X̂⟩ + ||X̂||²_F

        # ⟨X, X̂⟩ = trace(X_(0)ᵀ · A^(0) · diag(λ) · KR_othersᵀ)
        #          = Σ_r λ_r · Σ_{i} [X_(0) · kr_others]_{i,r} · A^(0)_{i,r}
        # Tức là: trace(M_0ᵀ · A^(0) · diag(λ))
        # với M_0 = MTTKRP(X, factors, 0)

        # Cách đơn giản hơn: sử dụng Gram matrices
        # ||X̂||²_F = Σ_{r,s} λ_r λ_s · Π_n (A^(n)ᵀA^(n))_{r,s}
        #          = λᵀ · (⊛_n G^(n)) · λ
        # với G^(n) = A^(n)ᵀ A^(n)

        gram_hadamard = np.ones((rank, rank))
        for n in range(N):
            gram_hadamard *= (factors[n].T @ factors[n])

        X_hat_norm_sq = weights @ gram_hadamard @ weights

        # ⟨X, X̂⟩ = Σ_r λ_r · [MTTKRP(X, factors, 0)]_{:,r}ᵀ · A^(0)_{:,r}
        # = Σ_r λ_r · (M_0 * A^(0))_{sum over rows, col r}
        M_0 = mttkrp(tensor, factors, 0)
        inner_product = np.sum(M_0 * factors[0] * weights[np.newaxis, :])

        residual_sq = X_norm**2 - 2 * inner_product + X_hat_norm_sq

        # Xử lý lỗi số (residual_sq có thể âm nhẹ do floating point)
        residual_sq = max(residual_sq, 0.0)

        fit = 1.0 - np.sqrt(residual_sq) / X_norm
        fit_history.append(fit)

        if verbose:
            fit_change = abs(fit - prev_fit)
            print(f"{iteration + 1:6d} {fit:12.8f} {fit_change:14.2e}")

        # Kiểm tra hội tụ
        if iteration > 0:
            fit_change = abs(fit - prev_fit)
            if prev_fit != 0:
                relative_change = fit_change / abs(prev_fit)
            else:
                relative_change = fit_change

            if relative_change < tol:
                if verbose:
                    print(f"\nConverged at iteration {iteration + 1}")
                info = {
                    'fit_history': fit_history,
                    'n_iter': iteration + 1,
                    'converged': True
                }
                if tensor.ndim == 3:
                    return weights, factors[0], factors[1], factors[2]
                return weights, factors, info

        prev_fit = fit

    # Không hội tụ
    if verbose:
        print(f"\nMax iterations ({max_iter}) reached without convergence")

    info = {
        'fit_history': fit_history,
        'n_iter': max_iter,
        'converged': False
    }
    if tensor.ndim == 3:
        return weights, factors[0], factors[1], factors[2]
    return weights, factors, info


# =============================================================================
# 6. UTILITY FUNCTIONS
# =============================================================================

def compute_gram_matrices(factors):
    """
    Compute Gram matrices A^(n)ᵀ A^(n) for all factor matrices.

    Parameters
    ----------
    factors : list of np.ndarray
        Factor matrices.

    Returns
    -------
    list of np.ndarray
        List of Gram matrices, each of shape (R, R).
    """
    return [A.T @ A for A in factors]


def hadamard_product_grams(grams, skip_mode=None):
    """
    Compute Hadamard (element-wise) product of Gram matrices,
    optionally skipping one mode.

    Parameters
    ----------
    grams : list of np.ndarray
        Gram matrices.
    skip_mode : int or None
        Mode to skip (for ALS update).

    Returns
    -------
    np.ndarray
        Hadamard product of Gram matrices.
    """
    R = grams[0].shape[0]
    result = np.ones((R, R))
    for n, G in enumerate(grams):
        if n != skip_mode:
            result *= G
    return result
