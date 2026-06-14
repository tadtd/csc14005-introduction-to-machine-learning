"""
Evaluation Metrics for Tensor Decomposition
=============================================

Provides quantitative metrics to assess the quality of tensor
decompositions. All metrics are computed from actual decomposition
results — no synthetic or fabricated values.

Metrics Included
----------------
1. Relative Error: ||X - X̂||_F / ||X||_F
2. Fit Score: 1 - relative_error (higher is better)
3. Core Consistency Diagnostic (CORCONDIA): for rank selection
4. Factor Similarity (Tucker Congruence Coefficient)
5. Convergence Analysis

References
----------
- Bro, R., & Kiers, H. A. (2003). A new efficient method for
  determining the number of components in PARAFAC models.
  Journal of Chemometrics, 17(5), 274-286.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from .cp_als import reconstruct, unfold, khatri_rao_fast


# =============================================================================
# 1. RELATIVE ERROR
# =============================================================================

def relative_error(original, reconstructed):
    """
    Compute relative reconstruction error.

    Sai số tương đối đo lường mức độ tái tạo tensor gốc:

        RE = ||X - X̂||_F / ||X||_F

    Giá trị:
    - RE = 0: tái tạo hoàn hảo
    - RE = 1: tái tạo kém (X̂ = 0)
    - RE > 1: tái tạo tệ hơn tensor zero

    Parameters
    ----------
    original : np.ndarray
        Original tensor X.
    reconstructed : np.ndarray
        Reconstructed tensor X̂.

    Returns
    -------
    float
        Relative error ∈ [0, ∞).

    Examples
    --------
    >>> X = np.random.rand(3, 4, 5)
    >>> X_hat = X + 0.01 * np.random.randn(3, 4, 5)
    >>> re = relative_error(X, X_hat)
    >>> print(f"Relative error: {re:.6f}")
    """
    X_norm = np.linalg.norm(original)
    if X_norm == 0:
        return 0.0 if np.linalg.norm(reconstructed) == 0 else np.inf

    return np.linalg.norm(original - reconstructed) / X_norm


# =============================================================================
# 2. FIT SCORE
# =============================================================================

def fit_score(original, reconstructed):
    """
    Compute fit score (complement of relative error).

    Fit score = 1 - relative_error

    Giá trị:
    - Fit = 1.0: tái tạo hoàn hảo
    - Fit = 0.0: tái tạo rất kém
    - Fit < 0: tệ hơn dự đoán zero

    Parameters
    ----------
    original : np.ndarray
        Original tensor.
    reconstructed : np.ndarray
        Reconstructed tensor.

    Returns
    -------
    float
        Fit score ∈ (-∞, 1].
    """
    return 1.0 - relative_error(original, reconstructed)


def fit_from_factors(tensor, factors, weights=None):
    """
    Compute fit score directly from CP factors (memory-efficient).

    Tính fit mà không cần reconstruct toàn bộ tensor.
    Sử dụng công thức:
        ||X - X̂||²_F = ||X||²_F - 2⟨X, X̂⟩ + ||X̂||²_F

    Parameters
    ----------
    tensor : np.ndarray
        Original tensor.
    factors : list of np.ndarray
        Factor matrices.
    weights : np.ndarray or None
        Component weights.

    Returns
    -------
    float
        Fit score.
    """
    from .cp_als import mttkrp

    X_norm = np.linalg.norm(tensor)
    if X_norm == 0:
        return 0.0

    N = len(factors)
    R = factors[0].shape[1]

    if weights is None:
        weights = np.ones(R)

    # ⟨X, X̂⟩
    M = mttkrp(tensor, factors, 0)
    inner = np.sum(M * factors[0] * weights[np.newaxis, :])

    # ||X̂||²_F = wᵀ · (⊛_n G^(n)) · w
    gram_hadamard = np.ones((R, R))
    for n in range(N):
        gram_hadamard *= (factors[n].T @ factors[n])
    X_hat_norm_sq = weights @ gram_hadamard @ weights

    residual_sq = max(X_norm**2 - 2 * inner + X_hat_norm_sq, 0.0)
    return 1.0 - np.sqrt(residual_sq) / X_norm


# =============================================================================
# 3. CORE CONSISTENCY DIAGNOSTIC (CORCONDIA)
# =============================================================================

def core_consistency(tensor, factors, weights=None):
    """
    Core Consistency Diagnostic (CORCONDIA) for rank selection.

    CORCONDIA đo lường mức độ "superdiagonal" của core tensor trong
    Tucker decomposition tương ứng. Nếu CP model đúng (rank phù hợp),
    core tensor sẽ gần với superdiagonal.

    Mathematical Definition
    -----------------------
    Given CP factors {A^(n)}, compute the Tucker core:
        G = X ×₁ A^(1)⁺ ×₂ A^(2)⁺ ×₃ ... ×_N A^(N)⁺

    where A^(n)⁺ is the pseudo-inverse of A^(n).

    The ideal core for a CP model is the superdiagonal tensor T where:
        T(r, r, ..., r) = λ_r  and  T(...) = 0  elsewhere

    CORCONDIA = 100 × (1 - Σ_{i,j,...} (G_{ij...} - T_{ij...})² / Σ_{i,j,...} T_{ij...}²)

    Interpretation
    --------------
    - CORCONDIA ≈ 100%: CP model is appropriate (rank is correct)
    - CORCONDIA ≈ 50-80%: CP model is acceptable but may be over-fit
    - CORCONDIA < 50%: Model is likely over-specified (rank too high)

    The recommended approach: increase rank until CORCONDIA drops below ~80%.

    Parameters
    ----------
    tensor : np.ndarray
        Original tensor (order 3 recommended).
    factors : list of np.ndarray
        CP factor matrices.
    weights : np.ndarray or None
        Component weights.

    Returns
    -------
    float
        Core consistency value (0-100, higher is better).

    Notes
    -----
    This implementation is most reliable for 3-way tensors.
    """
    N = len(factors)
    R = factors[0].shape[1]

    if weights is None:
        weights = np.ones(R)

    # Tính core tensor G bằng cách project tensor lên các factor matrices
    # G = X ×₁ A^(1)ᵀ ×₂ A^(2)ᵀ ... ×_N A^(N)ᵀ
    # (sử dụng pseudo-inverse thay cho transpose)

    # Bắt đầu với tensor gốc
    core = tensor.copy()

    for n in range(N):
        # Pseudo-inverse: A⁺ = (AᵀA)⁻¹Aᵀ
        A = factors[n]
        A_pinv = np.linalg.pinv(A)  # (R, I_n)

        # Mode-n product: core ×_n A⁺
        # Unfold → multiply → fold
        core_unf = unfold(core, n)
        core_unf = A_pinv @ core_unf

        # Fold back (new size along mode n is R)
        new_shape = list(core.shape)
        new_shape[n] = R
        from .cp_als import fold
        core = fold(core_unf, n, tuple(new_shape))

    # Xây dựng ideal superdiagonal tensor T
    T_ideal = np.zeros([R] * N)
    for r in range(R):
        idx = tuple([r] * N)
        T_ideal[idx] = weights[r]

    # CORCONDIA
    diff = core - T_ideal
    T_norm_sq = np.sum(T_ideal ** 2)

    if T_norm_sq == 0:
        return 0.0

    corcondia = 100.0 * (1.0 - np.sum(diff ** 2) / T_norm_sq)

    return corcondia


# =============================================================================
# 4. FACTOR SIMILARITY
# =============================================================================

def tucker_congruence(a, b):
    """
    Tucker Congruence Coefficient (TCC) between two vectors.

    TCC đo lường sự tương đồng giữa hai factor vectors, bất kể scale:

        TCC(a, b) = (aᵀb) / (||a|| · ||b||)

    Giá trị:
    - TCC = 1.0: hoàn toàn giống nhau
    - TCC = -1.0: ngược chiều hoàn toàn
    - TCC = 0: trực giao

    Quy ước (Lorenzo-Seva & ten Berge, 2006):
    - |TCC| ≥ 0.95: factor similarity is "good"
    - |TCC| ≥ 0.85: factor similarity is "fair"
    - |TCC| < 0.85: factors are considered dissimilar

    Parameters
    ----------
    a : np.ndarray
        First vector.
    b : np.ndarray
        Second vector.

    Returns
    -------
    float
        Tucker Congruence Coefficient ∈ [-1, 1].
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return np.dot(a, b) / (norm_a * norm_b)


def factor_similarity(factors_a, factors_b, greedy_permutation=True):
    """
    Compute factor similarity between two CP decompositions.

    So sánh hai kết quả CP decomposition bằng Tucker Congruence Coefficient.
    Vì thứ tự components trong CP có thể khác nhau (permutation ambiguity),
    ta cần tìm phép hoán vị tốt nhất.

    Parameters
    ----------
    factors_a : list of np.ndarray
        Factor matrices from first decomposition.
    factors_b : list of np.ndarray
        Factor matrices from second decomposition.
    greedy_permutation : bool, default=True
        If True, find best permutation of components.

    Returns
    -------
    similarities : np.ndarray, shape (R,)
        TCC for each component (after optimal permutation).
    mean_similarity : float
        Average TCC across all modes and components.
    permutation : np.ndarray
        Best permutation of components in factors_b.
    """
    N = len(factors_a)
    R = factors_a[0].shape[1]

    if greedy_permutation:
        # Tính ma trận TCC trung bình qua các mode
        avg_tcc_matrix = np.zeros((R, R))

        for n in range(N):
            for r1 in range(R):
                for r2 in range(R):
                    avg_tcc_matrix[r1, r2] += abs(
                        tucker_congruence(factors_a[n][:, r1],
                                          factors_b[n][:, r2])
                    )

        avg_tcc_matrix /= N

        # Hungarian matching: tìm permutation tối ưu
        cost_matrix = 1.0 - avg_tcc_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # col_ind chứa permutation tối ưu
        permutation = col_ind

    else:
        permutation = np.arange(R)

    # Tính TCC cho từng component theo permutation tìm được
    similarities = np.zeros(R)
    for r in range(R):
        tcc_sum = 0.0
        for n in range(N):
            tcc_sum += abs(tucker_congruence(
                factors_a[n][:, r],
                factors_b[n][:, permutation[r]]
            ))
        similarities[r] = tcc_sum / N

    mean_similarity = np.mean(similarities)

    return similarities, mean_similarity, permutation


# =============================================================================
# 5. CONVERGENCE ANALYSIS
# =============================================================================

def convergence_summary(fit_history):
    """
    Summarize convergence behavior of an iterative algorithm.

    Phân tích quá trình hội tụ từ lịch sử fit values.

    Parameters
    ----------
    fit_history : list of float
        Fit values at each iteration.

    Returns
    -------
    dict
        Summary statistics:
        - 'final_fit': last fit value
        - 'n_iter': number of iterations
        - 'initial_fit': first fit value
        - 'improvement': final - initial fit
        - 'rate_of_change': list of |fit[t] - fit[t-1]| / |fit[t-1]|
        - 'stalled_at': iteration where improvement became < 1e-6
    """
    if len(fit_history) == 0:
        return {
            'final_fit': None,
            'n_iter': 0,
            'initial_fit': None,
            'improvement': 0,
            'rate_of_change': [],
            'stalled_at': None
        }

    fit_history = np.array(fit_history)

    # Rate of change
    roc = []
    stalled_at = None
    for t in range(1, len(fit_history)):
        if abs(fit_history[t-1]) > 0:
            change = abs(fit_history[t] - fit_history[t-1]) / abs(fit_history[t-1])
        else:
            change = abs(fit_history[t] - fit_history[t-1])
        roc.append(change)

        if stalled_at is None and change < 1e-6:
            stalled_at = t

    return {
        'final_fit': float(fit_history[-1]),
        'n_iter': len(fit_history),
        'initial_fit': float(fit_history[0]),
        'improvement': float(fit_history[-1] - fit_history[0]),
        'rate_of_change': roc,
        'stalled_at': stalled_at
    }


# =============================================================================
# 6. MULTI-RANK EVALUATION
# =============================================================================

def evaluate_ranks(tensor, ranks, decompose_fn, **kwargs):
    """
    Evaluate CP decomposition across multiple ranks.

    Chạy decomposition với nhiều giá trị rank khác nhau và thu thập
    các metrics để hỗ trợ chọn rank phù hợp.

    LƯU Ý: Hàm này thực sự CHẠY decomposition cho mỗi rank.
    Kết quả phải được tính từ code, không được bịa.

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor.
    ranks : list of int
        List of ranks to evaluate.
    decompose_fn : callable
        Decomposition function (e.g., cp_als or cp_opt).
        Must return (weights, factors, info).
    **kwargs
        Additional arguments to decompose_fn.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'ranks': list of ranks evaluated
        - 'fits': list of final fit values
        - 'errors': list of relative errors
        - 'corcondia': list of CORCONDIA values
        - 'n_iters': list of iteration counts
        - 'all_results': list of (weights, factors, info) tuples
    """
    results = {
        'ranks': [],
        'fits': [],
        'errors': [],
        'corcondia': [],
        'n_iters': [],
        'all_results': []
    }

    for r in ranks:
        print(f"  Evaluating rank {r}...", end=" ")

        weights, factors, info = decompose_fn(tensor, r, **kwargs)

        # Tính metrics
        X_hat = reconstruct(factors, weights)
        re = relative_error(tensor, X_hat)
        fs = 1.0 - re
        cc = core_consistency(tensor, factors, weights)

        n_iter = info.get('n_iter', info.get('n_iter', 0))

        results['ranks'].append(r)
        results['fits'].append(fs)
        results['errors'].append(re)
        results['corcondia'].append(cc)
        results['n_iters'].append(n_iter)
        results['all_results'].append((weights, factors, info))

        print(f"fit={fs:.4f}, error={re:.4f}, corcondia={cc:.1f}%")

    return results
