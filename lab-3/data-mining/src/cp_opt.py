"""
CP-OPT: Canonical Polyadic Decomposition via Gradient-Based Optimization
=========================================================================

Uses scipy.optimize.minimize with L-BFGS-B method.

Mathematical Background
-----------------------
CP-OPT solves the same problem as CP-ALS but uses gradient-based
optimization instead of alternating least squares:

    min_{A^(1),...,A^(N)} f(A^(1),...,A^(N))
    
    where f = ||X - X̂||²_F = ||X - Σ_r a_r^(1) ∘ ... ∘ a_r^(N)||²_F

Vectorization Strategy
----------------------
scipy.optimize.minimize expects a function f(x) where x is a 1D vector.
We "vectorize" all factor matrices into a single vector:

    x = [vec(A^(1))ᵀ, vec(A^(2))ᵀ, ..., vec(A^(N))ᵀ]ᵀ

where vec(A) flattens A column-by-column (or row-by-row with C-order).

Total length of x: Σ_n (I_n × R)

Example for a 3×4×2 tensor with rank R=2:
    A^(1) ∈ ℝ^{3×2}: 6 elements
    A^(2) ∈ ℝ^{4×2}: 8 elements
    A^(3) ∈ ℝ^{2×2}: 4 elements
    x ∈ ℝ^{18}

Objective Function
------------------
f(x) = ||X||²_F - 2⟨X, X̂⟩ + ||X̂||²_F

where:
- ||X||²_F is precomputed (constant)
- ⟨X, X̂⟩ = Σ_r [MTTKRP(X, factors, n)]_{:,r}ᵀ · A^(n)_{:,r}  (for any n)
- ||X̂||²_F = Σ_{r,s} Π_n (A^(n)ᵀ A^(n))_{r,s}

Gradient
--------
∂f/∂A^(n) = -2 · MTTKRP(X, factors, n) + 2 · A^(n) · (⊛_{k≠n} A^(k)ᵀA^(k))

The gradient vector is constructed by concatenating vec(∂f/∂A^(n)) for all n.

L-BFGS-B Method
----------------
L-BFGS-B (Limited-memory BFGS with Bound constraints) is a quasi-Newton
method that approximates the Hessian using a limited number of previous
gradient evaluations. It is well-suited for large-scale optimization:
- Memory: O(m × p) where m is history size (~10), p is number of variables
- Convergence: superlinear (faster than ALS near optimum)
- No bound constraints needed here (factors are unconstrained)

Advantages over CP-ALS
-----------------------
1. Can converge faster, especially near the solution
2. Uses full gradient information (not just one mode at a time)
3. More robust to "swamp" behavior in ALS

Disadvantages
-------------
1. More expensive per iteration (full gradient computation)
2. Requires careful implementation of gradient
3. May converge to different local minimum

References
----------
- Acar, E., Dunlavy, D. M., & Kolda, T. G. (2011). A scalable
  optimization approach for fitting canonical tensor decompositions.
  Journal of Chemometrics, 25(2), 67-86.
"""

import numpy as np
from scipy.optimize import minimize
from .cp_als import unfold, khatri_rao_fast, mttkrp, reconstruct, fold


# =============================================================================
# 1. VECTORIZATION / UN-VECTORIZATION
# =============================================================================

def vectorize_factors(factors):
    """
    Flatten all factor matrices into a single 1D vector.

    Chuyển tất cả các factor matrices thành một vector 1D duy nhất
    để sử dụng với scipy.optimize.minimize.

    Layout in memory:
        x = [A^(1).ravel(), A^(2).ravel(), ..., A^(N).ravel()]

    Parameters
    ----------
    factors : list of np.ndarray
        Factor matrices [A^(1), ..., A^(N)].

    Returns
    -------
    x : np.ndarray
        1D vector chứa tất cả phần tử của các factor matrices.
    shapes : list of tuple
        Shapes của từng factor matrix, dùng để unvectorize.

    Examples
    --------
    >>> factors = [np.ones((3, 2)), np.ones((4, 2)), np.ones((5, 2))]
    >>> x, shapes = vectorize_factors(factors)
    >>> x.shape
    (24,)
    >>> shapes
    [(3, 2), (4, 2), (5, 2)]
    """
    shapes = [A.shape for A in factors]
    x = np.concatenate([A.ravel() for A in factors])
    return x, shapes


def unvectorize_factors(x, shapes):
    """
    Reconstruct factor matrices from a 1D vector.

    Phép nghịch đảo của vectorize_factors: chuyển vector 1D trở lại
    danh sách các factor matrices.

    Parameters
    ----------
    x : np.ndarray
        1D vector from vectorize_factors.
    shapes : list of tuple
        Shapes of each factor matrix.

    Returns
    -------
    factors : list of np.ndarray
        Reconstructed factor matrices.
    """
    factors = []
    offset = 0
    for shape in shapes:
        size = shape[0] * shape[1]
        A = x[offset:offset + size].reshape(shape)
        factors.append(A)
        offset += size
    return factors


# =============================================================================
# 2. OBJECTIVE FUNCTION
# =============================================================================

def objective(x, tensor, shapes, X_norm_sq):
    """
    Compute the CP objective function: f(x) = ||X - X̂||²_F

    Tính hàm mục tiêu cho bài toán tối ưu CP decomposition.

    Efficient computation (avoiding full tensor reconstruction):
        f = ||X||²_F - 2⟨X, X̂⟩ + ||X̂||²_F

    Cách tính từng thành phần:

    1. ||X||²_F: được truyền vào (precomputed), là hằng số

    2. ⟨X, X̂⟩: sử dụng MTTKRP
       ⟨X, X̂⟩ = Σ_r [MTTKRP(X, factors, 0)]_{:,r}ᵀ · A^(0)_{:,r}
       = trace(MTTKRP(X, factors, 0)ᵀ · A^(0))

    3. ||X̂||²_F: sử dụng Gram matrices
       ||X̂||²_F = 1ᵀ · (⊛_n A^(n)ᵀA^(n)) · 1
       (1 vector vì chưa normalize, weights = 1)

    Parameters
    ----------
    x : np.ndarray
        Vectorized factor matrices.
    tensor : np.ndarray
        Original tensor.
    shapes : list of tuple
        Shapes of factor matrices.
    X_norm_sq : float
        Precomputed ||X||²_F.

    Returns
    -------
    float
        Objective value ||X - X̂||²_F.
    """
    factors = unvectorize_factors(x, shapes)
    N = len(factors)
    R = factors[0].shape[1]

    # Tính ⟨X, X̂⟩ sử dụng MTTKRP ở mode 0
    M = mttkrp(tensor, factors, 0)
    inner = np.sum(M * factors[0])

    # Tính ||X̂||²_F sử dụng Hadamard product của Gram matrices
    gram_hadamard = np.ones((R, R))
    for n in range(N):
        gram_hadamard *= (factors[n].T @ factors[n])
    X_hat_norm_sq = np.sum(gram_hadamard)  # trace = sum all elements khi weights=1

    # f = ||X||²_F - 2⟨X, X̂⟩ + ||X̂||²_F
    obj = X_norm_sq - 2.0 * inner + X_hat_norm_sq

    return max(obj, 0.0)  # Tránh giá trị âm do lỗi floating point


# =============================================================================
# 3. GRADIENT
# =============================================================================

def gradient(x, tensor, shapes, X_norm_sq):
    """
    Compute the gradient of the CP objective function.

    Tính gradient của hàm mục tiêu CP w.r.t. tất cả factor matrices.

    Gradient Formula
    ----------------
    ∂f/∂A^(n) = -2 · MTTKRP(X, factors, n) + 2 · A^(n) · V_n

    where V_n = ⊛_{k≠n} (A^(k)ᵀ A^(k)) (Hadamard of Gram matrices, skip mode n)

    Giải thích:
    - Thành phần -2·MTTKRP: gradient từ inner product ⟨X, X̂⟩
    - Thành phần +2·A^(n)·V_n: gradient từ ||X̂||²_F

    Parameters
    ----------
    x : np.ndarray
        Vectorized factor matrices.
    tensor : np.ndarray
        Original tensor.
    shapes : list of tuple
        Shapes of factor matrices.
    X_norm_sq : float
        Precomputed ||X||²_F (not used in gradient, but kept for API consistency).

    Returns
    -------
    np.ndarray
        Gradient vector (same length as x).
    """
    factors = unvectorize_factors(x, shapes)
    N = len(factors)
    R = factors[0].shape[1]

    # Precompute Gram matrices
    grams = [A.T @ A for A in factors]

    grad_parts = []
    for n in range(N):
        # MTTKRP cho mode n
        M = mttkrp(tensor, factors, n)

        # V_n = ⊛_{k≠n} G^(k)
        V = np.ones((R, R))
        for k in range(N):
            if k != n:
                V *= grams[k]

        # ∂f/∂A^(n) = -2M + 2·A^(n)·V_n
        grad_n = -2.0 * M + 2.0 * (factors[n] @ V)

        grad_parts.append(grad_n.ravel())

    return np.concatenate(grad_parts)


# =============================================================================
# 4. COMBINED OBJECTIVE AND GRADIENT
# =============================================================================

def objective_and_gradient(x, tensor, shapes, X_norm_sq):
    """
    Compute both objective and gradient simultaneously.

    Tối ưu hiệu năng bằng cách tính cả objective và gradient trong
    cùng một lần gọi, tránh tính trùng lặp (MTTKRP, Gram matrices).

    L-BFGS-B trong scipy có thể sử dụng hàm trả về cả (f, g) thông
    qua tùy chọn jac=True.

    Parameters
    ----------
    x : np.ndarray
        Vectorized factor matrices.
    tensor : np.ndarray
        Original tensor.
    shapes : list of tuple
        Shapes of factor matrices.
    X_norm_sq : float
        Precomputed ||X||²_F.

    Returns
    -------
    obj : float
        Objective value.
    grad : np.ndarray
        Gradient vector.
    """
    factors = unvectorize_factors(x, shapes)
    N = len(factors)
    R = factors[0].shape[1]

    # Precompute tất cả Gram matrices
    grams = [A.T @ A for A in factors]

    # Precompute tất cả MTTKRP
    mttrkps = []
    for n in range(N):
        mttrkps.append(mttkrp(tensor, factors, n))

    # === OBJECTIVE ===
    # ⟨X, X̂⟩ (dùng mode 0)
    inner = np.sum(mttrkps[0] * factors[0])

    # ||X̂||²_F
    gram_hadamard = np.ones((R, R))
    for n in range(N):
        gram_hadamard *= grams[n]
    X_hat_norm_sq = np.sum(gram_hadamard)

    obj = max(X_norm_sq - 2.0 * inner + X_hat_norm_sq, 0.0)

    # === GRADIENT ===
    grad_parts = []
    for n in range(N):
        V = np.ones((R, R))
        for k in range(N):
            if k != n:
                V *= grams[k]

        grad_n = -2.0 * mttrkps[n] + 2.0 * (factors[n] @ V)
        grad_parts.append(grad_n.ravel())

    grad = np.concatenate(grad_parts)

    return obj, grad


# =============================================================================
# 5. CP-OPT MAIN FUNCTION
# =============================================================================

def cp_opt(tensor, rank, method='L-BFGS-B', max_iter=500, tol=1e-8,
           random_state=None, init='random', verbose=False, init_factors=None):
    """
    CP Decomposition via gradient-based optimization (L-BFGS-B).

    Sử dụng scipy.optimize.minimize để tìm CP decomposition bằng
    phương pháp tối ưu gradient-based.

    Algorithm
    ---------
    1. Vectorize factor matrices → x₀ ∈ ℝ^p
    2. Gọi scipy.optimize.minimize(f, x₀, method='L-BFGS-B', jac=True)
       - f(x) trả về (objective, gradient)
       - L-BFGS-B sử dụng lịch sử gradient để xấp xỉ Hessian
    3. Unvectorize kết quả → factor matrices
    4. Normalize factors

    Parameters
    ----------
    tensor : np.ndarray
        Input tensor of order ≥ 2.
    rank : int
        CP rank R.
    method : str, default='L-BFGS-B'
        Optimization method for scipy.optimize.minimize.
    max_iter : int, default=500
        Maximum iterations.
    tol : float, default=1e-8
        Tolerance for termination (on projected gradient norm for L-BFGS-B).
    random_state : int or None
        Random seed for initialization.
    init : str, default='random'
        Initialization method: 'random' or 'svd' (used if init_factors is None).
    verbose : bool, default=False
        Print optimization progress.
    init_factors : list of np.ndarray or None
        Initial factor matrices. If None, random initialization.

    Returns
    -------
    weights : np.ndarray
        Component weights λ ∈ ℝ^R.
    factors : list of np.ndarray
        Normalized factor matrices.
    info : dict
        Dictionary with keys:
        - 'obj_history': objective values during optimization
        - 'n_iter': number of iterations
        - 'converged': whether optimization converged
        - 'final_objective': final objective value
        - 'message': scipy optimization message
        - 'fit': final fit value (1 - relative error)
    """
    if tensor.ndim < 2:
        raise ValueError(f"Tensor must have at least 2 dimensions, got {tensor.ndim}")
    if rank < 1:
        raise ValueError(f"Rank must be ≥ 1, got {rank}")

    # Normalize input tensor to unit variance for better gradient optimization
    std_val = tensor.std() + 1e-8
    tensor = tensor / std_val

    shape = tensor.shape
    rng = np.random.RandomState(random_state)

    # Precompute ||X||²_F
    X_norm_sq = float(np.sum(tensor ** 2))
    X_norm = np.sqrt(X_norm_sq)

    if X_norm == 0:
        factors = [np.zeros((s, rank)) for s in shape]
        weights = np.zeros(rank)
        info = {
            'obj_history': [0.0],
            'n_iter': 0,
            'converged': True,
            'final_objective': 0.0,
            'message': 'Zero tensor',
            'fit': 0.0
        }
        return weights, factors, info

    # =========================================================================
    # Khởi tạo
    # =========================================================================
    if init_factors is not None:
        factors = [A.copy() for A in init_factors]
    elif init == 'svd':
        factors = []
        for mode in range(tensor.ndim):
            X_unf = unfold(tensor, mode)
            U, _, _ = np.linalg.svd(X_unf, full_matrices=False)
            factors.append(U[:, :rank])
    else:
        factors = [rng.randn(s, rank) for s in shape]

    # Vectorize
    x0, shapes = vectorize_factors(factors)

    # =========================================================================
    # Lưu lịch sử objective
    # =========================================================================
    obj_history = []

    def callback(xk):
        """Callback function to track optimization progress."""
        obj = objective(xk, tensor, shapes, X_norm_sq)
        obj_history.append(obj)
        if verbose and len(obj_history) % 10 == 0:
            fit = 1.0 - np.sqrt(obj) / X_norm
            print(f"  Iter {len(obj_history):4d}: obj={obj:.6e}, fit={fit:.6f}")

    # =========================================================================
    # Gọi scipy.optimize.minimize
    # =========================================================================
    if verbose:
        print(f"CP-OPT: tensor shape={shape}, rank={rank}, method={method}")
        print(f"  Total parameters: {len(x0)}")

    result = minimize(
        fun=objective_and_gradient,
        x0=x0,
        args=(tensor, shapes, X_norm_sq),
        method=method,
        jac=True,  # objective_and_gradient trả về (f, g)
        callback=callback,
        options={
            'maxiter': max_iter,
            'ftol': tol * X_norm_sq,  # Scale tolerance theo norm
            'gtol': tol,
            'disp': verbose,
            'maxcor': 10,  # L-BFGS history size
        }
    )

    # =========================================================================
    # Unvectorize và normalize kết quả
    # =========================================================================
    factors = unvectorize_factors(result.x, shapes)

    # Normalize factors
    R = rank
    weights = np.ones(R) * std_val
    for n in range(len(factors)):
        norms = np.linalg.norm(factors[n], axis=0)
        norms = np.maximum(norms, 1e-12)
        weights *= norms
        factors[n] = factors[n] / norms[np.newaxis, :]

    # Tính fit cuối cùng
    final_obj = result.fun
    fit = 1.0 - np.sqrt(max(final_obj, 0.0)) / X_norm

    if verbose:
        print(f"\nOptimization {'converged' if result.success else 'did NOT converge'}")
        print(f"  Message: {result.message}")
        print(f"  Final objective: {final_obj:.6e}")
        print(f"  Final fit: {fit:.6f}")
        print(f"  Iterations: {result.nit}")

    info = {
        'obj_history': obj_history,
        'n_iter': result.nit,
        'converged': result.success,
        'final_objective': final_obj,
        'message': str(result.message),
        'fit': fit,
        'scipy_result': result,
    }

    return weights, factors, info
