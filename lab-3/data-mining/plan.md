### 4.3 Part III — Data Mining (`data-mining/`)

**Experiment: CP decomposition for interpretable pattern discovery**

Apply CP decomposition to a real multi-way dataset and show that the extracted factors are interpretable — replicating the tutorial's core claim that CP uniqueness enables direct insight without post-hoc rotation.

**Dataset:** EEG Motor Imagery dataset from PhysioNet — structured as `subjects × channels × time-points` (3-way tensor). Publicly available, small enough to run locally.
**Fallback:** Synthetic data with known ground-truth factors — use this first to verify correctness before moving to real data.

### File structure

```
data-mining/
├── pyproject.toml
├── README.md
├── src/
│   ├── data.py            # Download & preprocess into tensor format
│   ├── cp_als.py          # CP-ALS from scratch using MTTKRP
│   ├── cp_opt.py          # CP-OPT: gradient-based via L-BFGS (scipy)
│   ├── parafac2.py        # PARAFAC2-ALS for time-varying spatial data
│   ├── evaluate.py        # Reconstruction error, Factor Match Score (FMS)
│   └── visualize.py       # Factor heatmaps, component plots
├── notebooks/
│   ├── 01_synthetic.ipynb        # CP on synthetic data with known factors
│   ├── 02_eeg_cp.ipynb           # CP on real EEG tensor
│   ├── 03_parafac2.ipynb         # PARAFAC2 for time-varying data
│   └── 04_cmtf.ipynb             # (Optional) Coupled factorization
└── results/
```

### Implementation steps

**Step 1 — `cp_als.py`: Implement CP-ALS from scratch**

```python
def cp_als(X, rank, n_iter=200, tol=1e-6):
    # Initialize factor matrices A, B, C with random normal entries
    # Normalize each column and track lambda (norms)
    # While not converged:
    #   Update A: A ← X_(1) · (C ⊙ B) · pinv((C^T C) * (B^T B))
    #   Update B: B ← X_(2) · (C ⊙ A) · pinv((C^T C) * (A^T A))
    #   Update C: C ← X_(3) · (B ⊙ A) · pinv((B^T B) * (A^T A))
    #   Normalize columns, absorb norms into lambda
    #   Convergence: relative change in ||X - X_hat||_F
    return lambda_, A, B, C
```

The key subroutine is **MTTKRP** (Matricized Tensor Times Khatri-Rao Product): `X_(n) · (factor_N ⊙ … ⊙ factor_1)` skipping mode `n`. Implement this efficiently using `numpy.einsum`.

**Step 2 — `cp_opt.py`: CP-OPT with L-BFGS**

- Vectorize all factor matrices into one flat parameter vector
- Compute gradient of `||X - X_hat||^2_F` analytically and pass to `scipy.optimize.minimize(method='L-BFGS-B')`
- Compare convergence speed vs CP-ALS on the same data

**Step 3 — Synthetic verification (`01_synthetic.ipynb`)**

- Generate ground-truth `A_true, B_true, C_true` with known rank `R`
- Construct `X = CP(lambda_true, A_true, B_true, C_true) + noise`
- Run CP-ALS and CP-OPT; measure **Factor Match Score (FMS)** against ground truth
- Key result: CP recovers true factors (up to permutation and scaling); SVD on the unfolded matrix does not
- Show FMS vs noise level as a robustness curve

**Step 4 — Real EEG experiment (`02_eeg_cp.ipynb`)**

- Load EEG Motor Imagery tensor: `subjects × channels × time-points`
- Run CP-ALS with ranks `R = 2, 4, 8, 16`; select R via elbow on reconstruction error
- Visualize each component's three factor vectors:
    - **Subject mode:** which subjects load on which component → group differences
    - **Channel mode:** spatial topography — which electrode clusters activate
    - **Time mode:** temporal signature — event-related response shape
- Compare interpretability vs plain truncated SVD on the mode-1 unfolding

**Step 5 — PARAFAC2 (`03_parafac2.ipynb`)**

- Use when trial lengths differ across subjects (PARAFAC2 relaxes strict multilinearity in one mode)
- Implement PARAFAC2-ALS: alternates between SVD-based projection updates and standard ALS steps
- Apply to synthetic irregular-length data and verify recovery of time-varying spatial patterns

**Step 6 — Visualizations**

- **Plot 1:** Reconstruction error vs rank for CP-ALS and CP-OPT — elbow plot for rank selection
- **Plot 2:** Factor matrices as heatmaps per component (subjects × channels × time)
- **Plot 3:** FMS vs noise level on synthetic data — robustness analysis
- **Plot 4:** PARAFAC2 vs CP spatial components side by side on fMRI-like synthetic data