# Kế hoạch Thực nghiệm Demo: CP Circuit trên Binarized MNIST

## Mục tiêu

Demo rằng một **CP-circuit** (probabilistic circuit dùng CP layer) có thể học ước lượng mật độ phân phối trên binarized MNIST, tương ứng với phần **Section 3 + 5.1** của bài báo Loconte et al. (TMLR 2025).

---

## 1. Tổng quan thực nghiệm

Học một monotonic PC với:
- **Region graph**: Linear Tree (LT) hoặc Random Tree (RND)
- **Sum-product layer**: CP⊤ (`CPT`) — tương đương với HCLT, phù hợp nhất cho demo đơn giản
- **Input layer**: Bernoulli distributions (cho binarized MNIST)
- **Optimizer**: Adam, maximum likelihood (log-likelihood objective)
- **Đánh giá**: test-set bits-per-dimension (bpd)

---

## 2. Dataset

- **Binarized MNIST**: 28×28 = 784 binary variables, mỗi pixel ∈ {0, 1}
- Train: 50,000 | Val: 10,000 | Test: 10,000
- Binarization: threshold 0.5 trên raw pixel values (hoặc dùng `torchvision` + threshold)

---

## 3. Thư viện

Dùng **`cirkit`** (the currently maintained package, như đã đề cập trong footnote bài báo):

```
cirkit >= 0.3  (from github.com/april-tools/cirkit)
torch >= 2.0
torchvision
```

Nếu `cirkit` vẫn gặp build issue (pybind11), fallback: **tự implement CP circuit thủ công bằng PyTorch thuần** theo các snippet trong Appendix F của bài báo.

---

## 4. Kiến trúc Circuit

### Region Graph
Dùng **Random Binary Tree (RND)** — đơn giản, không cần học từ data.

- Biến: X₁, …, X₇₈₄ (784 pixel)
- Recursive random split thành cây nhị phân cân bằng (~log₂(784) ≈ 10 tầng)

### Layers
Theo Appendix F (Figure F.4):

```
Input layers  : Bernoulli (K units per leaf region)
Sum-product   : CPT layer  (Eq. CPT-layer)
Folding       : Yes (stack same-depth layers)
```

CPT layer tại mỗi internal node tính:
```
ℓ(Y) = W · (ℓ₁(Z₁) ⊙ ℓ₂(Z₂))
```
với W ∈ ℝ^{K×K}, và ℓ₁, ℓ₂ là output của 2 child layers.

### Hyperparameters cần thử
| Tham số | Giá trị |
|---------|---------|
| K (layer width) | 64, 128, 256 |
| Learning rate | 1e-2 |
| Batch size | 256 |
| Epochs | 50 (early stopping patience=5 trên val LL) |
| Reparameterization | clamp (w = max(ε, θ), ε = 1e-19) |

---

## 5. Implementation Plan

### 5.1 Nếu dùng `cirkit`

```
Bước 1: pip install cirkit (hoặc uv add)
Bước 2: Định nghĩa RG bằng cirkit API
Bước 3: Build PC với CPT layer + Bernoulli input
Bước 4: Train loop với NLL loss + Adam
Bước 5: Eval trên test set → tính bpd
```

### 5.2 Nếu tự implement (fallback)

**File structure:**
```
probabilistic-circuit/
├── src/
│   ├── region_graph.py       # RND binary tree builder
│   ├── layers.py             # InputLayer, CPTLayer (theo snippet F.4)
│   ├── circuit.py            # PC forward pass
│   ├── train.py              # training loop
│   └── utils.py              # bpd computation, data loading
├── experiments/
│   └── run_mnist.py          # entry point
└── pyproject.toml
```

**`layers.py` — CPT layer** (trực tiếp từ Appendix F):

```python
def cpt_layer(log_prob, W):
    # log_prob: (num_folds, H, K, batch_size)
    # W: (num_folds, K, O)
    log_prob = log_prob.sum(dim=1, keepdim=True)
    max_lp = log_prob.max(dim=2, keepdim=True).values
    exp_lp = torch.exp(log_prob - max_lp)
    out = max_lp.sum(dim=1) + torch.einsum(
        "fib,fio->fob", exp_lp[:, 0], W
    ).log()
    return out
```

**`circuit.py` — forward pass:**
```
for each layer in topological order (leaves → root):
    if leaf: evaluate Bernoulli log-prob
    if internal: apply cpt_layer
return log_prob at root
```

**Training objective:**
```python
# Unnormalized PC → normalize via partition function
log_Z = circuit.log_partition()  # tractable vì smooth + decomposable
loss = -(log_prob - log_Z).mean()
# Sau mỗi step: clamp weights ≥ 1e-19
```

---

## 6. Metrics

**bits-per-dimension (bpd):**
```
bpd = -test_log_likelihood / (784 × log(2))
```

**Expected range** (từ Table 2 bài báo): ~1.2–1.7 bpd trên MNIST tùy K.

---

## 7. Kết quả cần report

| Model | K | Test bpd | Train time/epoch |
|-------|---|----------|-----------------|
| RND-CPT | 64 | ? | ? |
| RND-CPT | 128 | ? | ? |
| RND-CPT | 256 | ? | ? |

So sánh với baseline từ bài báo: RAT-SPN (RND-Tucker) ≈ 1.67 bpd, HCLT (CL-CPT) ≈ 1.21 bpd.

---

## 8. Rủi ro và fallback

| Rủi ro | Xử lý |
|--------|-------|
| `cirkit` build lỗi (pybind11) | Dùng PyTorch thuần theo Appendix F |
| Numerical instability trong log-sum-exp | Dùng `torch.logsumexp` thay vì manual max-trick |
| Partition function intractable | Đảm bảo circuit smooth + decomposable by construction |
| OOM với K=256 | Giảm K hoặc batch size |

---