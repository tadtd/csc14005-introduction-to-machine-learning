# Probabilistic Circuit Demo

This is a small from-scratch PyTorch demo inspired by Loconte et al. (TMLR 2025), *What is the Relationship between Tensor Factorizations and Circuits (and How Can We Exploit it)?*

The demo implements a binary CPJ/CP-T probabilistic circuit:

- categorical input layers for binary variables,
- a random balanced binary region graph,
- CPJ sum-product layers from Appendix F.4,
- maximum-likelihood training with Adam.

Run it with uv:

```powershell
uv run python experiments/run_toy.py
```

The toy experiment learns a 4-bit correlated distribution and enumerates all 16 assignments exactly, showing that the learned circuit defines a normalized probability tensor.

MNIST demo:

```powershell
uv run python experiments/run_mnist.py
```

The MNIST script downloads MNIST through `torchvision`, thresholds pixels at 0.5, trains with maximum likelihood, and reports validation/test bits-per-dimension. By default it uses a small subset for quick run; pass larger `--max-train`, `--max-val`, `--max-test`, `--width`, and `--epochs` values for a heavier experiment.

You can find the demo in this link: [Probabilistic Circuit Demo](https://www.kaggle.com/code/dtdat1234/demo-probabilistic-circuit)