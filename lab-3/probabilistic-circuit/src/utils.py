from __future__ import annotations

import math

import torch
from torchvision.datasets import MNIST


def load_binarized_mnist(
    root: str = "data",
    *,
    threshold: float = 0.5,
    validation_size: int = 10_000,
    download: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load MNIST as flattened binary vectors.

    The paper evaluates density estimation on binarized image data. This helper
    uses the standard MNIST split, reserves the last 10k training images for
    validation, and thresholds pixels after scaling them to [0, 1].
    """

    train_set = MNIST(root=root, train=True, download=download)
    test_set = MNIST(root=root, train=False, download=download)

    train_all = (train_set.data.float().view(-1, 28 * 28) / 255.0 >= threshold).long()
    test = (test_set.data.float().view(-1, 28 * 28) / 255.0 >= threshold).long()

    if not 0 < validation_size < train_all.size(0):
        raise ValueError("validation_size must be between 1 and len(train_set) - 1")

    train = train_all[:-validation_size]
    val = train_all[-validation_size:]
    return train, val, test


def maybe_limit(data: torch.Tensor, limit: int | None) -> torch.Tensor:
    if limit is None or limit <= 0:
        return data
    return data[:limit]


def bits_per_dimension(log_likelihood: torch.Tensor, num_variables: int) -> torch.Tensor:
    return -log_likelihood.mean() / (num_variables * math.log(2.0))
