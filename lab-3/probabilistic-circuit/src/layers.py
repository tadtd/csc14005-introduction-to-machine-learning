from __future__ import annotations

import torch


def cpj_layer(log_prob: torch.Tensor, weight_logits: torch.Tensor) -> torch.Tensor:
    """Evaluate a CPJ/CP-T sum-product layer from Appendix F.4.

    This is the log-domain form of

        out_o = sum_k W[k, o] * prod_h child_h[k]

    with W column-normalized by log-softmax so every output remains a normalized
    distribution when the child outputs are normalized distributions.

    Args:
        log_prob: Input log probabilities, shape (batch_size, arity, width).
        weight_logits: Raw weights, shape (width, out_width).

    Returns:
        Output log probabilities, shape (batch_size, out_width).
    """

    if log_prob.ndim != 3:
        raise ValueError("log_prob must have shape (batch_size, arity, width)")
    if weight_logits.ndim != 2:
        raise ValueError("weight_logits must have shape (width, out_width)")

    product_log_prob = log_prob.sum(dim=1)
    log_weights = weight_logits.log_softmax(dim=0)
    return torch.logsumexp(product_log_prob[:, :, None] + log_weights[None, :, :], dim=1)
