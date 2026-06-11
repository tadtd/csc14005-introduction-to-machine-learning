from __future__ import annotations

import torch
from torch import nn

from layers import cpj_layer
from region_graph import RegionNode, internal_nodes_postorder, leaves, make_random_binary_tree


class BinaryCPJCircuit(nn.Module):
    """A small monotonic probabilistic circuit for binary variables.

    The circuit follows the paper's "mix-and-match" recipe in miniature:
    a binary region graph supplies decomposable products, categorical input
    distributions model leaves, and CPJ/CP-T layers combine child regions.
    """

    def __init__(
        self,
        num_variables: int,
        width: int = 8,
        seed: int = 0,
        root: RegionNode | None = None,
    ) -> None:
        super().__init__()
        if num_variables <= 0:
            raise ValueError("num_variables must be positive")
        if width <= 0:
            raise ValueError("width must be positive")

        self.num_variables = num_variables
        self.width = width
        self.root = root or make_random_binary_tree(num_variables, seed)
        self.leaf_nodes = leaves(self.root)
        self.internal_nodes = internal_nodes_postorder(self.root)

        self.leaf_logits = nn.Parameter(torch.randn(num_variables, width, 2) * 0.01)
        self.weight_logits = nn.ParameterList()
        for node in self.internal_nodes:
            out_width = 1 if node is self.root else width
            self.weight_logits.append(nn.Parameter(torch.randn(width, out_width) * 0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log p(x) for a batch of binary assignments."""

        if x.ndim != 2 or x.size(1) != self.num_variables:
            raise ValueError(f"x must have shape (batch_size, {self.num_variables})")

        x = x.long()
        values: dict[RegionNode, torch.Tensor] = {}

        leaf_log_probs = self.leaf_logits.log_softmax(dim=-1)
        for node in self.leaf_nodes:
            variable = node.scope[0]
            values[node] = leaf_log_probs[variable, :, x[:, variable]].transpose(0, 1)

        for node, weight_logits in zip(self.internal_nodes, self.weight_logits):
            assert node.left is not None and node.right is not None
            child_log_probs = torch.stack([values[node.left], values[node.right]], dim=1)
            values[node] = cpj_layer(child_log_probs, weight_logits)

        return values[self.root].squeeze(-1)
