from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class RegionNode:
    """A binary region-graph node.

    Leaves contain one variable index. Internal nodes split their scope into two
    disjoint child scopes, which is what gives the resulting circuit
    decomposability.
    """

    scope: tuple[int, ...]
    left: "RegionNode | None" = None
    right: "RegionNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def make_balanced_binary_tree(variables: list[int] | tuple[int, ...]) -> RegionNode:
    """Build a balanced binary region tree from an ordered variable list."""

    variables = tuple(variables)
    if not variables:
        raise ValueError("at least one variable is required")
    if len(variables) == 1:
        return RegionNode(scope=variables)

    mid = len(variables) // 2
    left = make_balanced_binary_tree(variables[:mid])
    right = make_balanced_binary_tree(variables[mid:])
    return RegionNode(scope=variables, left=left, right=right)


def make_random_binary_tree(num_variables: int, seed: int = 0) -> RegionNode:
    """Build the paper's RND-style binary region graph.

    The implementation shuffles variables once and then recursively creates a
    balanced binary tree, giving random scopes while keeping depth small.
    """

    if num_variables <= 0:
        raise ValueError("num_variables must be positive")

    variables = list(range(num_variables))
    random.Random(seed).shuffle(variables)
    return make_balanced_binary_tree(variables)


def leaves(root: RegionNode) -> list[RegionNode]:
    if root.is_leaf:
        return [root]
    assert root.left is not None and root.right is not None
    return leaves(root.left) + leaves(root.right)


def internal_nodes_postorder(root: RegionNode) -> list[RegionNode]:
    """Return internal nodes in child-before-parent order."""

    if root.is_leaf:
        return []
    assert root.left is not None and root.right is not None
    return (
        internal_nodes_postorder(root.left)
        + internal_nodes_postorder(root.right)
        + [root]
    )
