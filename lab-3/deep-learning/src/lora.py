from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("query", "value", "q_lin", "v_lin")
    inject_all_linear: bool = False
    exclude_keywords: tuple[str, ...] = (
        "classifier",
        "pre_classifier",
        "score",
        "lm_head",
        "pooler",
    )


class LoRALinear(nn.Module):
    """Linear layer with a trainable low-rank update.

    The frozen branch keeps the pretrained matrix W0. The trainable branch
    learns Delta W = B @ A with rank r.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.weight = nn.Parameter(base_layer.weight.detach().clone(), requires_grad=False)
        if base_layer.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(base_layer.bias.detach().clone(), requires_grad=False)

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        frozen_output = F.linear(input_tensor, self.weight, self.bias)
        lora_hidden = F.linear(self.dropout(input_tensor), self.lora_A)
        lora_output = F.linear(lora_hidden, self.lora_B)
        return frozen_output + self.scaling * lora_output

    def delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)

    def merged_weight(self) -> torch.Tensor:
        return self.weight + self.delta_weight()


def _get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _should_replace(name: str, module: nn.Module, config: LoRAConfig) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    lowered = name.lower()
    if any(keyword in lowered for keyword in config.exclude_keywords):
        return False
    if config.inject_all_linear:
        return True
    leaf_name = name.split(".")[-1]
    return leaf_name in set(config.target_modules)


def inject_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    """Replace selected Linear layers with LoRALinear modules."""
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if not _should_replace(name, module, config):
            continue
        parent, child_name = _get_parent_module(model, name)
        setattr(
            parent,
            child_name,
            LoRALinear(
                module,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
            ),
        )
        replaced.append(name)
    return replaced


def iter_lora_layers(model: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }


def lora_layer_norms(model: nn.Module) -> dict[str, float]:
    return {
        name: float(layer.delta_weight().detach().norm().cpu())
        for name, layer in iter_lora_layers(model)
    }


def first_lora_singular_values(model: nn.Module) -> list[float]:
    for _, layer in iter_lora_layers(model):
        values = torch.linalg.svdvals(layer.delta_weight().detach().cpu())
        return [float(v) for v in values]
    return []
