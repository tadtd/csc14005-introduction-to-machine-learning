from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .lora import LoRAConfig, first_lora_singular_values, inject_lora, lora_layer_norms
from .utils import count_parameters


CLASSIFIER_KEYWORDS = ("classifier", "pre_classifier", "score")


def _set_requires_grad(model: nn.Module, value: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = value


def _unfreeze_classifier(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if any(keyword in name.lower() for keyword in CLASSIFIER_KEYWORDS):
            parameter.requires_grad = True


def _build_lora_config(config: dict[str, Any]) -> LoRAConfig:
    lora_config = config.get("lora", {})
    return LoRAConfig(
        rank=int(lora_config.get("rank", 4)),
        alpha=float(lora_config.get("alpha", 8.0)),
        dropout=float(lora_config.get("dropout", 0.0)),
        target_modules=tuple(lora_config.get("target_modules", ["query", "value", "q_lin", "v_lin"])),
        inject_all_linear=bool(lora_config.get("inject_all_linear", False)),
    )


def build_model_and_tokenizer(config: dict[str, Any]):
    model_config = config.get("model", {})
    experiment_config = config.get("experiment", {})
    model_name = model_config.get("name", "prajjwal1/bert-tiny")
    num_labels = int(model_config.get("num_labels", 2))
    use_fast_tokenizer = bool(model_config.get("use_fast_tokenizer", False))
    mode = experiment_config.get("mode", "lora")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    replaced_modules: list[str] = []

    if mode == "full_finetune":
        _set_requires_grad(model, True)
    elif mode == "frozen":
        _set_requires_grad(model, False)
        _unfreeze_classifier(model)
    elif mode == "lora":
        _set_requires_grad(model, False)
        replaced_modules = inject_lora(model, _build_lora_config(config))
        _unfreeze_classifier(model)
    else:
        raise ValueError(f"Unknown experiment mode: {mode}")

    metadata = {
        "model_name": model_name,
        "mode": mode,
        "replaced_modules": replaced_modules,
        **count_parameters(model),
    }
    return model, tokenizer, metadata


def model_diagnostics(model: torch.nn.Module) -> dict[str, Any]:
    return {
        "params": count_parameters(model),
        "lora_layer_norms": lora_layer_norms(model),
        "first_lora_singular_values": first_lora_singular_values(model),
    }
