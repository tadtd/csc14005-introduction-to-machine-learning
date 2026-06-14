from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class EncodedTextDataset(Dataset):
    encodings: dict[str, torch.Tensor]
    labels: torch.Tensor

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def _limit(items: list[Any], limit: int | None) -> list[Any]:
    if limit is None or limit <= 0:
        return items
    return items[:limit]


def _toy_split() -> tuple[list[str], list[int], list[str], list[int]]:
    positive = [
        "a warm and charming movie",
        "the acting is excellent",
        "this film is surprisingly good",
        "a delightful and funny story",
        "the plot is simple but effective",
        "i loved the emotional ending",
        "a smart and enjoyable performance",
        "the movie feels fresh and lively",
    ]
    negative = [
        "a dull and boring movie",
        "the acting is terrible",
        "this film is painfully bad",
        "a disappointing and slow story",
        "the plot is messy and weak",
        "i disliked the flat ending",
        "a noisy and forgettable performance",
        "the movie feels empty and stale",
    ]
    texts = positive + negative
    labels = [1] * len(positive) + [0] * len(negative)
    train_texts = texts * 16
    train_labels = labels * 16
    eval_texts = texts * 4
    eval_labels = labels * 4
    return train_texts, train_labels, eval_texts, eval_labels


def _encode(
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    labels: list[int],
    max_length: int,
) -> EncodedTextDataset:
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return EncodedTextDataset(dict(encodings), torch.tensor(labels, dtype=torch.long))


def _load_remote_dataset(config: dict[str, Any]) -> tuple[list[str], list[int], list[str], list[int]]:
    dataset_name = config.get("name", "glue")
    subset_name = config.get("subset", "sst2")
    text_column = config.get("text_column", "sentence")
    label_column = config.get("label_column", "label")

    if dataset_name == "glue":
        dataset = load_dataset("glue", subset_name)
    elif dataset_name == "csv":
        dataset = load_dataset(
            "csv",
            data_files={
                "train": config["train_file"],
                "validation": config["validation_file"],
            },
        )
    else:
        dataset = load_dataset(dataset_name, subset_name if subset_name else None)

    train_split = dataset[config.get("train_split", "train")]
    eval_split = dataset[config.get("eval_split", "validation")]

    train_texts = list(train_split[text_column])
    train_labels = [int(x) for x in train_split[label_column]]
    eval_texts = list(eval_split[text_column])
    eval_labels = [int(x) for x in eval_split[label_column]]
    return train_texts, train_labels, eval_texts, eval_labels


def build_dataloaders(
    config: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[DataLoader, DataLoader]:
    data_config = config.get("dataset", {})
    max_length = int(data_config.get("max_length", 128))
    train_subset = data_config.get("train_subset", 5000)
    eval_subset = data_config.get("eval_subset", 1000)
    batch_size = int(config.get("training", {}).get("batch_size", 16))

    try:
        train_texts, train_labels, eval_texts, eval_labels = _load_remote_dataset(data_config)
    except Exception:
        if not data_config.get("fallback_to_toy", True):
            raise
        train_texts, train_labels, eval_texts, eval_labels = _toy_split()

    train_texts = _limit(train_texts, train_subset)
    train_labels = _limit(train_labels, train_subset)
    eval_texts = _limit(eval_texts, eval_subset)
    eval_labels = _limit(eval_labels, eval_subset)

    train_dataset = _encode(tokenizer, train_texts, train_labels, max_length)
    eval_dataset = _encode(tokenizer, eval_texts, eval_labels, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, eval_loader
