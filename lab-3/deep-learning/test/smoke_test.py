from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["MPLBACKEND"] = "Agg"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from src.config import load_config
from src.data import build_dataloaders
from src.lora import LoRAConfig, LoRALinear, inject_lora, iter_lora_layers
from src.train import evaluate, train_one_epoch
from src.utils import count_parameters, set_seed
from src.visualize import main as visualize_main


class TinyTokenizer:
    pad_token = "[PAD]"
    eos_token = "[EOS]"
    pad_token_id = 0

    def __call__(self, texts, truncation=True, padding=True, max_length=16, return_tensors="pt"):
        rows = []
        masks = []
        for text in texts:
            ids = [min(ord(ch), 255) % 97 + 1 for ch in text][:max_length]
            mask = [1] * len(ids)
            if padding:
                while len(ids) < max_length:
                    ids.append(0)
                    mask.append(0)
            rows.append(ids)
            masks.append(mask)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


class TinyClassifier(nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_size: int = 12, num_labels: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids).mean(dim=1)
        x = torch.relu(self.encoder(x))
        logits = self.classifier(x)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return SimpleNamespace(loss=loss, logits=logits)


def reset_tmp() -> Path:
    tmp = ROOT / "test" / "tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    return tmp


def test_config_files_exist_and_load() -> None:
    required = [
        "frozen.yaml",
        "full_finetune.yaml",
        "lora_r2.yaml",
        "lora_r4.yaml",
        "lora_r8.yaml",
    ]
    for name in required:
        config = load_config(ROOT / "configs" / name)
        assert config["experiment"]["name"]
        assert config["model"]["name"] == "prajjwal1/bert-tiny"
        assert config["dataset"]["fallback_to_toy"] is True


def test_lora_linear_forward_and_gradients() -> None:
    set_seed(7)
    base = nn.Linear(5, 3)
    layer = LoRALinear(base, rank=2, alpha=4, dropout=0.0)
    x = torch.randn(4, 5)
    y = layer(x).sum()
    y.backward()

    assert layer(x).shape == (4, 3)
    assert layer.weight.requires_grad is False
    assert layer.lora_A.requires_grad is True
    assert layer.lora_B.requires_grad is True
    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None
    assert layer.delta_weight().shape == layer.weight.shape


def test_inject_lora_replaces_expected_linear_layers() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    replaced = inject_lora(
        model,
        LoRAConfig(
            rank=2,
            alpha=4,
            target_modules=("0", "2"),
            exclude_keywords=(),
        ),
    )
    assert replaced == ["0", "2"]
    assert len(list(iter_lora_layers(model))) == 2


def test_toy_dataloader_fallback() -> None:
    config = {
        "dataset": {
            "name": "csv",
            "train_file": "missing_train.csv",
            "validation_file": "missing_validation.csv",
            "max_length": 16,
            "train_subset": 12,
            "eval_subset": 8,
            "fallback_to_toy": True,
        },
        "training": {"batch_size": 4},
    }
    train_loader, eval_loader = build_dataloaders(config, TinyTokenizer())
    train_batch = next(iter(train_loader))
    eval_batch = next(iter(eval_loader))

    assert len(train_loader.dataset) == 12
    assert len(eval_loader.dataset) == 8
    assert train_batch["input_ids"].shape == (4, 16)
    assert eval_batch["labels"].dtype == torch.long


def test_train_eval_loop_on_tiny_model() -> None:
    set_seed(11)
    config = {
        "dataset": {
            "name": "csv",
            "train_file": "missing_train.csv",
            "validation_file": "missing_validation.csv",
            "max_length": 16,
            "train_subset": 16,
            "eval_subset": 8,
            "fallback_to_toy": True,
        },
        "training": {"batch_size": 4},
    }
    train_loader, eval_loader = build_dataloaders(config, TinyTokenizer())
    model = TinyClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    metrics = evaluate(model, eval_loader, device)

    assert train_loss > 0
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["loss"] > 0
    params = count_parameters(model)
    assert params["trainable_params"] == params["total_params"]


def test_visualize_from_fake_metrics() -> None:
    tmp = reset_tmp()
    runs = tmp / "runs"
    plots = tmp / "plots"
    for name, mode, trainable, acc in [
        ("frozen", "frozen", 50, 0.60),
        ("full_finetune", "full_finetune", 1000, 0.75),
        ("lora_r2", "lora", 80, 0.68),
        ("lora_r4", "lora", 120, 0.70),
        ("lora_r8", "lora", 200, 0.72),
    ]:
        run_dir = runs / name
        run_dir.mkdir(parents=True)
        payload = {
            "run_name": name,
            "best_val_accuracy": acc,
            "final_val_accuracy": acc,
            "final_val_loss": 0.5,
            "elapsed_seconds": 1.0,
            "history": [
                {"epoch": 1, "train_loss": 0.8, "val_loss": 0.7, "val_accuracy": acc - 0.02},
                {"epoch": 2, "train_loss": 0.6, "val_loss": 0.5, "val_accuracy": acc},
            ],
            "model": {
                "mode": mode,
                "trainable_params": trainable,
                "total_params": 1000,
                "trainable_ratio": trainable / 1000,
            },
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    old_argv = sys.argv[:]
    try:
        sys.argv = ["visualize", "--results", str(runs), "--out", str(plots)]
        visualize_main()
    finally:
        sys.argv = old_argv

    assert (plots / "metrics.csv").exists()
    assert (plots / "accuracy_vs_params.png").exists()
    assert (plots / "trainable_params.png").exists()
    assert (plots / "loss_curves.png").exists()
    assert (plots / "accuracy_vs_rank.png").exists()


def main() -> None:
    tests = [
        test_config_files_exist_and_load,
        test_lora_linear_forward_and_gradients,
        test_inject_lora_replaces_expected_linear_layers,
        test_toy_dataloader_fallback,
        test_train_eval_loop_on_tiny_model,
        test_visualize_from_fake_metrics,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
