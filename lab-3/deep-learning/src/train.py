from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from .config import load_config
from .data import build_dataloaders
from .model import build_model_and_tokenizer, model_diagnostics
from .utils import Timer, ensure_dir, get_device, save_json, set_seed


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    correct = 0

    for batch in loader:
        batch = _move_batch(batch, device)
        outputs = model(**batch)
        logits = outputs.logits
        labels = batch["labels"]
        batch_size = labels.shape[0]
        total_loss += float(outputs.loss.detach().cpu()) * batch_size
        correct += int((logits.argmax(dim=-1) == labels).sum().detach().cpu())
        total_items += batch_size

    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": correct / max(total_items, 1),
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        batch_size = batch["labels"].shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)

    device = get_device(bool(config.get("training", {}).get("prefer_cuda", True)))
    model, tokenizer, metadata = build_model_and_tokenizer(config)
    model.to(device)

    train_loader, eval_loader = build_dataloaders(config, tokenizer)
    training_config = config.get("training", {})
    learning_rate = float(training_config.get("learning_rate", 2e-4))
    weight_decay = float(training_config.get("weight_decay", 0.0))
    epochs = int(training_config.get("epochs", 3))

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)

    run_name = config.get("experiment", {}).get("name", "run")
    output_dir = ensure_dir(Path(config.get("output_dir", "results/runs")) / run_name)
    best_path = output_dir / "best_model.pt"
    timer = Timer()
    best_accuracy = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        eval_metrics = evaluate(model, eval_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": eval_metrics["loss"],
            "val_accuracy": eval_metrics["accuracy"],
        }
        history.append(record)

        if eval_metrics["accuracy"] > best_accuracy:
            best_accuracy = eval_metrics["accuracy"]
            torch.save(model.state_dict(), best_path)

    diagnostics = model_diagnostics(model)
    metrics = {
        "run_name": run_name,
        "config_path": config.get("_config_path", ""),
        "seed": seed,
        "device": str(device),
        "elapsed_seconds": timer.elapsed(),
        "best_val_accuracy": best_accuracy,
        "final_val_accuracy": history[-1]["val_accuracy"] if history else None,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "history": history,
        "model": metadata,
        "diagnostics": diagnostics,
        "checkpoint": str(best_path),
    }
    save_json(metrics, output_dir / "metrics.json")
    save_json(config, output_dir / "config.resolved.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    args = parser.parse_args()
    config = load_config(args.config)
    run_training(config)


if __name__ == "__main__":
    main()
