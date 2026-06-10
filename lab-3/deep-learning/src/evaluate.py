from __future__ import annotations

import argparse

import torch

from .config import load_config
from .data import build_dataloaders
from .model import build_model_and_tokenizer, model_diagnostics
from .train import evaluate
from .utils import get_device, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="results/eval_metrics.json")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("training", {}).get("seed", 42))
    set_seed(seed)
    device = get_device(bool(config.get("training", {}).get("prefer_cuda", True)))

    model, tokenizer, metadata = build_model_and_tokenizer(config)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)

    _, eval_loader = build_dataloaders(config, tokenizer)
    metrics = evaluate(model, eval_loader, device)
    output = {
        "checkpoint": args.checkpoint,
        "config_path": args.config,
        "device": str(device),
        "metrics": metrics,
        "model": metadata,
        "diagnostics": model_diagnostics(model),
    }
    save_json(output, args.out)


if __name__ == "__main__":
    main()
