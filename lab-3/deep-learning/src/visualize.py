from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


def _load_metrics(results_dir: Path) -> list[dict[str, Any]]:
    metrics = []
    for path in sorted(results_dir.glob("*/metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            item = json.load(f)
        item["_path"] = str(path)
        metrics.append(item)
    return metrics


def _summary_frame(metrics: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in metrics:
        model = item.get("model", {})
        history = item.get("history", [])
        lora_rank = None
        name = item.get("run_name", "")
        if "r2" in name:
            lora_rank = 2
        elif "r4" in name:
            lora_rank = 4
        elif "r8" in name:
            lora_rank = 8
        elif "r16" in name:
            lora_rank = 16
        rows.append(
            {
                "run_name": name,
                "mode": model.get("mode"),
                "rank": lora_rank,
                "trainable_params": model.get("trainable_params", 0),
                "total_params": model.get("total_params", 0),
                "trainable_ratio": model.get("trainable_ratio", 0.0),
                "best_val_accuracy": item.get("best_val_accuracy"),
                "final_val_accuracy": item.get("final_val_accuracy"),
                "final_val_loss": item.get("final_val_loss"),
                "elapsed_seconds": item.get("elapsed_seconds"),
                "epochs": len(history),
            }
        )
    return pd.DataFrame(rows)


def _save_accuracy_vs_params(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.scatter(df["trainable_params"], df["best_val_accuracy"], s=80)
    for _, row in df.iterrows():
        plt.annotate(row["run_name"], (row["trainable_params"], row["best_val_accuracy"]))
    plt.xscale("log")
    plt.xlabel("Trainable parameters (log scale)")
    plt.ylabel("Best validation accuracy")
    plt.title("Accuracy vs trainable parameter count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_params.png", dpi=200)
    plt.close()


def _save_params_bar(df: pd.DataFrame, out_dir: Path) -> None:
    sorted_df = df.sort_values("trainable_params")
    plt.figure(figsize=(8, 4.5))
    plt.bar(sorted_df["run_name"], sorted_df["trainable_params"])
    plt.yscale("log")
    plt.ylabel("Trainable parameters (log scale)")
    plt.title("Trainable parameter count by config")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "trainable_params.png", dpi=200)
    plt.close()


def _save_loss_curves(metrics: list[dict[str, Any]], out_dir: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    for item in metrics:
        history = item.get("history", [])
        if not history:
            continue
        epochs = [row["epoch"] for row in history]
        val_loss = [row["val_loss"] for row in history]
        plt.plot(epochs, val_loss, marker="o", label=item.get("run_name", "run"))
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Validation loss curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=200)
    plt.close()


def _save_accuracy_vs_rank(df: pd.DataFrame, out_dir: Path) -> None:
    lora_df = df[df["rank"].notna()].sort_values("rank")
    if lora_df.empty:
        return
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(lora_df["rank"], lora_df["best_val_accuracy"], marker="o")
    plt.xlabel("LoRA rank")
    plt.ylabel("Best validation accuracy")
    plt.title("LoRA rank sweep")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_rank.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/runs")
    parser.add_argument("--out", default="results/plots")
    args = parser.parse_args()

    results_dir = Path(args.results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = _load_metrics(results_dir)
    if not metrics:
        raise SystemExit(f"No metrics.json files found under {results_dir}.")

    df = _summary_frame(metrics)
    df.to_csv(out_dir / "metrics.csv", index=False)

    _save_accuracy_vs_params(df, out_dir)
    _save_params_bar(df, out_dir)
    _save_loss_curves(metrics, out_dir)
    _save_accuracy_vs_rank(df, out_dir)


if __name__ == "__main__":
    main()
