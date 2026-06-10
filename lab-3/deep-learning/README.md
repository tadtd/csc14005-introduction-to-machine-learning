# Deep Learning Demo: LoRA Rank Sweep

This folder contains the deep-learning experiment for the NeurIPS 2025 tutorial
`Foundations of Tensor Computations for AI`.

The demo compares standard fine-tuning with low-rank adaptation (LoRA) on a
small text-classification task. It is designed to show the tutorial's main
deep-learning message: low-rank factorization can reduce trainable parameters
while preserving useful task performance.

## What Is Implemented

- Frozen encoder baseline.
- Full fine-tuning baseline.
- LoRA rank sweep with ranks 2, 4, and 8.
- Metric logging: loss, accuracy, trainable parameters, runtime, device, seed.
- Plot generation from saved run metrics.
- Optional toy-data fallback if the remote dataset cannot be downloaded.

## Setup

From this folder:

```bash
uv sync
```

## Run Experiments

Run a smoke test first by lowering `train_subset` and `eval_subset` in a config,
or create a temporary config with small values.

```bash
uv run python -m src.train --config configs/frozen.yaml
uv run python -m src.train --config configs/full_finetune.yaml
uv run python -m src.train --config configs/lora_r2.yaml
uv run python -m src.train --config configs/lora_r4.yaml
uv run python -m src.train --config configs/lora_r8.yaml
```

Each run writes outputs under:

```text
results/runs/<run_name>/
```

## Evaluate A Checkpoint

```bash
uv run python -m src.evaluate --config configs/lora_r4.yaml --checkpoint results/runs/lora_r4/best_model.pt
```

## Generate Plots

```bash
uv run python -m src.visualize --results results/runs --out results/plots
```

The plotting script creates:

- `accuracy_vs_params.png`
- `trainable_params.png`
- `loss_curves.png`
- `accuracy_vs_rank.png`
- `metrics.csv`

## Notes

- The default model is `prajjwal1/bert-tiny` so the experiment can run on modest
  hardware.
- The default dataset is GLUE SST-2. Downloading the dataset/model requires
  internet access the first time.
- Do not report estimated numbers. Only copy metrics from generated
  `metrics.json` and `metrics.csv` into the LaTeX report after running.
