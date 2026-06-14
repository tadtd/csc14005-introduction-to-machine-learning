from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuit import BinaryCPJCircuit
from utils import bits_per_dimension, load_binarized_mnist, maybe_limit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small CPJ probabilistic circuit on binarized MNIST."
    )
    parser.add_argument("--data-dir", default="data", help="MNIST cache directory.")
    parser.add_argument("--width", type=int, default=16, help="Circuit layer width K.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-train", type=int, default=5_000)
    parser.add_argument("--max-val", type=int, default=1_000)
    parser.add_argument("--max-test", type=int, default=1_000)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Use CUDA when available by default.",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: BinaryCPJCircuit,
    data: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    loader = DataLoader(TensorDataset(data), batch_size=batch_size)
    total_ll = 0.0
    seen = 0
    for (batch,) in loader:
        batch = batch.to(device)
        log_likelihood = model(batch)
        total_ll += log_likelihood.sum().item()
        seen += batch.size(0)

    mean_ll = total_ll / seen
    bpd = bits_per_dimension(torch.tensor(mean_ll), model.num_variables).item()
    return mean_ll, bpd


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train, val, test = load_binarized_mnist(
        args.data_dir,
        threshold=args.threshold,
        download=not args.no_download,
    )
    train = maybe_limit(train, args.max_train)
    val = maybe_limit(val, args.max_val)
    test = maybe_limit(test, args.max_test)

    model = BinaryCPJCircuit(num_variables=28 * 28, width=args.width, seed=args.seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(
        TensorDataset(train),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    best_val_ll = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    print("MNIST CPJ probabilistic circuit")
    print(
        f"device={device}, width={args.width}, train={len(train)}, "
        f"val={len(val)}, test={len(test)}"
    )

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        total_loss = 0.0
        seen = 0

        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = -model(batch).mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)
            seen += batch.size(0)

        train_nll = total_loss / seen
        val_ll, val_bpd = evaluate(
            model,
            val,
            batch_size=args.batch_size,
            device=device,
        )
        elapsed = time.perf_counter() - started

        print(
            f"epoch {epoch:03d} | train NLL {train_nll:.3f} | "
            f"val LL {val_ll:.3f} | val bpd {val_bpd:.4f} | {elapsed:.1f}s"
        )

        if val_ll > best_val_ll:
            best_val_ll = val_ll
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping after {epoch} epochs")
                break

    model.load_state_dict(best_state)
    test_ll, test_bpd = evaluate(
        model,
        test,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"best val LL: {best_val_ll:.3f}")
    print(f"test LL: {test_ll:.3f}")
    print(f"test bpd: {test_bpd:.4f}")


if __name__ == "__main__":
    main()
