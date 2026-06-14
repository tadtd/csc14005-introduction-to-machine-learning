from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from circuit import BinaryCPJCircuit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny CPJ probabilistic circuit and enumerate its probability "
            "tensor, following the paper's circuits-as-tensor-factorizations view."
        )
    )
    parser.add_argument("--width", type=int, default=8, help="CPJ layer width K.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def all_binary_assignments(num_variables: int) -> torch.Tensor:
    return torch.tensor(
        list(itertools.product([0, 1], repeat=num_variables)),
        dtype=torch.long,
    )


def target_distribution(assignments: torch.Tensor) -> torch.Tensor:
    x0, x1, x2, x3 = assignments.T
    same_pair = (x0 == x1).float()
    xor_pair = ((x2 ^ x3) == 1).float()
    parity = ((x0 ^ x1 ^ x2 ^ x3) == 0).float()

    logits = 1.8 * same_pair + 1.4 * xor_pair + 0.8 * parity
    return torch.softmax(logits, dim=0)


def sample_from_distribution(
    assignments: torch.Tensor,
    probabilities: torch.Tensor,
    *,
    num_samples: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.multinomial(
        probabilities,
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )
    return assignments[indices]


@torch.no_grad()
def enumerate_model(model: BinaryCPJCircuit, assignments: torch.Tensor) -> torch.Tensor:
    return model(assignments).exp()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    assignments = all_binary_assignments(num_variables=4)
    target = target_distribution(assignments)
    train = sample_from_distribution(
        assignments,
        target,
        num_samples=2_000,
        seed=args.seed,
    )

    model = BinaryCPJCircuit(num_variables=4, width=args.width, seed=args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = -model(train).mean()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            learned = enumerate_model(model, assignments)
            kl = (target * (target.log() - learned.clamp_min(1e-12).log())).sum()
            print(
                f"epoch {epoch:03d} | train NLL {loss.item():.4f} | "
                f"target KL {kl.item():.4f}"
            )

    learned = enumerate_model(model, assignments)
    print("\nAssignment tensor entries")
    print("x0 x1 x2 x3 | target p(x) | circuit p(x)")
    for assignment, target_prob, learned_prob in zip(assignments, target, learned):
        bits = "  ".join(str(int(bit)) for bit in assignment.tolist())
        print(f"{bits} | {target_prob.item():.5f}     | {learned_prob.item():.5f}")

    print(f"\nSum of enumerated circuit probabilities: {learned.sum().item():.6f}")
    print(
        "The CPJ circuit evaluates one entry p(x) of a non-negative probability "
        "tensor at a time; enumerating all 2^4 entries verifies normalization."
    )


if __name__ == "__main__":
    main()
