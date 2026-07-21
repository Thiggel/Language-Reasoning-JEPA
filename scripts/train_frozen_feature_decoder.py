"""Train one fixed-capacity sentence readout on a frozen NPZ representation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from textjepa.analysis.reconstruction import FrozenFeatureDecoder
from textjepa.utils import seed_everything


def _split_groups(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = rng.permutation(np.unique(groups))
    cut = max(1, min(len(unique) - 1, int(round(.8 * len(unique)))))
    fit_groups = set(unique[:cut].tolist())
    return (
        np.asarray([i for i, value in enumerate(groups) if value in fit_groups]),
        np.asarray([i for i, value in enumerate(groups) if value not in fit_groups]),
    )


@torch.no_grad()
def _metrics(model, features, targets, batch_size: int) -> dict[str, float]:
    model.eval()
    losses, correct, total, exact, sequences = [], 0, 0, 0, 0
    for x, y in DataLoader(TensorDataset(features, targets), batch_size=batch_size):
        x, y = x.to(next(model.parameters()).device), y.to(next(model.parameters()).device)
        loss = model.loss(x, y)
        losses.append(float(loss))
        prediction = model.generate(x, y.shape[1])
        mask = y.ne(model.pad_id)
        correct += int(((prediction == y) & mask).sum())
        total += int(mask.sum())
        exact += int((((prediction == y) | ~mask).all(1)).sum())
        sequences += len(y)
    mean_loss = sum(losses) / len(losses)
    return {
        "cross_entropy": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "token_accuracy": correct / max(total, 1),
        "exact_match": exact / max(sequences, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--target", choices=("action", "outcome"), required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    seed_everything(args.seed)
    train, test = np.load(args.train), np.load(args.test)
    key = f"target_{args.target}_tokens"
    train_x = torch.from_numpy(train["representations"]).float()
    train_y = torch.from_numpy(train[key]).long()
    test_x = torch.from_numpy(test["representations"]).float()
    test_y = torch.from_numpy(test[key]).long()
    fit, val = _split_groups(train["problem_id"], args.seed)
    model = FrozenFeatureDecoder(
        train_x.shape[1], int(train["vocab_size"]), int(train["pad_id"]),
        args.hidden_dim,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(train_x[fit], train_y[fit]), batch_size=args.batch_size,
        shuffle=True,
    )
    best, best_state = float("inf"), None
    for _ in range(args.epochs):
        model.train()
        for x, y in loader:
            loss = model.loss(x.to(args.device), y.to(args.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        score = _metrics(model, train_x[val], train_y[val], args.batch_size)
        if score["cross_entropy"] < best:
            best = score["cross_entropy"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    result = {
        "target": args.target, "learning_rate": args.lr, "seed": args.seed,
        "validation_cross_entropy": best,
        "test": _metrics(model, test_x, test_y, args.batch_size),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
