"""Audit goal-conditioned primitive-action selection on macro spans."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import collate
from textjepa.training.trainer import to_device
from textjepa.utils.checkpoint import build_dataset, load_run


def summarize(
    cost: torch.Tensor,
    positive: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
    positive = positive & pair_valid
    selected = cost.masked_fill(~valid.unsqueeze(1), torch.inf).argmin(-1)
    chosen_positive = positive.gather(
        -1, selected.unsqueeze(-1)
    ).squeeze(-1)
    row_valid = valid
    best_positive = cost.masked_fill(~positive, torch.inf).min(-1).values
    rank = 1 + (
        (cost < best_positive.unsqueeze(-1)) & pair_valid
    ).sum(-1)
    negative = pair_valid & ~positive
    comparisons = positive.unsqueeze(-1) & negative.unsqueeze(-2)
    correct_pairs = (
        cost.unsqueeze(-1) < cost.unsqueeze(-2)
    ) & comparisons
    random_accuracy = (
        positive.sum(-1).float()
        / pair_valid.sum(-1).clamp_min(1).float()
    )
    return {
        "top1_accuracy": float(chosen_positive[row_valid].float().mean()),
        "mean_rank": float(rank[row_valid].float().mean()),
        "pair_accuracy": float(
            correct_pairs.sum().float() / comparisons.sum().clamp_min(1)
        ),
        "random_top1": float(random_accuracy[row_valid].mean()),
        "rows": int(row_valid.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=2000)
    args = parser.parse_args()
    model, vocab, cfg = load_run(args.ckpt, args.device)
    dataset = build_dataset(cfg, vocab, split="val", size=args.examples)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        collate_fn=partial(collate, pad_id=vocab.pad_id),
    )
    true_costs, predicted_costs, positives, valids = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, torch.device(args.device))
            out = model(batch)
            true_costs.append(out.extras["subgoal_action_cost"].cpu())
            predicted_costs.append(
                out.extras["subgoal_action_cost_pred"].cpu()
            )
            positives.append(out.extras["subgoal_action_positive"].cpu())
            valids.append(out.extras["macro_cf_valid"].cpu())
    true_cost = torch.cat(true_costs)
    predicted_cost = torch.cat(predicted_costs)
    positive = torch.cat(positives)
    valid = torch.cat(valids)
    result = {
        "checkpoint": args.ckpt,
        "examples": args.examples,
        "true_subgoal": summarize(true_cost, positive, valid),
        "predicted_subgoal": summarize(predicted_cost, positive, valid),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
