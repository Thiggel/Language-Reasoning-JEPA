"""Fit controller-specific macro outcome heads on closed-loop rollouts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from textjepa.utils.checkpoint import load_run


def _group_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    group: torch.Tensor,
) -> dict[str, float]:
    prediction = prediction.detach().float().cpu()
    target = target.detach().float().cpu()
    group = group.detach().cpu()
    centered_p = prediction - prediction.mean()
    centered_t = target - target.mean()
    corr = float(
        (centered_p * centered_t).mean()
        / (centered_p.std(unbiased=False) * centered_t.std(unbiased=False))
        .clamp_min(1e-8)
    )
    pair_correct = pair_total = top1 = 0
    regrets = []
    for value in group.unique():
        mask = group == value
        pred_g, target_g = prediction[mask], target[mask]
        better = target_g.unsqueeze(1) < target_g.unsqueeze(0)
        pair_correct += int(
            ((pred_g.unsqueeze(1) < pred_g.unsqueeze(0)) & better).sum()
        )
        pair_total += int(better.sum())
        chosen = int(pred_g.argmin())
        best = float(target_g.min())
        top1 += int(float(target_g[chosen]) == best)
        regrets.append(float(target_g[chosen]) - best)
    n_groups = max(1, len(group.unique()))
    return {
        "mae": float((prediction - target).abs().mean()),
        "corr": corr,
        "pair_accuracy": pair_correct / max(1, pair_total),
        "top1_optimal": top1 / n_groups,
        "top1_regret": sum(regrets) / n_groups,
        "groups": n_groups,
        "examples": len(target),
    }


def _selection_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    groups: list[torch.Tensor],
    rank_weight: float,
    list_weight: float,
    margin: float,
) -> torch.Tensor:
    losses = [F.smooth_l1_loss(prediction, target)]
    offset = 0
    rank_losses = []
    list_losses = []
    for indices in groups:
        length = len(indices)
        pred_g = prediction[offset:offset + length]
        target_g = target[offset:offset + length]
        offset += length
        better = target_g.unsqueeze(1) < target_g.unsqueeze(0)
        if better.any():
            pair = F.relu(
                margin + pred_g.unsqueeze(1) - pred_g.unsqueeze(0)
            )
            rank_losses.append(pair[better].mean())
        best = target_g.min()
        optimal = target_g == best
        list_losses.append(
            torch.logsumexp(-pred_g, dim=0)
            - torch.logsumexp(-pred_g[optimal], dim=0)
        )
    if rank_weight and rank_losses:
        losses.append(rank_weight * torch.stack(rank_losses).mean())
    if list_weight and list_losses:
        losses.append(list_weight * torch.stack(list_losses).mean())
    return torch.stack(losses).sum()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--groups-per-batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--list-weight", type=float, default=0.0)
    parser.add_argument("--rank-margin", type=float, default=0.5)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=321)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, _, _ = load_run(args.checkpoint, args.device)
    model.requires_grad_(False)
    model.core.controller_remaining_head.requires_grad_(True)
    model.core.controller_residual_head.requires_grad_(True)
    model.train()

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    group = data["group"].long()
    unique_groups = group.unique().tolist()
    random.Random(args.seed).shuffle(unique_groups)
    n_val = max(1, round(len(unique_groups) * args.val_fraction))
    val_groups = set(unique_groups[:n_val])
    train_groups = unique_groups[n_val:]
    group_indices = {
        value: torch.where(group == value)[0]
        for value in unique_groups
    }
    optimizer = torch.optim.AdamW(
        list(model.core.controller_remaining_head.parameters())
        + list(model.core.controller_residual_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    generator = random.Random(args.seed)
    for epoch in range(args.epochs):
        generator.shuffle(train_groups)
        losses = []
        for start in range(0, len(train_groups), args.groups_per_batch):
            values = train_groups[start:start + args.groups_per_batch]
            batches = [group_indices[value] for value in values]
            indices = torch.cat(batches)
            state = data["state"][indices].to(args.device)
            initial = data["initial"][indices].to(args.device)
            subgoal = data["subgoal"][indices].to(args.device)
            remaining = data["remaining"][indices].to(args.device)
            residual = data["residual"][indices].to(args.device)
            predicted_remaining = model.core.controller_remaining_head(
                state, initial, subgoal
            )
            predicted_residual = model.core.controller_residual_head(
                state, initial, subgoal
            )
            selection = _selection_loss(
                predicted_remaining,
                remaining,
                batches,
                args.rank_weight,
                args.list_weight,
                args.rank_margin,
            )
            loss = selection + args.residual_weight * F.smooth_l1_loss(
                predicted_residual, residual
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"epoch={epoch + 1} loss={sum(losses) / max(1, len(losses)):.4f}",
                flush=True,
            )

    model.eval()
    metrics = {}
    with torch.no_grad():
        for split, values in (
            ("train", train_groups), ("val", sorted(val_groups))
        ):
            indices = torch.cat([group_indices[value] for value in values])
            state = data["state"][indices].to(args.device)
            initial = data["initial"][indices].to(args.device)
            subgoal = data["subgoal"][indices].to(args.device)
            predicted_remaining = model.core.controller_remaining_head(
                state, initial, subgoal
            )
            predicted_residual = model.core.controller_residual_head(
                state, initial, subgoal
            )
            metrics[split] = {
                "remaining": _group_metrics(
                    predicted_remaining,
                    data["remaining"][indices],
                    group[indices],
                ),
                "residual": _group_metrics(
                    predicted_residual,
                    data["residual"][indices],
                    group[indices],
                ),
            }

    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    payload["model"] = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    payload["controller_outcome"] = {
        "source_dataset": args.dataset,
        "rank_weight": args.rank_weight,
        "list_weight": args.list_weight,
        "rank_margin": args.rank_margin,
        "residual_weight": args.residual_weight,
        "metrics": metrics,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    out.with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"saved to {out}", flush=True)


if __name__ == "__main__":
    main()
