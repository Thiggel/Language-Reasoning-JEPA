"""Measure action-feasibility support on true and model-predicted states."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import collate
from textjepa.training.trainer import to_device
from textjepa.utils.checkpoint import build_dataset, load_run


def metrics(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> dict[str, float]:
    prediction = logits >= 0
    positive = valid & target
    negative = valid & ~target
    pair_valid = positive.unsqueeze(-1) & negative.unsqueeze(-2)
    pair_correct = (
        logits.unsqueeze(-1) > logits.unsqueeze(-2)
    ) & pair_valid
    return {
        "accuracy": float(prediction[valid].eq(target[valid]).float().mean()),
        "positive_recall": float(prediction[positive].float().mean()),
        "negative_recall": float((~prediction[negative]).float().mean()),
        "pair_accuracy": float(
            pair_correct.sum().float() / pair_valid.sum().clamp_min(1)
        ),
        "positive_logit": float(logits[positive].mean()),
        "negative_logit": float(logits[negative].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=2000)
    parser.add_argument("--length", type=int)
    args = parser.parse_args()
    model, vocab, cfg = load_run(args.ckpt, args.device)
    cfg.data.all_action_supervision = True
    if args.length is not None:
        cfg.data.steps_range = [args.length, args.length]
        cfg.data.n_vars_range = [6, 12]
        cfg.data.strict_steps_range = True
        cfg.data.problem_max_tries = 10000
    dataset = build_dataset(cfg, vocab, split="val", size=args.examples)
    loader = DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=0,
        collate_fn=partial(collate, pad_id=vocab.pad_id),
    )
    collected = {name: [] for name in ("true", "one_step", "open_loop")}
    targets, masks = [], []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, torch.device(args.device))
            out = model(batch)
            logits = out.extras["action_support_logits"]
            if logits.dim() == 3:
                logits = logits.unsqueeze(1)
            if logits.shape[1] not in {1, 3}:
                raise RuntimeError("unexpected action-support state axis")
            for index, name in enumerate(("true", "one_step", "open_loop")):
                if index < logits.shape[1]:
                    collected[name].append(logits[:, index].cpu())
            targets.append(batch["action_feasible"].cpu())
            masks.append((
                out.step_mask.unsqueeze(-1)
                & batch["action_candidate_mask"].unsqueeze(1)
            ).cpu())
    max_t = max(tensor.shape[1] for tensor in targets)
    max_v = max(tensor.shape[2] for tensor in targets)

    def padded(parts: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat([
            F.pad(
                tensor,
                (0, max_v - tensor.shape[2], 0, max_t - tensor.shape[1]),
            )
            for tensor in parts
        ])

    target = padded(targets)
    valid = padded(masks)
    result = {
        "checkpoint": args.ckpt,
        "examples": args.examples,
        "exact_length": args.length,
        "metrics": {
            name: metrics(padded(parts), target, valid)
            for name, parts in collected.items() if parts
        },
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
