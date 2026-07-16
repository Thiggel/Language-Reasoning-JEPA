"""Measure objective competition on the shared causal token encoder."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from train_token_hierarchy_v2 import compute_losses
from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.lm import LMDataset, collate_lm
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA


def gradient(model, loss):
    parameters = tuple(model.encoder.parameters())
    if not loss.requires_grad:
        return tuple(torch.zeros_like(parameter) for parameter in parameters)
    values = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True
    )
    return tuple(
        torch.zeros_like(parameter) if value is None else value.detach()
        for parameter, value in zip(parameters, values)
    )


def norm(values):
    return float(torch.sqrt(sum(value.square().sum() for value in values)))


def cosine(left, right):
    dot = sum((a * b).sum() for a, b in zip(left, right))
    nl = torch.sqrt(sum(value.square().sum() for value in left))
    nr = torch.sqrt(sum(value.square().sum() for value in right))
    return float(dot / (nl * nr).clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.train()
    dataset = LMDataset(
        vocab, size=args.batch_size, seed=cfg.data.val_seed + 7919,
        modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range),
        leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    batch = next(iter(DataLoader(
        dataset, batch_size=args.batch_size,
        collate_fn=partial(collate_lm, pad_id=vocab.pad_id),
    )))
    out = model(
        batch["tokens"].to(args.device), batch["prompt_len"].to(args.device)
    )
    _, items = compute_losses(out, cfg)
    names = ["low_prediction", "low_dense", "goal_prediction", "vicreg"]
    for index in range(len(out["levels"])):
        names.extend([
            f"level{index + 1}_prediction",
            f"level{index + 1}_dense",
            f"level{index + 1}_reachability",
            f"level{index + 1}_value",
        ])
    if "token_prior" in items:
        names.extend([
            name for name in ("token_prior", "token_prior_rollout")
            if name in items
        ])
    gradients = {name: gradient(model, items[name]) for name in names}
    result = {
        "encoder_gradient_norm": {
            name: norm(value) for name, value in gradients.items()
        },
        "encoder_gradient_cosine": {
            left: {
                right: cosine(gradients[left], gradients[right])
                for right in names
            }
            for left in names
        },
        "parameter_count": {
            "trainable": sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "including_ema_teacher": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
        "batch_size": args.batch_size,
    }
    destination = Path(args.ckpt).parent / "gradient_diagnostics.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
