"""Train bootstrap macro-predictor heads in one frozen JEPA coordinate system.

Independent full JEPAs cannot form a valid epistemic ensemble because their
latent coordinates may rotate.  These heads share the checkpoint's frozen
encoder, EMA targets, token actions, and macro-action encoders; only the
macro dynamics predictors are independently initialized and trained.
"""

from __future__ import annotations

import argparse
import copy
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.lm import LMDataset, collate_lm
from textjepa.models import MultilevelTokenHierarchyJEPA


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, vocab, cfg


def make_dataset(cfg, vocab, size, seed):
    return LMDataset(
        vocab, size=size, seed=seed, modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range), leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )


def reset_predictor(module):
    for child in module.modules():
        if child is not module and hasattr(child, "reset_parameters"):
            child.reset_parameters()
    if hasattr(module, "pos"):
        nn.init.normal_(module.pos, std=0.02)


def masked_loss(prediction, target, mask, bootstrap):
    prediction = F.layer_norm(prediction, prediction.shape[-1:])
    target = F.layer_norm(target, target.shape[-1:])
    weight = mask.float() * bootstrap
    error = (prediction - target).square().mean(-1)
    return (error * weight).sum() / weight.sum().clamp_min(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=991)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model, vocab, cfg = load_model(args.ckpt, args.device)
    model.requires_grad_(False)
    ensembles = nn.ModuleList()
    for level in model.levels:
        members = nn.ModuleList()
        for _ in range(args.members):
            predictor = copy.deepcopy(level.predictor)
            reset_predictor(predictor)
            predictor.requires_grad_(True)
            members.append(predictor)
        ensembles.append(members)
    ensembles.to(args.device).train()
    optimizer = torch.optim.AdamW(ensembles.parameters(), lr=args.lr, weight_decay=.01)
    dataset = make_dataset(cfg, vocab, args.examples, cfg.data.train_seed + 32452843)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=partial(collate_lm, pad_id=vocab.pad_id), drop_last=True,
    )
    history = []
    for epoch in range(args.epochs):
        sums, count = [0.0] * len(ensembles), 0
        for batch in loader:
            with torch.no_grad():
                out = model(
                    batch["tokens"].to(args.device),
                    batch["prompt_len"].to(args.device),
                )
            losses = []
            for level_index, members in enumerate(ensembles):
                level = out["levels"][level_index]
                for member_index, member in enumerate(members):
                    # Independent online bootstrap samples create genuine
                    # function diversity while retaining shared coordinates.
                    bootstrap = torch.poisson(torch.ones_like(level["valid"], dtype=torch.float))
                    pred = member(level["prev"], level["codes"], level["valid"])
                    loss = masked_loss(pred, level["target"], level["valid"], bootstrap)
                    losses.append(loss)
                    sums[level_index] += float(loss.detach()) / len(members)
            total = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(ensembles.parameters(), 5.0)
            optimizer.step()
            count += 1
        row = {f"level{i + 1}_loss": value / max(count, 1) for i, value in enumerate(sums)}
        history.append(row)
        print(f"epoch={epoch} " + " ".join(f"{k}={v:.5f}" for k, v in row.items()), flush=True)
    output = Path(args.output) if args.output else Path(args.ckpt).parent / "aligned_macro_ensemble.pt"
    torch.save({
        "checkpoint": str(Path(args.ckpt).resolve()),
        "members": args.members,
        "levels": [[member.state_dict() for member in level] for level in ensembles],
        "history": history,
        "args": vars(args),
    }, output)
    print(output)


if __name__ == "__main__":
    main()
