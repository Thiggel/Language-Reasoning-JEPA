"""Calibration and token-class diagnostics for the internal JEPA policy prior."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.lm import LMDataset, collate_lm
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA


def token_class(token: str) -> str:
    if token.isdigit():
        return "number"
    if token in {".", ",", ":", "=", "+", "-", "*", "/", "?"}:
        return "operator_or_punctuation"
    if token.startswith("v") and token[1:].isdigit():
        return "variable"
    return "word"


def summarize(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probabilities = logits.softmax(-1)
    confidence, prediction = probabilities.max(-1)
    correct = prediction.eq(labels)
    rank = logits.ge(logits.gather(-1, labels[:, None])).sum(-1).float()
    bins = torch.linspace(0, 1, 11, device=logits.device)
    ece = logits.new_zeros(())
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.float().mean() * (
                confidence[mask].mean() - correct[mask].float().mean()
            ).abs()
    return {
        "n": int(len(labels)),
        "nll": float(F.cross_entropy(logits, labels)),
        "top1": float(correct.float().mean()),
        "top5": float(logits.topk(min(5, logits.shape[-1]), -1).indices.eq(
            labels[:, None]
        ).any(-1).float().mean()),
        "mean_reference_rank": float(rank.mean()),
        "mean_confidence": float(confidence.mean()),
        "entropy": float(torch.distributions.Categorical(logits=logits).entropy().mean()),
        "ece_10bin": float(ece),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=512)
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    if model.token_prior is None:
        raise ValueError("checkpoint has no token prior")
    dataset = LMDataset(
        vocab, size=args.examples, seed=cfg.data.val_seed,
        modulus=cfg.data.modulus, n_vars_range=tuple(cfg.data.n_vars_range),
        leaf_prob=cfg.data.leaf_prob, steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    loader = DataLoader(
        dataset, batch_size=32, collate_fn=partial(collate_lm, pad_id=vocab.pad_id)
    )
    all_logits, all_labels, all_positions = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["tokens"].to(args.device), batch["prompt_len"].to(args.device)
            )
            valid = out["valid"]
            all_logits.append(out["token_prior_logits"][valid])
            all_labels.append(out["action_ids"][valid])
            positions = torch.arange(valid.shape[1], device=args.device)[None].expand_as(valid)
            all_positions.append(positions[valid])
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    positions = torch.cat(all_positions)
    result = {"overall": summarize(logits, labels), "by_class": {}, "by_phase8": {}}
    classes = [token_class(vocab.id_to_token[int(label)]) for label in labels]
    for name in sorted(set(classes)):
        mask = torch.tensor([value == name for value in classes], device=args.device)
        result["by_class"][name] = summarize(logits[mask], labels[mask])
    for phase in range(8):
        mask = positions.remainder(8).eq(phase)
        if mask.any():
            result["by_phase8"][str(phase)] = summarize(logits[mask], labels[mask])
    destination = Path(args.ckpt).parent / "token_prior_calibration.json"
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
