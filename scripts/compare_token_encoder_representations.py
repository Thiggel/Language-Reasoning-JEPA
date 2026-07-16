"""Compare causal encoder states at identical tokens across checkpoints."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

try:
    from probe_token_hierarchy_v2 import (
        boundary_labels, classification_probe, regression_probe, token_labels,
    )
except ModuleNotFoundError:  # imported as ``scripts.*`` by tests/tools
    from scripts.probe_token_hierarchy_v2 import (
        boundary_labels, classification_probe, regression_probe, token_labels,
    )
from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.lm import LMDataset, collate_lm
from textjepa.models.semantic_token_hierarchy import SemanticBoundaryTokenHierarchyJEPA
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA
from textjepa.utils.metrics import effective_rank, feature_std


def load(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    cls = (
        SemanticBoundaryTokenHierarchyJEPA
        if "boundary_mode" in cfg else MultilevelTokenHierarchyJEPA
    )
    model = cls(vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, vocab, cfg


def linear_cka(left, right):
    left = left - left.mean(0, keepdim=True)
    right = right - right.mean(0, keepdim=True)
    cross = (left.T @ right).square().sum()
    left_norm = (left.T @ left).square().sum().sqrt()
    right_norm = (right.T @ right).square().sum().sqrt()
    return float(cross / (left_norm * right_norm).clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="LABEL=PATH; repeat for every model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--max-points", type=int, default=12000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parsed = [entry.split("=", 1) for entry in args.checkpoint]
    loaded = [(label, *load(path, args.device)) for label, path in parsed]
    _, vocab, reference_cfg = loaded[0][1:]
    dataset = LMDataset(
        vocab, size=args.examples, seed=reference_cfg.data.val_seed,
        modulus=reference_cfg.data.modulus,
        n_vars_range=tuple(reference_cfg.data.n_vars_range),
        leaf_prob=reference_cfg.data.leaf_prob,
        steps_range=tuple(reference_cfg.data.steps_range),
        distractor_prob=reference_cfg.data.distractor_prob,
        max_distractors=reference_cfg.data.max_distractors,
    )
    loader = list(DataLoader(
        dataset, batch_size=32,
        collate_fn=partial(collate_lm, pad_id=vocab.pad_id),
    ))
    labels = dict(token=[], kind=[], remaining=[], answer=[], group=[],
                  sentence_position=[], boundary_distance=[])
    for batch_index, batch in enumerate(loader):
        lengths = batch["tokens"].ne(vocab.pad_id).sum(1) - batch["prompt_len"]
        width = int(lengths.max())
        ids = torch.full((len(lengths), width), vocab.pad_id, dtype=torch.long)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for row, length in enumerate(lengths.tolist()):
            start = int(batch["prompt_len"][row])
            ids[row, :length] = batch["tokens"][row, start:start + length]
            valid[row, :length] = True
        sentence_position, boundary_distance = boundary_labels(
            ids.numpy(), valid.numpy(), vocab.token_to_id["."]
        )
        flat = ids[valid].numpy()
        count = valid.sum(1).numpy()
        episode = batch_index * 32 + np.arange(len(lengths))
        answers = np.asarray([
            dataset.igsm.problem(int(index))[0].answer for index in episode
        ])
        position = np.concatenate([
            np.arange(int(length), 0, -1) / max(int(length), 1)
            for length in lengths
        ])
        labels["token"].append(flat)
        labels["kind"].append(token_labels(flat, vocab))
        labels["remaining"].append(position)
        labels["answer"].append(np.repeat(answers, count))
        labels["group"].append(np.repeat(episode, count))
        labels["sentence_position"].append(sentence_position[valid.numpy()])
        labels["boundary_distance"].append(boundary_distance[valid.numpy()])
    labels = {name: np.concatenate(value)[:args.max_points] for name, value in labels.items()}
    features = {}
    with torch.no_grad():
        for label, model, _, _ in loaded:
            rows = []
            for batch in loader:
                tokens = batch["tokens"].to(args.device)
                state = model.encoder(tokens)
                for row in range(len(tokens)):
                    start = int(batch["prompt_len"][row])
                    length = int(tokens[row].ne(vocab.pad_id).sum()) - start
                    rows.append(state[row, start:start + length].cpu())
            features[label] = torch.cat(rows)[:args.max_points]
    result = {"models": {}, "linear_cka": {}}
    for label, state in features.items():
        x = state.numpy()
        result["models"][label] = {
            "std": feature_std(state), "effective_rank": effective_rank(state[:4096]),
            "token_identity_accuracy": classification_probe(
                x, labels["token"], 0, labels["group"]
            ),
            "token_type_accuracy": classification_probe(
                x, labels["kind"], 1, labels["group"]
            ),
            "remaining_fraction_r2": regression_probe(
                x, labels["remaining"], 2, labels["group"]
            ),
            "final_answer_accuracy": classification_probe(
                x, labels["answer"], 3, labels["group"]
            ),
            "sentence_position_r2": regression_probe(
                x, labels["sentence_position"], 0, labels["group"]
            ),
            "boundary_distance_r2": regression_probe(
                x, labels["boundary_distance"], 1, labels["group"]
            ),
        }
    for left, left_state in features.items():
        result["linear_cka"][left] = {
            right: linear_cka(left_state, right_state)
            for right, right_state in features.items()
        }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
