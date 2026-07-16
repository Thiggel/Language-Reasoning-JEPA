"""Levelwise abstraction probes for variable phrase/sentence boundaries."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from probe_token_hierarchy_v2 import classification_probe, regression_probe, token_labels
from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.semantic_lm import SemanticBoundaryLMDataset, collate_semantic_lm
from textjepa.models.semantic_token_hierarchy import SemanticBoundaryTokenHierarchyJEPA
from textjepa.utils.metrics import effective_rank, feature_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=1000)
    parser.add_argument("--max-points", type=int, default=12000)
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = SemanticBoundaryTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    dataset = SemanticBoundaryLMDataset(
        vocab, size=args.examples, seed=cfg.data.val_seed,
        boundary_mode=cfg.boundary_mode, modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range),
        leaf_prob=cfg.data.leaf_prob, steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    loader = DataLoader(
        dataset, batch_size=32,
        collate_fn=partial(collate_semantic_lm, pad_id=vocab.pad_id),
    )
    stores = [dict(x=[], token=[], kind=[], remaining=[], answer=[], group=[],
                   length=[], code=[], first=[], last=[]) for _ in range(3)]
    with torch.no_grad():
        offset = 0
        for batch in loader:
            batch = {name: value.to(args.device) for name, value in batch.items()}
            out = model(**batch)
            size = batch["tokens"].shape[0]
            answers = np.asarray([
                dataset.igsm.problem(offset + row)[0].answer for row in range(size)
            ])
            episode = offset + np.arange(size)
            offset += size
            valid = out["valid"]
            ids = out["action_ids"][valid].cpu().numpy()
            counts = valid.sum(1).cpu().numpy()
            stores[0]["x"].append(out["target"][valid].cpu())
            stores[0]["token"].append(ids)
            stores[0]["kind"].append(token_labels(ids, vocab))
            stores[0]["remaining"].append(out["low_remaining_target"][valid].cpu().numpy())
            stores[0]["answer"].append(np.repeat(answers, counts))
            stores[0]["group"].append(np.repeat(episode, counts))
            stores[0]["length"].append(np.ones(len(ids)))
            for level_index, level in enumerate(out["levels"], 1):
                mask = level["valid"]
                raw_mask = level["raw_action_valid"][mask]
                raw_ids = level["raw_action_ids"][mask]
                lengths = raw_mask.sum(1).cpu().numpy()
                first = raw_ids[:, 0].cpu().numpy()
                last_index = raw_mask.sum(1).sub(1).clamp_min(0)
                last = raw_ids.gather(1, last_index[:, None]).squeeze(1).cpu().numpy()
                counts = mask.sum(1).cpu().numpy()
                store = stores[level_index]
                store["x"].append(level["target"][mask].cpu())
                store["token"].append(last)
                store["kind"].append(token_labels(last, vocab))
                store["remaining"].append(level["remaining_target"][mask].cpu().numpy())
                store["answer"].append(np.repeat(answers, counts))
                store["group"].append(np.repeat(episode, counts))
                store["length"].append(lengths)
                store["code"].append(level["codes"][mask].cpu())
                store["first"].append(first)
                store["last"].append(last)
    results = {"boundary_mode": cfg.boundary_mode}
    for index, store in enumerate(stores):
        name = ("token", "phrase", "sentence")[index]
        x = torch.cat(store["x"])[:args.max_points]
        arrays = {
            key: np.concatenate(store[key])[:len(x)]
            for key in ("token", "kind", "remaining", "answer", "group", "length")
        }
        features = x.numpy()
        results[name] = {
            "std": feature_std(x), "effective_rank": effective_rank(x[:4096]),
            "endpoint_token_accuracy": classification_probe(
                features, arrays["token"], 0, arrays["group"]
            ),
            "endpoint_type_accuracy": classification_probe(
                features, arrays["kind"], 1, arrays["group"]
            ),
            "remaining_fraction_r2": regression_probe(
                features, arrays["remaining"], 2, arrays["group"]
            ),
            "final_answer_accuracy": classification_probe(
                features, arrays["answer"], 3, arrays["group"]
            ),
            "segment_length_r2": regression_probe(
                features, arrays["length"], 4, arrays["group"]
            ),
        }
        if index:
            code = torch.cat(store["code"])[:args.max_points]
            first = np.concatenate(store["first"])[:len(code)]
            last = np.concatenate(store["last"])[:len(code)]
            groups = arrays["group"][:len(code)]
            results[name].update(
                macro_action_std=feature_std(code),
                macro_action_rank=effective_rank(code[:4096]),
                macro_first_token_accuracy=classification_probe(
                    code.numpy(), first, 0, groups
                ),
                macro_last_token_accuracy=classification_probe(
                    code.numpy(), last, 1, groups
                ),
            )
    destination = Path(args.ckpt).parent / "semantic_representation_probes.json"
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
