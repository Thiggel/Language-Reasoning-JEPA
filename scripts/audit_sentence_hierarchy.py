"""Representation, support, value, and reachability diagnostics for sentence JEPA."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

try:
    from probe_token_hierarchy_v2 import classification_probe, regression_probe
except ModuleNotFoundError:  # imported as ``scripts.*`` by tests
    from scripts.probe_token_hierarchy_v2 import classification_probe, regression_probe
from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.semantic_lm import SemanticBoundaryLMDataset, collate_semantic_lm
from textjepa.models.sentence_hierarchy import SentenceHierarchyJEPA
from textjepa.utils.metrics import effective_rank, feature_std


def normalized_error(left, right):
    return F.mse_loss(
        F.layer_norm(left, left.shape[-1:]),
        F.layer_norm(right, right.shape[-1:]), reduction="none",
    ).mean(-1)


def pairwise_accuracy(score, target):
    delta_target = target.unsqueeze(2) - target.unsqueeze(1)
    delta_score = score.unsqueeze(2) - score.unsqueeze(1)
    valid = delta_target.ne(0)
    if not valid.any():
        return float("nan")
    return float((delta_target[valid] * delta_score[valid] > 0).float().mean())


def safe_probe(function, *args):
    try:
        return function(*args)
    except ValueError:
        return float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--max-points", type=int, default=4000)
    parser.add_argument("--gar-k", type=int, default=8)
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = SentenceHierarchyJEPA(
        len(vocab), vocab.pad_id, **cfg.model
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    dataset = SemanticBoundaryLMDataset(
        vocab, size=args.examples, seed=cfg.data.val_seed + 7919,
        boundary_mode="semantic", modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range),
        leaf_prob=cfg.data.leaf_prob, steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    loader = DataLoader(
        dataset, batch_size=min(16, args.examples),
        collate_fn=partial(collate_semantic_lm, pad_id=vocab.pad_id),
    )
    store = {key: [] for key in (
        "low", "high", "code", "group", "remaining", "length", "first",
        "last", "answer", "high_error", "bridge_error", "low_bridge_error",
        "prior_error", "mono_violation", "gar_pair", "gar_std", "support_auc",
        "reach_auc",
    )}
    offset = 0
    with torch.no_grad():
        for batch in loader:
            gpu = {key: value.to(args.device) for key, value in batch.items()}
            out = model(gpu["tokens"], gpu["prompt_len"], gpu["sentence_ends"])
            level = out["sentence_level"]
            mask = level["valid"]
            count = mask.sum(1).cpu().numpy()
            episode = np.arange(offset, offset + len(count))
            answers = np.asarray([
                dataset.igsm.problem(offset + row)[0].answer
                for row in range(len(count))
            ])
            offset += len(count)
            raw_valid = level["raw_action_valid"][mask]
            raw_ids = level["raw_action_ids"][mask]
            last_index = raw_valid.sum(1).sub(1).clamp_min(0)
            store["low"].append(level["low_target"][mask].cpu())
            store["high"].append(level["target"][mask].cpu())
            store["code"].append(level["codes"][mask].cpu())
            store["group"].append(np.repeat(episode, count))
            store["remaining"].append(level["remaining_target"][mask].cpu().numpy())
            store["length"].append(raw_valid.sum(1).cpu().numpy())
            store["first"].append(raw_ids[:, 0].cpu().numpy())
            store["last"].append(
                raw_ids.gather(1, last_index[:, None]).squeeze(1).cpu().numpy()
            )
            store["answer"].append(np.repeat(answers, count))
            store["high_error"].append(normalized_error(
                level["pred"][mask], level["target"][mask]
            ).cpu())
            store["bridge_error"].append(normalized_error(
                level["low_endpoint_high"][mask], level["target"][mask]
            ).cpu())
            store["low_bridge_error"].append(normalized_error(
                level["high_target_low"][mask], level["low_target"][mask]
            ).cpu())
            store["prior_error"].append(normalized_error(
                level["macro_p_mu"][mask], level["codes"][mask]
            ).cpu())
            pair_valid = mask[:, 1:] & mask[:, :-1]
            if pair_valid.any():
                violation = (
                    level["value"][:, 1:] > level["value"][:, :-1]
                )[pair_valid]
                store["mono_violation"].append(violation.float().cpu())
            positive = model.macro_support(level["prev"], level["codes"])
            negative = model.macro_support(level["prev"], level["codes"].roll(1, 0))
            valid_score = mask
            if valid_score.any():
                y = np.r_[np.ones(int(valid_score.sum())), np.zeros(int(valid_score.sum()))]
                s = np.r_[positive[valid_score].cpu().numpy(), negative[valid_score].cpu().numpy()]
                store["support_auc"].append(roc_auc_score(y, s))
            reach_pos = model.reachability(level["low_start"], level["target"])
            reach_neg = model.reachability(level["low_start"], level["target"].roll(1, 0))
            if valid_score.any():
                s = np.r_[reach_pos[valid_score].cpu().numpy(), reach_neg[valid_score].cpu().numpy()]
                store["reach_auc"].append(roc_auc_score(y, s))
            cf = model.sentence_counterfactuals(
                out, gpu["tokens"], gpu["prompt_len"], k=args.gar_k,
                source="nearest",
            )
            store["gar_pair"].append(pairwise_accuracy(
                cf["value"], cf["advantage_target"]
            ))
            store["gar_std"].append(float(cf["advantage_target"].std()))

    tensors = {key: torch.cat(store[key])[:args.max_points] for key in (
        "low", "high", "code", "high_error", "bridge_error",
        "low_bridge_error", "prior_error",
    )}
    arrays = {
        key: np.concatenate(store[key])[:args.max_points]
        for key in ("group", "remaining", "length", "first", "last", "answer")
    }
    results = {
        "examples": args.examples,
        "uses_symbolic_feasibility": False,
        "uses_auxiliary_lm": False,
        "state_spaces_are_distinct": True,
        "sentence_state_boundary": "last token of each completed sentence",
        "low_state": {
            "std": feature_std(tensors["low"]),
            "effective_rank": effective_rank(tensors["low"][:4096]),
            "remaining_r2": safe_probe(regression_probe,
                tensors["low"].numpy(), arrays["remaining"], 1, arrays["group"]
            ),
            "answer_accuracy": safe_probe(classification_probe,
                tensors["low"].numpy(), arrays["answer"], 2, arrays["group"]
            ),
        },
        "sentence_state": {
            "std": feature_std(tensors["high"]),
            "effective_rank": effective_rank(tensors["high"][:4096]),
            "remaining_r2": safe_probe(regression_probe,
                tensors["high"].numpy(), arrays["remaining"], 3, arrays["group"]
            ),
            "answer_accuracy": safe_probe(classification_probe,
                tensors["high"].numpy(), arrays["answer"], 4, arrays["group"]
            ),
            "prediction_mse": float(tensors["high_error"].mean()),
        },
        "macro_action": {
            "dimension": int(tensors["code"].shape[-1]),
            "std": feature_std(tensors["code"]),
            "effective_rank": effective_rank(tensors["code"][:4096]),
            "sentence_length_r2": safe_probe(regression_probe,
                tensors["code"].numpy(), arrays["length"], 5, arrays["group"]
            ),
            "first_token_accuracy": safe_probe(classification_probe,
                tensors["code"].numpy(), arrays["first"], 6, arrays["group"]
            ),
            "last_token_accuracy": safe_probe(classification_probe,
                tensors["code"].numpy(), arrays["last"], 7, arrays["group"]
            ),
            "conditional_prior_mse": float(tensors["prior_error"].mean()),
        },
        "interfaces": {
            "low_rollout_to_high_target_mse": float(tensors["bridge_error"].mean()),
            "high_target_to_low_target_mse": float(tensors["low_bridge_error"].mean()),
            "value_monotonicity_violation_rate": float(torch.cat(store["mono_violation"]).mean()) if store["mono_violation"] else float("nan"),
            "support_auroc": float(np.mean(store["support_auc"])),
            "reachability_auroc": float(np.mean(store["reach_auc"])),
            "gar_pairwise_accuracy": float(np.nanmean(store["gar_pair"])),
            "counterfactual_advantage_std": float(np.mean(store["gar_std"])),
        },
    }
    destination = Path(args.ckpt).parent / "sentence_hierarchy_audit.json"
    destination.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
