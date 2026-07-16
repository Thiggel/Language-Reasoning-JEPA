"""Layer/level representation and failure probes for token hierarchy v2."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.lm import LMDataset, collate_lm
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA
from textjepa.utils.metrics import effective_rank, feature_std


def ln_l1(x, y):
    return (F.layer_norm(x, x.shape[-1:]) - F.layer_norm(y, y.shape[-1:])).abs().mean(-1)


def classification_probe(x, y, split, groups=None):
    fold = np.arange(len(y)) if groups is None else groups
    train = fold % 5 != split
    test = ~train
    if len(np.unique(y[train])) < 2 or not test.any():
        return float("nan")
    model = RidgeClassifier(alpha=1.0).fit(x[train], y[train])
    return float(accuracy_score(y[test], model.predict(x[test])))


def regression_probe(x, y, split, groups=None):
    fold = np.arange(len(y)) if groups is None else groups
    train = fold % 5 != split
    test = ~train
    model = Ridge(alpha=1.0).fit(x[train], y[train])
    return float(r2_score(y[test], model.predict(x[test])))


def token_labels(ids, vocab):
    words = [vocab.id_to_token[int(i)] for i in ids]
    kind = []
    for word in words:
        if word.isdigit():
            kind.append(1)
        elif word in {".", "?", "="}:
            kind.append(2)
        elif word in {"plus", "minus", "times"}:
            kind.append(3)
        elif word in {"so", "the", "number", "of", "is"}:
            kind.append(4)
        else:
            kind.append(5)
    return np.asarray(kind)


def boundary_labels(action_ids, valid, period_id):
    """Position since and distance until the nearest sentence boundary."""
    positions = np.zeros(action_ids.shape, dtype=np.int64)
    distances = np.zeros(action_ids.shape, dtype=np.int64)
    for row in range(action_ids.shape[0]):
        length = int(valid[row].sum())
        since = 0
        for col in range(length):
            positions[row, col] = since
            since = 0 if action_ids[row, col] == period_id else since + 1
        until = 0
        for col in reversed(range(length)):
            distances[row, col] = until
            until = 0 if action_ids[row, col] == period_id else until + 1
    return positions, distances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=2000)
    parser.add_argument("--max-points", type=int, default=20000)
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
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
    stores = None
    goal_errors, support_pos, support_neg = [], [], []
    direct_errors, recursive_errors = [], []
    answers = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            tokens = batch["tokens"].to(args.device)
            prompt_len = batch["prompt_len"].to(args.device)
            out = model(tokens, prompt_len)
            if stores is None:
                stores = [dict(x=[], token=[], kind=[], remaining=[], answer=[], group=[],
                               sentence_position=[], boundary_distance=[],
                               code=[], first=[], last=[])
                          for _ in range(1 + len(out["levels"]))]
            batch_answers = np.asarray([
                dataset.igsm.problem(batch_index * 32 + i)[0].answer
                for i in range(tokens.shape[0])
            ])
            episode_ids = batch_index * 32 + np.arange(tokens.shape[0])
            mask = out["valid"]
            ids = out["action_ids"][mask].cpu().numpy()
            n_per = mask.sum(1).cpu().numpy()
            raw_ids = out["action_ids"].cpu().numpy()
            raw_valid = mask.cpu().numpy()
            sentence_position, boundary_distance = boundary_labels(
                raw_ids, raw_valid, vocab.token_to_id["."]
            )
            stores[0]["x"].append(out["target"][mask].cpu())
            stores[0]["token"].append(ids)
            stores[0]["kind"].append(token_labels(ids, vocab))
            stores[0]["remaining"].append(out["low_remaining_target"][mask].cpu().numpy())
            stores[0]["answer"].append(np.repeat(batch_answers, n_per))
            stores[0]["group"].append(np.repeat(episode_ids, n_per))
            stores[0]["sentence_position"].append(sentence_position[raw_valid])
            stores[0]["boundary_distance"].append(boundary_distance[raw_valid])
            goal_errors.append(ln_l1(out["goal_pred"], out["final_target"]).cpu())
            for level_index, level in enumerate(out["levels"], start=1):
                valid = level["valid"]
                endpoint_ids = level["raw_action_ids"][..., -1][valid].cpu().numpy()
                n_level = valid.sum(1).cpu().numpy()
                store = stores[level_index]
                store["x"].append(level["target"][valid].cpu())
                store["token"].append(endpoint_ids)
                store["kind"].append(token_labels(endpoint_ids, vocab))
                store["remaining"].append(level["remaining_target"][valid].cpu().numpy())
                store["answer"].append(np.repeat(batch_answers, n_level))
                store["group"].append(np.repeat(episode_ids, n_level))
                endpoint_index = (
                    (np.arange(valid.shape[1]) + 1) * int(level["span"]) - 1
                )
                endpoint_index = np.broadcast_to(
                    endpoint_index[None], valid.shape
                )[valid.cpu().numpy()]
                endpoint_batch = np.broadcast_to(
                    np.arange(valid.shape[0])[:, None], valid.shape
                )[valid.cpu().numpy()]
                store["sentence_position"].append(
                    sentence_position[endpoint_batch, endpoint_index]
                )
                store["boundary_distance"].append(
                    boundary_distance[endpoint_batch, endpoint_index]
                )
                store["code"].append(level["codes"][valid].cpu())
                store["first"].append(level["raw_action_ids"][..., 0][valid].cpu().numpy())
                store["last"].append(endpoint_ids)
                direct_errors.append((level_index, ln_l1(level["pred"][valid], level["target"][valid]).cpu()))
                recursive_errors.append((level_index, ln_l1(level["recursive_low_endpoint"][valid], level["target"][valid]).cpu()))
                support_pos.append(level["support_pos"][valid].cpu())
                support_neg.append(level["support_neg"][valid].cpu())
    results = {"goal_l1": float(torch.cat(goal_errors).mean())}
    for level_index, store in enumerate(stores or []):
        name = "token" if level_index == 0 else f"level{level_index}"
        x = torch.cat(store["x"])[:args.max_points]
        arrays = {key: np.concatenate(store[key])[:len(x)] for key in (
            "token", "kind", "remaining", "answer", "sentence_position",
            "boundary_distance", "group",
        )}
        xn = x.numpy()
        results[name] = {
            "std": feature_std(x), "effective_rank": effective_rank(x[:4096]),
            "token_identity_accuracy": classification_probe(
                xn, arrays["token"], 0, arrays["group"]
            ),
            "token_type_accuracy": classification_probe(
                xn, arrays["kind"], 1, arrays["group"]
            ),
            "remaining_fraction_r2": regression_probe(
                xn, arrays["remaining"], 2, arrays["group"]
            ),
            "final_answer_accuracy": classification_probe(
                xn, arrays["answer"], 3, arrays["group"]
            ),
            "sentence_position_r2": regression_probe(
                xn, arrays["sentence_position"], 0, arrays["group"]
            ),
            "boundary_distance_r2": regression_probe(
                xn, arrays["boundary_distance"], 1, arrays["group"]
            ),
        }
        numeric = arrays["kind"] == 1
        results[name]["numeric_token_accuracy"] = (
            classification_probe(
                xn[numeric], arrays["token"][numeric], 2,
                arrays["group"][numeric],
            )
            if numeric.sum() >= 20 else float("nan")
        )
        if level_index:
            code = torch.cat(store["code"])[:args.max_points].numpy()
            first = np.concatenate(store["first"])[:len(code)]
            last = np.concatenate(store["last"])[:len(code)]
            results[name].update(
                macro_first_token_accuracy=classification_probe(
                    code, first, 0, arrays["group"][:len(code)]
                ),
                macro_last_token_accuracy=classification_probe(
                    code, last, 1, arrays["group"][:len(code)]
                ),
                macro_action_std=feature_std(torch.from_numpy(code)),
                macro_action_rank=effective_rank(torch.from_numpy(code[:4096])),
            )
            direct = torch.cat([v for i, v in direct_errors if i == level_index])
            recursive = torch.cat([v for i, v in recursive_errors if i == level_index])
            results[name]["direct_prediction_l1"] = float(direct.mean())
            results[name]["recursive_low_l1"] = float(recursive.mean())
    if support_pos:
        pos, neg = torch.cat(support_pos).numpy(), torch.cat(support_neg).numpy()
        results["support_auc"] = float(roc_auc_score(
            np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]
        ))
    if stores and len(stores) > 1:
        base = results["token"]
        results["abstraction_summary"] = {
            f"level{i}_answer_gain": results[f"level{i}"]["final_answer_accuracy"]
            - base["final_answer_accuracy"]
            for i in range(1, len(stores))
        }
        results["abstraction_summary"].update({
            f"level{i}_token_identity_drop": base["token_identity_accuracy"]
            - results[f"level{i}"]["token_identity_accuracy"]
            for i in range(1, len(stores))
        })
    dest = Path(args.ckpt).parent / "representation_probes.json"
    dest.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
