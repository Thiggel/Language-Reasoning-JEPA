"""Representation and drift audit for the non-symbolic faithful token editor."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.data import DataLoader

from textjepa.utils.checkpoint import build_dataset, collate_for, load_run
from textjepa.utils.metrics import effective_rank, feature_std


def masked_mean(values, mask):
    return float(values[mask].mean()) if mask.any() else 0.0


def probe(x, y, classification=False):
    split = max(1, int(0.7 * len(x)))
    if split >= len(x) or len(np.unique(y[:split])) < 2:
        return None
    model = (
        LogisticRegression(max_iter=500, class_weight="balanced")
        if classification else Ridge(alpha=1.0)
    )
    model.fit(x[:split], y[:split])
    prediction = model.predict(x[split:])
    return float(
        accuracy_score(y[split:], prediction)
        if classification else r2_score(y[split:], prediction)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--out")
    args = parser.parse_args()
    model, vocab, cfg = load_run(args.ckpt, args.device)
    dataset = build_dataset(cfg, vocab, "val", size=args.examples)
    loader = DataLoader(
        dataset, batch_size=min(16, cfg.train.batch_size),
        collate_fn=partial(collate_for(cfg), pad_id=vocab.pad_id),
    )
    direct, recursive, states, remaining, ops, delta = [], [], [], [], [], []
    high, ldad_correct, ldad_total, goal_distance, goal_remaining = [], 0, 0, [], []
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(args.device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            out = model(batch)
            mask = out.step_mask
            norm = lambda x: F.layer_norm(x, x.shape[-1:])
            direct.append((norm(out.preds) - norm(out.step_states_tgt)).abs().mean(-1)[mask])
            recursive.append((norm(out.rollout) - norm(out.step_states_tgt)).abs().mean(-1)[mask])
            states.append(out.step_states[mask])
            remaining.append(batch["remaining"][mask])
            ops.append(batch["op"][mask])
            delta.append((out.step_states - out.prev_states)[mask])
            last = mask.sum(1).clamp_min(1) - 1
            goal = out.step_states_tgt[torch.arange(mask.shape[0], device=mask.device), last]
            distance = (norm(out.step_states_tgt) - norm(goal).unsqueeze(1)).abs().mean(-1)
            goal_distance.append(distance[mask])
            goal_remaining.append(batch["remaining"][mask])
            if out.hi_preds is not None:
                high.append((norm(out.hi_preds) - norm(out.hi_targets)).abs().mean(-1)[out.hi_mask])
            logits = out.extras.get("observed_action_logits")
            if logits is not None:
                length = min(logits.shape[-2], batch["action_tokens"].shape[-1])
                target = batch["action_tokens"][..., :length]
                valid = mask.unsqueeze(-1) & target.ne(vocab.pad_id)
                ldad_correct += int(logits[..., :length, :].argmax(-1)[valid].eq(target[valid]).sum())
                ldad_total += int(valid.sum())
    state = torch.cat(states)
    rem = torch.cat(remaining).cpu().numpy()
    op = torch.cat(ops).cpu().numpy()
    displacement = torch.cat(delta).cpu().numpy()
    geometry = torch.cat(goal_distance).cpu().numpy()
    geometry_remaining = torch.cat(goal_remaining).cpu().numpy()
    payload = {
        "examples": len(dataset),
        "one_step_ln_l1": float(torch.cat(direct).mean()),
        "recursive_ln_l1": float(torch.cat(recursive).mean()),
        "macro_ln_l1": float(torch.cat(high).mean()) if high else None,
        "state_std": float(feature_std(state)),
        "state_effective_rank": float(effective_rank(state[:4096])),
        "remaining_edit_probe_r2": probe(state.cpu().numpy(), rem),
        "operation_from_displacement_accuracy": probe(displacement, op, True),
        "terminal_geometry_remaining_correlation": float(
            np.corrcoef(geometry, geometry_remaining)[0, 1]
        ),
        "ldad_token_accuracy": ldad_correct / max(ldad_total, 1),
        "symbolic_reasoning_labels_used": False,
    }
    destination = Path(args.out or Path(args.ckpt).parent / "token_edit_audit.json")
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
