"""Head-only GAR adaptation on frozen-policy replay or teacher states.

The clean target is used to label candidate edit advantages during training.
It is never passed to the GAR head.  This makes the resulting checkpoint a
target-privileged training diagnostic with deployment-feasible inference.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from scripts.plan_faithful_token_edits import (
        OPS, apply_edit, buffer_distance, copy_buffer, gar_scores,
        pad_token_state_for_insertions, proposal_tokens, propose_edits,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from plan_faithful_token_edits import (
        OPS, apply_edit, buffer_distance, copy_buffer, gar_scores,
        pad_token_state_for_insertions, proposal_tokens, propose_edits,
    )
from textjepa.utils.checkpoint import build_dataset, load_run


def candidate_labels(current, target, candidates):
    before = buffer_distance(current, target)
    labels = []
    for action in candidates:
        outcome = copy_buffer(current)
        apply_edit(outcome, action)
        labels.append(float(before - buffer_distance(outcome, target)))
    return labels


@torch.no_grad()
def candidate_features(model, buffer, candidates, pad_id, device):
    try:
        from scripts.plan_faithful_token_edits import _buffer_tensor
    except ModuleNotFoundError:
        from plan_faithful_token_edits import _buffer_tensor

    raw = _buffer_tensor(buffer, pad_id, device)
    states, mask = model.encode_token_buffers(raw, mode="online")
    states, mask = pad_token_state_for_insertions(states[:, 0], mask[:, 0])
    pooled = model._pool_tokens(states, mask)
    operations = torch.tensor([OPS[a[0]] for a in candidates], device=device)
    positions = torch.tensor([a[1] for a in candidates], device=device)
    content_ids = torch.tensor([
        pad_id if a[2] is None else a[2] for a in candidates
    ], device=device)
    content = model.chunk_encoder.tok(content_ids)
    actions = model.token_pred.encode_action(
        states.expand(len(candidates), -1, -1),
        mask.expand(len(candidates), -1), operations, positions, content,
    )
    return torch.cat([pooled.expand(len(candidates), -1), actions], dim=-1)


def select_states(model, item, regime, depth, candidates, pad_id, device, rng):
    target = item["buffers"][-1]
    if regime == "teacher":
        upper = max(1, len(item["buffers"]) - 1)
        return [copy_buffer(item["buffers"][min(step, upper - 1)])
                for step in range(depth)]
    current = copy_buffer(item["buffers"][0])
    states = []
    for _ in range(depth):
        states.append(copy_buffer(current))
        pool = proposal_tokens(item["prompt"], current, "current_buffer")
        proposed = propose_edits(current, pool, candidates, rng)
        if not proposed:
            break
        values = gar_scores(model, current, proposed, pad_id, device)
        apply_edit(current, proposed[max(range(len(values)), key=values.__getitem__)])
        if current == target:
            break
    return states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--regime", choices=["replay", "teacher"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.examples, args.depth, args.candidates, args.epochs) < 1:
        parser.error("integer budgets must be positive")

    torch.manual_seed(args.seed)
    model, vocab, cfg = load_run(args.ckpt, args.device)
    model.requires_grad_(False)
    model.gar_head.requires_grad_(True)
    model.eval()
    dataset = build_dataset(cfg, vocab, "train", size=args.examples)
    # Materialize the state distribution before the first optimizer step so
    # the replay behavior policy is the immutable input checkpoint, not the
    # progressively adapted head.
    records = []
    for index in range(len(dataset)):
        item = dataset[index]
        rng = random.Random(f"gar-adapt-states:{args.seed}:{index}")
        for state_index, current in enumerate(select_states(
            model, item, args.regime, args.depth, args.candidates,
            vocab.pad_id, args.device, rng,
        )):
            records.append((index, state_index, item, current))
    optimizer = torch.optim.AdamW(model.gar_head.parameters(), lr=args.lr)
    updates = 0
    losses = []
    positive_rates = []
    for epoch in range(args.epochs):
        order = list(range(len(records)))
        random.Random(f"gar-adapt-order:{args.seed}:{epoch}").shuffle(order)
        for record_index in order:
            index, state_index, item, current = records[record_index]
            rng = random.Random(
                f"gar-adapt-candidates:{args.seed}:{epoch}:{index}:{state_index}"
            )
            target = item["buffers"][-1]
            tokens = proposal_tokens(item["prompt"], current, "current_buffer")
            proposed = propose_edits(current, tokens, args.candidates, rng)
            if len(proposed) < 2:
                continue
            labels = torch.tensor(candidate_labels(current, target, proposed),
                                  device=args.device)
            features = candidate_features(
                model, current, proposed, vocab.pad_id, args.device
            )
            prediction = model.gar_head(features).squeeze(-1)
            regression = F.mse_loss(prediction, labels)
            better = labels[:, None] > labels[None, :]
            if better.any():
                pairwise = F.softplus(
                    -(prediction[:, None] - prediction[None, :])[better]
                ).mean()
            else:
                pairwise = prediction.sum() * 0
            loss = 0.25 * regression + pairwise
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
            positive_rates.append(float((labels > 0).float().mean()))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_dir / "model"
    model_dir.mkdir(exist_ok=True)
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    payload["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(payload, model_dir / "best.pt")
    metrics = {
        "regime": args.regime,
        "information_regime": "oracle_target_advantage_training_diagnostic",
        "examples": args.examples, "depth": args.depth,
        "candidates": args.candidates, "epochs": args.epochs,
        "materialized_states": len(records),
        "updates": updates,
        "mean_loss": sum(losses) / max(len(losses), 1),
        "candidate_positive_rate": sum(positive_rates) / max(len(positive_rates), 1),
        "frozen_components": "all parameters except gar_head",
    }
    (args.out_dir / "adapt_metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
