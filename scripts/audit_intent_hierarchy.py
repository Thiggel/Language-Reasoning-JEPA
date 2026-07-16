"""Direct high-level versus recursively composed low-level dynamics audit."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from textjepa.training.trainer import to_device
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import (
    build_dataset,
    collate_for,
    load_run,
)


def distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        F.layer_norm(x, x.shape[-1:])
        - F.layer_norm(y, y.shape[-1:])
    ).abs().mean(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=640)
    parser.add_argument("--steps-range", nargs=2, type=int)
    parser.add_argument("--n-vars-range", nargs=2, type=int)
    parser.add_argument("--max-high-horizon", type=int, default=4)
    parser.add_argument("--max-low-horizon", type=int, default=8)
    args = parser.parse_args()
    seed_everything(123)
    model, vocab, cfg = load_run(args.ckpt, args.device)
    if args.steps_range:
        cfg.data.steps_range = list(args.steps_range)
    if args.n_vars_range:
        cfg.data.n_vars_range = list(args.n_vars_range)
    dataset = build_dataset(cfg, vocab, "val", size=args.examples)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_for(cfg), pad_id=vocab.pad_id),
    )
    direct_sum = recursive_sum = reach_sum = value_sum = count = 0.0
    horizon_sums = [0.0] * args.max_high_horizon
    horizon_counts = [0.0] * args.max_high_horizon
    low_horizon_sums = [0.0] * args.max_low_horizon
    low_horizon_counts = [0.0] * args.max_low_horizon
    K = int(cfg.model.macro_k)
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, torch.device(args.device))
            out = model(batch)
            mask = out.hi_mask.float()
            direct = distance(out.hi_preds, out.hi_targets)
            cur = out.extras["hi_low_rollout_target"]
            recursive = distance(cur, out.hi_targets)
            reach = distance(out.hi_preds, cur)
            direct_sum += float((direct * mask).sum())
            recursive_sum += float((recursive * mask).sum())
            reach_sum += float((reach * mask).sum())
            if "hi_value_pred" in out.extras:
                target = out.extras["hi_value_target"] * 5.0
                value_sum += float(
                    ((out.extras["hi_value_pred"] - target).abs() * mask).sum()
                )
            count += float(mask.sum())
            T = out.actions.shape[1]
            low_limit = min(args.max_low_horizon, T)
            for horizon in range(1, low_limit + 1):
                n_origins = T - horizon + 1
                cur = out.prev_states[:, :n_origins].reshape(
                    out.prev_states.shape[0] * n_origins, -1
                )
                for offset in range(horizon):
                    action = out.actions[
                        :, offset:offset + n_origins
                    ].reshape(cur.shape[0], -1)
                    cur = model.core.predictor(cur, action)
                prediction = cur.reshape(
                    out.prev_states.shape[0], n_origins, -1
                )
                target = out.step_states_tgt[:, horizon - 1:]
                valid = out.step_mask[:, horizon - 1:].float()
                low_horizon_sums[horizon - 1] += float(
                    (distance(prediction, target) * valid).sum()
                )
                low_horizon_counts[horizon - 1] += float(valid.sum())
            macro = out.extras["macro_codes"]
            S = macro.shape[1]
            starts = torch.arange(
                0, K * S, K, device=out.prev_states.device
            )
            origins = out.prev_states[:, starts]
            limit = min(args.max_high_horizon, S)
            for horizon in range(1, limit + 1):
                n_origins = S - horizon + 1
                code_windows = torch.stack([
                    macro[:, start:start + horizon]
                    for start in range(n_origins)
                ], dim=1)
                prediction = model.core._high_rollout(
                    origins[:, :n_origins].reshape(
                        origins.shape[0] * n_origins, -1
                    ),
                    code_windows.reshape(
                        origins.shape[0] * n_origins,
                        horizon,
                        macro.shape[-1],
                    ),
                )[:, -1].reshape(origins.shape[0], n_origins, -1)
                target = out.hi_targets[:, horizon - 1:]
                valid = out.hi_mask[:, horizon - 1:].float()
                horizon_sums[horizon - 1] += float(
                    (distance(prediction, target) * valid).sum()
                )
                horizon_counts[horizon - 1] += float(valid.sum())
    direct = direct_sum / count
    recursive = recursive_sum / count
    result = {
        "checkpoint": args.ckpt,
        "examples": args.examples,
        "valid_windows": int(count),
        "direct_high_l1": direct,
        "recursive_low_l1": recursive,
        "high_to_low_rollout_l1": reach_sum / count,
        "relative_reduction": 1.0 - direct / recursive,
        "high_value_mae": value_sum / count,
        "recursive_high_l1_by_horizon": {
            str(horizon + 1): horizon_sums[horizon] /
            horizon_counts[horizon]
            for horizon in range(args.max_high_horizon)
            if horizon_counts[horizon]
        },
        "recursive_low_l1_by_horizon": {
            str(horizon + 1): low_horizon_sums[horizon] /
            low_horizon_counts[horizon]
            for horizon in range(args.max_low_horizon)
            if low_horizon_counts[horizon]
        },
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
