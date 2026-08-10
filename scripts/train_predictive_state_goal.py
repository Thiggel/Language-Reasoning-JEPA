#!/usr/bin/env python3
"""Train Stage 3B prompt-conditioned distance and direct-value baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from textjepa.analysis.predictive_state import progress_metrics
from textjepa.data.predictive_state import sha256_path, validate_reasoning_bundle
from textjepa.planning.predictive_state import (
    DirectValueModel,
    goal_distance_loss,
    make_goal_distance_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-size", type=int, default=128)
    parser.add_argument(
        "--distance-kind", choices=("euclidean", "quasimetric"),
        default="euclidean",
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def training_problem_ids(records: list[dict]) -> set[str]:
    problems = sorted(
        {str(record["problem_id"]) for record in records},
        key=lambda value: hashlib.sha256(value.encode()).digest(),
    )
    if len(problems) < 2:
        raise ValueError("goal training needs at least two problems")
    test_count = max(1, round(0.2 * len(problems)))
    return set(problems[test_count:])


def tensors(record: dict, device: str) -> dict:
    return {
        "state": record["states"].float().to(device),
        "prompt": record["prompt_state"].float().to(device),
        "valid": record["valid"].bool().to(device),
        "remaining": record["remaining_chunks"].float().to(device),
        "correct": torch.tensor([record["correct"]], device=device),
        "terminal": torch.tensor([record["terminal_index"]], device=device),
    }


@torch.no_grad()
def evaluate(distance_model, value_model, records, train_ids, device: str) -> dict:
    trajectories, value_logits, distance_values, labels = [], [], [], []
    for record in records:
        if str(record["problem_id"]) in train_ids:
            continue
        batch = tensors(record, device)
        distance = distance_model(batch["state"], batch["prompt"])
        valid = batch["valid"]
        if record["correct"]:
            trajectories.append(distance[valid].cpu())
        budget = batch["remaining"] / batch["remaining"].max().clamp_min(1)
        logit = value_model(batch["state"], batch["prompt"], budget)
        value_logits.append(logit[valid].cpu())
        distance_values.append(distance[valid].cpu())
        labels.append(torch.full_like(logit[valid].cpu(), float(record["correct"])))
    progress = progress_metrics(trajectories)
    if value_logits:
        logits = torch.cat(value_logits)
        distance = torch.cat(distance_values)
        target = torch.cat(labels)
        distance_probability = torch.exp(-distance).clamp(1e-6, 1 - 1e-6)
        distance_logit = torch.logit(distance_probability)
        hybrid = logits - distance
        direct_bce = F.binary_cross_entropy_with_logits(logits, target)
        hybrid_bce = F.binary_cross_entropy_with_logits(hybrid, target)
        progress.update({
            "distance_value_bce": float(
                F.binary_cross_entropy_with_logits(distance_logit, target)
            ),
            "distance_value_accuracy": float(
                ((distance_logit >= 0) == target.bool()).float().mean()
            ),
            "direct_value_bce": float(direct_bce),
            "direct_value_accuracy": float(((logits >= 0) == target.bool()).float().mean()),
            "hybrid_value_bce": float(hybrid_bce),
            "hybrid_value_accuracy": float(
                ((hybrid >= 0) == target.bool()).float().mean()
            ),
            "hybrid_bce_improvement_over_direct": float(direct_bce - hybrid_bce),
            "value_examples": len(target),
        })
    return progress


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    payload = torch.load(args.features, map_location="cpu", weights_only=True)
    validate_reasoning_bundle(payload)
    records = payload["records"]
    train_ids = training_problem_ids(records)
    train_records = [
        record for record in records
        if str(record["problem_id"]) in train_ids
    ]
    if not train_records:
        raise ValueError("no training trajectories")
    width = int(train_records[0]["states"].shape[-1])
    distance_model = make_goal_distance_model(
        args.distance_kind, width, args.geometry_size
    ).to(args.device)
    value_model = DirectValueModel(width).to(args.device)
    optimizer = torch.optim.AdamW(
        list(distance_model.parameters()) + list(value_model.parameters()),
        lr=args.learning_rate, weight_decay=1e-3,
    )
    history = []
    for step in range(1, args.steps + 1):
        record = train_records[(step - 1) % len(train_records)]
        batch = tensors(record, args.device)
        distance = distance_model(batch["state"], batch["prompt"])
        metric_loss = goal_distance_loss(
            distance[None], valid=batch["valid"][None],
            remaining_chunks=batch["remaining"][None],
            correct=batch["correct"], terminal_indices=batch["terminal"],
        )
        budget = batch["remaining"] / batch["remaining"].max().clamp_min(1)
        value_logit = value_model(batch["state"], batch["prompt"], budget)
        value_target = torch.full_like(value_logit, float(record["correct"]))
        value_loss = F.binary_cross_entropy_with_logits(
            value_logit[batch["valid"]], value_target[batch["valid"]]
        )
        loss = metric_loss.total + value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(distance_model.parameters()) + list(value_model.parameters()), 1.0
        )
        optimizer.step()
        if step == 1 or step % 100 == 0:
            row = {
                "step": step, "total": float(loss.detach()),
                "terminal": float(metric_loss.terminal.detach()),
                "time": float(metric_loss.time.detach()),
                "monotonicity": float(metric_loss.monotonicity.detach()),
                "negative": float(metric_loss.negative.detach()),
                "value_bce": float(value_loss.detach()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    distance_model.eval()
    value_model.eval()
    held_out = evaluate(
        distance_model, value_model, records, train_ids, args.device
    )
    checkpoint = {
        "schema_version": 1,
        "kind": "predictive_state_goal_distance_and_value_v1",
        "state_size": width,
        "geometry_size": args.geometry_size,
        "distance_kind": args.distance_kind,
        "features_sha256": sha256_path(args.features),
        "features_metadata": payload["metadata"],
        "distance_model": distance_model.state_dict(),
        "value_model": value_model.state_dict(),
        "training": vars(args) | {"features": str(args.features), "output": str(args.output)},
        "history": history,
        "held_out": held_out,
        "oracle_terminal_states_in_training": True,
        "candidate_privileged_verified_outcomes": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({
        "schema_version": 1,
        "held_out": held_out,
        "history": history,
        "oracle_terminal_states_in_training": True,
        "candidate_privileged_verified_outcomes": True,
        "direct_value_is_principal_baseline": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"held_out": held_out}), flush=True)


if __name__ == "__main__":
    main()
