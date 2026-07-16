"""Audit learned remaining-distance calibration for hierarchy switching."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch

from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import prompt_sentences
from textjepa.planning.hierarchical_search import HierarchicalLatentPlanner
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, load_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=400)
    args = parser.parse_args()
    seed_everything(811)
    model, vocab, cfg = load_run(args.ckpt, args.device)
    dataset = build_dataset(cfg, vocab, split="val", size=args.episodes)
    planner = HierarchicalLatentPlanner(
        model, vocab, torch.device(args.device), method="shooting"
    )
    values: dict[int, list[float]] = defaultdict(list)
    flat_correct: dict[int, list[float]] = defaultdict(list)
    with torch.no_grad():
        for index in range(args.episodes):
            problem, _ = dataset.problem(index)
            env = SymbolicEnv(problem)
            prompt = prompt_sentences(problem, random.Random(811 + index))
            prompt_tokens = planner._tokens(prompt)
            prompt_mask = torch.ones(
                1, len(prompt), dtype=torch.bool, device=planner.device
            )
            step_texts: list[str] = []
            while not env.solved:
                state = planner._current_state(
                    prompt_tokens, prompt_mask, step_texts
                )
                s0 = planner._s0(prompt_tokens, prompt_mask)
                remaining = env.remaining_necessary()
                values[remaining].append(float(model.value_head(state, s0)))
                feasible = env.feasible_actions()
                selected = planner._flat_value_action(
                    problem, feasible, state, s0
                )
                flat_correct[remaining].append(float(
                    selected in problem.query_ancestors
                ))
                necessary = [
                    action for action in feasible
                    if action in problem.query_ancestors
                ]
                step_texts.append(env.step(min(necessary)))
    result = {
        "checkpoint": args.ckpt,
        "states": sum(map(len, values.values())),
        "by_remaining": {},
    }
    for remaining, predictions in sorted(values.items()):
        tensor = torch.tensor(predictions)
        result["by_remaining"][str(remaining)] = {
            "count": len(predictions),
            "predicted_mean": float(tensor.mean()),
            "predicted_std": float(tensor.std(unbiased=False)),
            "mae": float((tensor - remaining).abs().mean()),
            "flat_first_action_accuracy": sum(flat_correct[remaining])
            / len(flat_correct[remaining]),
            **{
                f"macro_rate_threshold_{threshold}": float(
                    (tensor > threshold).float().mean()
                )
                for threshold in (2, 3, 4)
            },
        }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
