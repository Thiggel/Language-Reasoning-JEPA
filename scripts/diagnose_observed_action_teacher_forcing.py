"""Teacher-forced audit of proposal support and JEPA reranking.

This diagnostic deliberately uses stored expert actions and privileged
feasibility labels.  It is not a deployment metric: it localizes whether a
closed-loop failure comes from proposal recall or latent value ordering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from textjepa.planning.catalogue import CatalogueLatentPlanner
from textjepa.utils.checkpoint import build_dataset, load_run


def _rate(count: int, total: int) -> float:
    return count / max(total, 1)


@torch.no_grad()
def diagnose(checkpoint: Path, device: str, split: str, episodes: int,
             top_ms: tuple[int, ...], beam_width: int) -> dict:
    model, vocab, cfg = load_run(checkpoint, device)
    dataset = build_dataset(cfg, vocab, split)
    if not hasattr(dataset, "episodes"):
        raise ValueError("teacher-forced audit requires recorded episodes")
    planner = CatalogueLatentPlanner(
        model, vocab, torch.device(device), simulation_depth=2,
        proposal_top_m=max(top_ms), beam_width=beam_width,
    )
    depth_two_planner = CatalogueLatentPlanner(
        model, vocab, torch.device(device), simulation_depth=2,
        proposal_top_m=8, beam_width=beam_width,
    )
    counts = {
        str(m): {
            "expert_in_support_top_m": 0,
            "prior_top1_is_expert": 0,
            "prior_top1_is_feasible": 0,
            "one_step_rerank_is_expert": 0,
            "one_step_rerank_is_feasible": 0,
        }
        for m in top_ms
    }
    depth_two_expert = 0
    depth_two_feasible = 0
    total = 0
    for episode in dataset.episodes[:episodes]:
        outcomes: list[str] = []
        actions: list[str] = []
        for transition in episode.transitions:
            catalogue = tuple(transition.catalogue)
            expert = transition.action
            if expert not in catalogue:
                raise RuntimeError("expert action absent from recorded catalogue")
            feasible = set(transition.available)
            s0, state_history, action_history = planner._observed_history(
                episode.prompt, outcomes, actions
            )
            codes = planner._action_codes(catalogue)
            anchor = state_history[:, -1]
            support = model.core.action_support_head(
                anchor.expand(len(catalogue), -1), codes
            )
            order = support.argsort(descending=True)
            for m in top_ms:
                chosen = order[:min(m, len(catalogue))]
                chosen_indices = chosen.tolist()
                chosen_actions = [catalogue[index] for index in chosen_indices]
                cell = counts[str(m)]
                cell["expert_in_support_top_m"] += int(expert in chosen_actions)
                prior_action = chosen_actions[0]
                cell["prior_top1_is_expert"] += int(prior_action == expert)
                cell["prior_top1_is_feasible"] += int(prior_action in feasible)
                future = codes[chosen].unsqueeze(1)
                leaf = planner._rollout(state_history, action_history, future)
                energy = model.value_head(
                    leaf, s0.expand(leaf.shape[0], -1)
                )
                reranked = catalogue[int(chosen[int(energy.argmin())])]
                cell["one_step_rerank_is_expert"] += int(reranked == expert)
                cell["one_step_rerank_is_feasible"] += int(reranked in feasible)
            depth_two = depth_two_planner.choose(
                episode.prompt, outcomes, actions, catalogue
            )
            depth_two_expert += int(depth_two == expert)
            depth_two_feasible += int(depth_two in feasible)
            total += 1
            actions.append(expert)
            outcomes.append(transition.outcome)
    rates = {
        m: {name: _rate(value, total) for name, value in cell.items()}
        for m, cell in counts.items()
    }
    return {
        "checkpoint": str(checkpoint),
        "split": split,
        "episodes": min(episodes, len(dataset.episodes)),
        "states": total,
        "information_boundary": (
            "candidate-privileged diagnostic: stored expert and feasibility "
            "labels are used only for localization"
        ),
        "by_top_m": rates,
        "depth_two_top8_beam_expert": _rate(depth_two_expert, total),
        "depth_two_top8_beam_feasible": _rate(depth_two_feasible, total),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--top-m", type=int, nargs="+", default=(1, 4, 8, 16, 32, 64))
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    top_ms = tuple(sorted(set(args.top_m)))
    if not top_ms or top_ms[0] < 1 or args.beam_width < 1:
        parser.error("top-M and beam width must be positive")
    payload = {
        path.parents[1].name: diagnose(
            path, args.device, args.split, args.episodes, top_ms,
            args.beam_width,
        )
        for path in args.checkpoint
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
