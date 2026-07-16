"""Measure whether a high-level waypoint identifies its generating span."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import prompt_sentences
from textjepa.planning.hierarchical_search import HierarchicalLatentPlanner
from textjepa.planning.search import _sequences
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, load_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--anchors", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=128)
    args = parser.parse_args()
    seed_everything(719)
    model, vocab, cfg = load_run(args.ckpt, args.device)
    dataset = build_dataset(cfg, vocab, split="val", size=args.anchors * 3)
    planner = HierarchicalLatentPlanner(
        model, vocab, torch.device(args.device), method="shooting"
    )
    K = model.core.macro_k
    totals = {
        "exact_span": 0.0,
        "same_first_action": 0.0,
        "selected_first_necessary": 0.0,
        "retrieved_first_necessary": 0.0,
        "oracle_terminal_first_necessary": 0.0,
        "oracle_receding_first_necessary": 0.0,
        "matched_distance": 0.0,
        "nearest_distance": 0.0,
        "matched_rank": 0.0,
        "candidate_count": 0.0,
    }
    by_remaining: dict[int, dict[str, float]] = {}
    anchors = 0
    with torch.no_grad():
        for index in range(len(dataset)):
            if anchors >= args.anchors:
                break
            problem, _ = dataset.problem(index)
            env = SymbolicEnv(problem)
            prompt = prompt_sentences(problem, random.Random(719 + index))
            prompt_tokens = planner._tokens(prompt)
            prompt_mask = torch.ones(
                1, len(prompt), dtype=torch.bool, device=planner.device
            )
            step_texts: list[str] = []
            while not env.solved and anchors < args.anchors:
                seqs = _sequences(
                    problem,
                    frozenset(env.resolved_set),
                    K,
                    args.max_candidates,
                )
                seqs = [seq for seq in seqs if len(seq) == K]
                if len(seqs) >= 2:
                    state = planner._current_state(
                        prompt_tokens, prompt_mask, step_texts
                    )
                    s0 = planner._s0(prompt_tokens, prompt_mask)
                    actions = planner._action_codes(
                        problem, [action for seq in seqs for action in seq]
                    ).reshape(len(seqs), K, -1)
                    codes = model.core.macro_encoder(actions)
                    high = model.core.hi_predictor(
                        state.expand(len(seqs), -1), codes
                    )
                    low = state.expand(len(seqs), -1)
                    for step in range(K):
                        low = model.core.predictor(low, actions[:, step])
                    q = model.core.macro_value_head(
                        state.expand(len(seqs), -1),
                        s0.expand(len(seqs), -1),
                        codes,
                    )
                    selected = int(q.argmin())
                    terminal_costs = []
                    prefix_costs = []
                    for sequence in seqs:
                        clone = env.clone()
                        prefix = []
                        for action in sequence:
                            clone.step(action)
                            prefix.append(clone.remaining_necessary())
                        terminal_costs.append(prefix[-1])
                        weights = [0.5 ** step for step in range(K)]
                        prefix_costs.append(sum(
                            weight * remaining
                            for weight, remaining in zip(weights, prefix)
                        ) / sum(weights))
                    terminal_selected = min(
                        range(len(seqs)), key=lambda i: terminal_costs[i]
                    )
                    receding_selected = min(
                        range(len(seqs)),
                        key=lambda i: (terminal_costs[i], prefix_costs[i]),
                    )
                    remaining = env.remaining_necessary()
                    bucket = by_remaining.setdefault(remaining, {
                        "anchors": 0.0,
                        "selected_first_necessary": 0.0,
                        "oracle_terminal_first_necessary": 0.0,
                        "oracle_receding_first_necessary": 0.0,
                    })
                    bucket["anchors"] += 1.0
                    bucket["selected_first_necessary"] += float(
                        seqs[selected][0] in problem.query_ancestors
                    )
                    bucket["oracle_terminal_first_necessary"] += float(
                        seqs[terminal_selected][0] in problem.query_ancestors
                    )
                    bucket["oracle_receding_first_necessary"] += float(
                        seqs[receding_selected][0] in problem.query_ancestors
                    )
                    distances = planner._ln_l1(
                        low, high[selected].expand_as(low)
                    )
                    retrieved = int(distances.argmin())
                    rank = int(
                        (distances < distances[selected]).sum()
                    ) + 1
                    totals["exact_span"] += float(retrieved == selected)
                    totals["same_first_action"] += float(
                        seqs[retrieved][0] == seqs[selected][0]
                    )
                    totals["selected_first_necessary"] += float(
                        seqs[selected][0] in problem.query_ancestors
                    )
                    totals["retrieved_first_necessary"] += float(
                        seqs[retrieved][0] in problem.query_ancestors
                    )
                    totals["oracle_terminal_first_necessary"] += float(
                        seqs[terminal_selected][0] in problem.query_ancestors
                    )
                    totals["oracle_receding_first_necessary"] += float(
                        seqs[receding_selected][0] in problem.query_ancestors
                    )
                    totals["matched_distance"] += float(distances[selected])
                    totals["nearest_distance"] += float(distances[retrieved])
                    totals["matched_rank"] += rank
                    totals["candidate_count"] += len(seqs)
                    anchors += 1
                necessary = [
                    action for action in env.feasible_actions()
                    if action in problem.query_ancestors
                ]
                step_texts.append(env.step(min(necessary)))
    result = {
        "checkpoint": args.ckpt,
        "anchors": anchors,
        "metrics": {
            name: value / max(anchors, 1) for name, value in totals.items()
        },
        "by_remaining": {
            str(remaining): {
                name: (
                    value if name == "anchors"
                    else value / max(bucket["anchors"], 1.0)
                )
                for name, value in bucket.items()
            }
            for remaining, bucket in sorted(by_remaining.items())
        },
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
