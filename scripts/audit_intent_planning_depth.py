"""Diagnose why deeper intent-JEPA search can select worse root actions.

The audit freezes a trained checkpoint and evaluates identical, balanced
candidate trees with learned latent transitions and exact symbolic
transitions.  It compares the deployed terminal-only sequence score with a
one-step root score and an oracle endpoint score.  Symbolic labels are used
only after model scoring.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from textjepa.analysis.intent_decisions import (
    aggregate_rank_metrics,
    spearman_tied,
    summarize_depth_decision,
)
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import prompt_sentences
from textjepa.planning.search import LatentPlanner, _sequences
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, load_run


def _exact_labels(env: SymbolicEnv, sequences: list[list[int | None]]):
    endpoint = []
    root_exact = {}
    root_necessary = {}
    for sequence in sequences:
        root = int(sequence[0])
        if root not in root_exact:
            clone = env.clone()
            clone.step(root)
            root_exact[root] = 1 + clone.remaining_necessary()
            root_necessary[root] = root in env.p.query_ancestors
        clone = env.clone()
        for action in sequence:
            if action is not None:
                clone.step(action)
        endpoint.append(clone.remaining_necessary())
    return np.asarray(endpoint, dtype=np.float64), root_exact, root_necessary


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    model, vocab, cfg = load_run(args.checkpoint, args.device)
    if cfg.data.get("name", "igsm") != "igsm":
        raise ValueError("planning-depth audit currently requires stylized iGSM")
    dataset = build_dataset(cfg, vocab, split=args.split, size=args.episodes)
    device = torch.device(args.device)
    rows = []
    metric_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    cross_rows: dict[str, list[float]] = defaultdict(list)

    for episode in range(args.episodes):
        problem, _ = dataset.problem(episode)
        env = SymbolicEnv(problem)
        prompt = prompt_sentences(problem, random.Random(args.seed + episode))
        latent = LatentPlanner(model, vocab, device, simulator="latent")
        prompt_tokens = latent._tokens(prompt)
        prompt_mask = torch.ones(1, len(prompt), dtype=torch.bool, device=device)
        history: list[str] = []
        action_history: list[int] = []
        step = 0
        while not env.solved:
            state = latent._current_state(prompt_tokens, prompt_mask, history)
            s0 = latent._s0(prompt_tokens, prompt_mask)
            state_history, action_codes = latent._causal_history(
                prompt_tokens, prompt_mask, history, problem, action_history
            )
            roots = env.feasible_actions()
            root_sequences = [[root] for root in roots]
            latent.lookahead = 1
            one_latent = latent._flat_costs(
                state, s0, problem, root_sequences, None,
                state_history, action_codes,
            ).cpu().numpy()
            symbolic = LatentPlanner(model, vocab, device, simulator="symbolic")
            symbolic.lookahead = 1
            one_symbolic = symbolic._symbolic_costs(
                problem, env, history, state, s0, prompt_tokens, prompt_mask,
                root_sequences, None,
            ).cpu().numpy()
            one_step = {
                "latent": dict(zip(roots, map(float, one_latent))),
                "symbolic": dict(zip(roots, map(float, one_symbolic))),
            }

            for cell in args.cells:
                depth, cap = map(int, cell.split(":"))
                sequences = _sequences(
                    problem, frozenset(env.resolved_set), depth, cap,
                    random.Random(f"{args.seed}:{episode}:{step}:{cell}"),
                )
                endpoint, root_exact, root_necessary = _exact_labels(env, sequences)
                latent.lookahead = depth
                symbolic.lookahead = depth
                costs = {
                    "latent": latent._flat_costs(
                        state, s0, problem, sequences, None,
                        state_history, action_codes,
                    ).cpu().numpy(),
                    "symbolic": symbolic._symbolic_costs(
                        problem, env, history, state, s0, prompt_tokens,
                        prompt_mask, sequences, None,
                    ).cpu().numpy(),
                }
                key = f"depth{depth}_cap{cap}"
                for simulator in ("latent", "symbolic"):
                    metrics, root_rows = summarize_depth_decision(
                        sequences, costs[simulator], endpoint, root_exact,
                        root_necessary, one_step[simulator],
                    )
                    metric_rows[(simulator, key)].append(metrics)
                    for row in root_rows:
                        rows.append({
                            "episode": episode, "step": step, "cell": key,
                            "simulator": simulator, **row,
                        })
                correlation = spearman_tied(costs["latent"], costs["symbolic"])
                if correlation is not None:
                    cross_rows[f"{key}:score_spearman"].append(correlation)
                latent_root = sequences[int(np.argmin(costs["latent"]))][0]
                symbolic_root = sequences[int(np.argmin(costs["symbolic"]))][0]
                cross_rows[f"{key}:selected_root_agreement"].append(
                    float(latent_root == symbolic_root)
                )

            necessary = [
                action for action in env.feasible_actions()
                if action in problem.query_ancestors
            ]
            chosen = min(necessary)
            history.append(env.step(chosen))
            action_history.append(chosen)
            step += 1

    summary = {}
    for simulator in ("latent", "symbolic"):
        summary[simulator] = {}
        for cell in args.cells:
            depth, cap = map(int, cell.split(":"))
            key = f"depth{depth}_cap{cap}"
            values = metric_rows[(simulator, key)]
            summary[simulator][key] = {
                score: aggregate_rank_metrics([value[score] for value in values])
                for score in (
                    "deployed_terminal_score", "one_step_root_score",
                    "oracle_endpoint_score",
                )
            }
            for scalar in (
                "sequence_score_endpoint_spearman",
                "selected_sequence_endpoint_regret", "selection_optimism",
            ):
                observed = [value[scalar] for value in values if value[scalar] is not None]
                summary[simulator][key][scalar] = (
                    float(np.mean(observed)) if observed else None
                )
    summary["latent_vs_symbolic"] = {
        key: float(np.mean(values)) for key, values in cross_rows.items()
    }
    return {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "protocol": "fixed-checkpoint-balanced-tree-depth-causal-audit-v1",
        "split": args.split,
        "episodes": args.episodes,
        "seed": args.seed,
        "cells": args.cells,
        "information_boundary": (
            "Both model scorers receive the same future feasible-action trees. "
            "Depth above one is candidate-privileged. Exact remaining actions "
            "and endpoint labels are used only for post-hoc diagnosis."
        ),
        "summary": summary,
        "root_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7411)
    parser.add_argument(
        "--cells", nargs="+",
        default=["1:64", "2:4", "2:8", "2:16", "2:32", "2:64",
                 "4:4", "4:8", "4:16", "4:32", "4:64"],
    )
    args = parser.parse_args()
    result = run(args)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
