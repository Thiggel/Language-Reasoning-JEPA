"""Localize intent-JEPA planning-depth failure with paired interventions.

Every learned cell receives the same symbolic feasible-action tree.  Exact
states and exact remaining-step labels are used only in explicitly privileged
diagnostic cells or for post-hoc measurements.
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
    transition_metrics,
)
from textjepa.analysis.intent_depth_localization import search_symbolic_tree
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import prompt_sentences
from textjepa.planning.search import LatentPlanner, _sequences
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, load_run


def _exact_sequence_states(
    planner: LatentPlanner,
    env: SymbolicEnv,
    history: list[str],
    prompt_tokens: torch.Tensor,
    prompt_mask: torch.Tensor,
    sequences: list[list[int | None]],
    current: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Encode exact predecessor/successor pairs for candidate sequences."""

    rendered: list[list[str]] = []
    remaining = []
    active_sequences: list[list[int]] = []
    for sequence in sequences:
        actions = [int(action) for action in sequence if action is not None]
        if not actions:
            raise ValueError("exact transition audit requires non-empty sequences")
        clone = env.clone()
        future = [clone.step(action) for action in actions]
        rendered.append(history + future)
        active_sequences.append(actions)
        remaining.append(clone.remaining_necessary())

    n = len(rendered)
    chunks = max(len(texts) for texts in rendered)
    length = max(
        len(planner.vocab.encode(text))
        for texts in rendered for text in texts
    )
    tokens = torch.full(
        (n, chunks, length), planner.vocab.pad_id,
        dtype=torch.long, device=planner.device,
    )
    mask = torch.zeros(n, chunks, dtype=torch.bool, device=planner.device)
    for row, texts in enumerate(rendered):
        for column, text in enumerate(texts):
            ids = planner.vocab.encode(text)
            tokens[row, column, :len(ids)] = torch.tensor(
                ids, device=planner.device
            )
            mask[row, column] = True
    _, states = planner.model.encode_states(
        prompt_tokens.expand(n, -1, -1), prompt_mask.expand(n, -1),
        tokens, mask,
    )
    last = mask.sum(1) - 1
    rows = torch.arange(n, device=planner.device)
    successor = states[rows, last]
    predecessor = current.expand(n, -1).clone()
    multi_step = torch.tensor(
        [len(actions) > 1 for actions in active_sequences],
        dtype=torch.bool, device=planner.device,
    )
    if multi_step.any():
        selected = rows[multi_step]
        predecessor[multi_step] = states[selected, last[multi_step] - 1]
    return predecessor, successor, np.asarray(remaining, dtype=np.float64)


def _teacher_forced_costs(
    planner: LatentPlanner,
    problem,
    sequences: list[list[int | None]],
    predecessor: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    """Score a one-step prediction from each exact hypothetical predecessor."""

    if getattr(planner.model, "geo_rank_score_mode", "value") != "transition":
        raise ValueError("teacher-forced audit requires transition Energy")
    if hasattr(planner.model.predictor, "rollout"):
        raise ValueError(
            "teacher-forced audit currently supports the MLP predictor only"
        )
    final_actions = [
        int([action for action in sequence if action is not None][-1])
        for sequence in sequences
    ]
    codes = planner._action_codes(problem, final_actions)
    predicted = planner.model.predictor(predecessor, codes)
    return planner.model.core.transition_energy_head(
        predecessor, predicted, initial.expand(len(sequences), -1)
    )


def _latent_endpoints(
    planner: LatentPlanner,
    problem,
    sequences: list[list[int | None]],
    current: torch.Tensor,
) -> torch.Tensor:
    """Return recursively imagined endpoints for MLP transition checkpoints."""

    if hasattr(planner.model.predictor, "rollout"):
        raise ValueError("endpoint drift audit currently supports MLP predictor only")
    state = current.expand(len(sequences), -1).clone()
    depth = max(len(sequence) for sequence in sequences)
    for level in range(depth):
        rows = [
            row for row, sequence in enumerate(sequences)
            if level < len(sequence) and sequence[level] is not None
        ]
        if not rows:
            continue
        actions = [int(sequences[row][level]) for row in rows]
        index = torch.tensor(rows, device=planner.device)
        state[index] = planner.model.predictor(
            state[index], planner._action_codes(problem, actions)
        )
    return state


def _root_labels(env: SymbolicEnv) -> tuple[dict[int, int], dict[int, bool]]:
    exact, useful = {}, {}
    for root in env.feasible_actions():
        clone = env.clone()
        clone.step(root)
        exact[root] = 1 + clone.remaining_necessary()
        useful[root] = root in env.p.query_ancestors
    return exact, useful


def _score_factory(
    mode: str,
    planner: LatentPlanner,
    problem,
    env: SymbolicEnv,
    history: list[str],
    prompt_tokens: torch.Tensor,
    prompt_mask: torch.Tensor,
    current: torch.Tensor,
    initial: torch.Tensor,
    state_history: torch.Tensor,
    action_history: torch.Tensor,
):
    def score(sequences: list[list[int | None]]) -> np.ndarray:
        planner.lookahead = max(len(sequence) for sequence in sequences)
        if mode == "latent":
            return planner._flat_costs(
                current, initial, problem, sequences, None,
                state_history, action_history,
            ).detach().cpu().numpy()
        predecessor, successor, remaining = _exact_sequence_states(
            planner, env, history, prompt_tokens, prompt_mask,
            sequences, current,
        )
        if mode == "teacher_forced":
            return _teacher_forced_costs(
                planner, problem, sequences, predecessor, initial
            ).detach().cpu().numpy()
        if mode == "exact_transition":
            return planner.model.core.transition_energy_head(
                predecessor, successor, initial.expand(len(sequences), -1)
            ).detach().cpu().numpy()
        if mode == "oracle_endpoint":
            return remaining
        raise ValueError(f"unknown score mode: {mode}")

    return score


def _policy_cells() -> list[dict]:
    return [
        {"name": f"latent_global_d{depth}_b8", "mode": "latent",
         "depth": depth, "width": 8, "strategy": "global"}
        for depth in (1, 2, 3, 4)
    ] + [
        {"name": "latent_global_d4_b32", "mode": "latent",
         "depth": 4, "width": 32, "strategy": "global"},
        {"name": "latent_global_d4_b128", "mode": "latent",
         "depth": 4, "width": 128, "strategy": "global"},
        {"name": "latent_root_balanced_d4_b8", "mode": "latent",
         "depth": 4, "width": 8, "strategy": "root_balanced"},
        {"name": "teacher_forced_global_d4_b8", "mode": "teacher_forced",
         "depth": 4, "width": 8, "strategy": "global"},
        {"name": "exact_transition_global_d4_b8", "mode": "exact_transition",
         "depth": 4, "width": 8, "strategy": "global"},
        {"name": "oracle_endpoint_global_d4_b8", "mode": "oracle_endpoint",
         "depth": 4, "width": 8, "strategy": "global"},
    ]


@torch.no_grad()
def _evaluate_cell(model, vocab, dataset, device, args, cell: dict) -> dict:
    solved_steps = []
    distractors = decisions = 0
    survival: dict[int, list[float]] = defaultdict(list)
    useful_choices = []
    regrets = []
    for episode in range(args.episodes):
        problem, _ = dataset.problem(episode)
        env = SymbolicEnv(problem)
        prompt = prompt_sentences(problem, random.Random(args.seed + episode))
        planner = LatentPlanner(
            model, vocab, device, lookahead=max(1, cell["depth"]),
            max_expand=cell["width"], simulator="latent",
            allow_oracle_future_actions=cell["depth"] > 1,
            transition_energy_composition="terminal",
        )
        prompt_tokens = planner._tokens(prompt)
        prompt_mask = torch.ones(
            1, len(prompt), dtype=torch.bool, device=device
        )
        history: list[str] = []
        actions: list[int] = []
        budget = problem.n_necessary_steps + 2
        while not env.solved and len(actions) < budget:
            current = planner._current_state(
                prompt_tokens, prompt_mask, history
            )
            initial = planner._s0(prompt_tokens, prompt_mask)
            state_history, action_history = planner._causal_history(
                prompt_tokens, prompt_mask, history, problem, actions
            )
            score = _score_factory(
                cell["mode"], planner, problem, env, history,
                prompt_tokens, prompt_mask, current, initial,
                state_history, action_history,
            )
            result = search_symbolic_tree(
                problem, frozenset(env.resolved_set), cell["depth"],
                cell["width"], cell["strategy"], score,
            )
            chosen = int(result.sequence[0])
            root_exact, root_useful = _root_labels(env)
            useful_choices.append(float(root_useful[chosen]))
            regrets.append(float(root_exact[chosen] - min(root_exact.values())))
            distractors += int(not root_useful[chosen])
            decisions += 1
            for level in result.levels:
                survival[level.depth].append(float(level.optimal_root_survives))
            history.append(env.step(chosen))
            actions.append(chosen)
        solved_steps.append(len(actions) if env.solved else None)
    return {
        **cell,
        "episodes": args.episodes,
        "strict_success": float(np.mean([
            steps is not None and steps <= dataset.problem(index)[0].n_necessary_steps
            for index, steps in enumerate(solved_steps)
        ])),
        "slack2_success": float(np.mean([steps is not None for steps in solved_steps])),
        "mean_executed_steps": float(np.mean([
            steps if steps is not None
            else dataset.problem(index)[0].n_necessary_steps + 2
            for index, steps in enumerate(solved_steps)
        ])),
        "distractor_rate": distractors / max(decisions, 1),
        "useful_action_top1": float(np.mean(useful_choices)),
        "mean_action_regret": float(np.mean(regrets)),
        "optimal_root_survival": {
            str(depth): float(np.mean(values))
            for depth, values in sorted(survival.items())
        },
    }


@torch.no_grad()
def _diagnose_expert_histories(model, vocab, dataset, device, args) -> dict:
    rank_rows: dict[tuple[int, str], list[dict]] = defaultdict(list)
    drift_rows: dict[int, list[dict]] = defaultdict(list)
    correlation: dict[tuple[int, str], list[float]] = defaultdict(list)
    for episode in range(min(args.diagnostic_episodes, args.episodes)):
        problem, _ = dataset.problem(episode)
        env = SymbolicEnv(problem)
        prompt = prompt_sentences(problem, random.Random(args.seed + episode))
        planner = LatentPlanner(
            model, vocab, device, lookahead=1, simulator="latent",
            transition_energy_composition="terminal",
        )
        prompt_tokens = planner._tokens(prompt)
        prompt_mask = torch.ones(
            1, len(prompt), dtype=torch.bool, device=device
        )
        history: list[str] = []
        actions: list[int] = []
        while not env.solved:
            current = planner._current_state(prompt_tokens, prompt_mask, history)
            initial = planner._s0(prompt_tokens, prompt_mask)
            state_history, action_history = planner._causal_history(
                prompt_tokens, prompt_mask, history, problem, actions
            )
            root_exact, root_useful = _root_labels(env)
            one_sequences = [[root] for root in env.feasible_actions()]
            planner.lookahead = 1
            one_cost = planner._flat_costs(
                current, initial, problem, one_sequences, None,
                state_history, action_history,
            ).cpu().numpy()
            one_step = dict(zip(env.feasible_actions(), map(float, one_cost)))
            for depth in (1, 2, 3, 4):
                sequences = _sequences(
                    problem, frozenset(env.resolved_set), depth,
                    args.diagnostic_cap,
                    random.Random(f"{args.seed}:{episode}:{len(actions)}:{depth}"),
                )
                predecessor, exact_endpoint, remaining = _exact_sequence_states(
                    planner, env, history, prompt_tokens, prompt_mask,
                    sequences, current,
                )
                planner.lookahead = depth
                costs = {
                    "latent": planner._flat_costs(
                        current, initial, problem, sequences, None,
                        state_history, action_history,
                    ).cpu().numpy(),
                    "teacher_forced": _teacher_forced_costs(
                        planner, problem, sequences, predecessor, initial
                    ).cpu().numpy(),
                    "exact_transition": planner._symbolic_costs(
                        problem, env, history, current, initial,
                        prompt_tokens, prompt_mask, sequences, None,
                    ).cpu().numpy(),
                }
                predicted = _latent_endpoints(
                    planner, problem, sequences, current
                )
                groups = np.zeros(len(sequences), dtype=np.int64)
                drift_rows[depth].append(transition_metrics(
                    predicted.cpu().numpy(), exact_endpoint.cpu().numpy(), groups
                ))
                for mode, values in costs.items():
                    metrics, _ = summarize_depth_decision(
                        sequences, values, remaining, root_exact,
                        root_useful, one_step,
                    )
                    rank_rows[(depth, mode)].append(
                        metrics["deployed_terminal_score"]
                    )
                    observed = spearman_tied(values, remaining)
                    if observed is not None:
                        correlation[(depth, mode)].append(observed)
            useful = [
                root for root in env.feasible_actions()
                if root in problem.query_ancestors
            ]
            chosen = min(useful)
            history.append(env.step(chosen))
            actions.append(chosen)

    ranking = {}
    for depth in (1, 2, 3, 4):
        ranking[str(depth)] = {}
        for mode in ("latent", "teacher_forced", "exact_transition"):
            ranking[str(depth)][mode] = {
                **aggregate_rank_metrics(rank_rows[(depth, mode)]),
                "sequence_score_endpoint_spearman": (
                    float(np.mean(correlation[(depth, mode)]))
                    if correlation[(depth, mode)] else None
                ),
            }
    drift = {}
    for depth, rows in drift_rows.items():
        keys = sorted({key for row in rows for key in row if key != "candidates"})
        drift[str(depth)] = {
            key: float(np.mean([
                row[key] for row in rows
                if key in row and row[key] is not None and np.isfinite(row[key])
            ]))
            for key in keys
        }
        drift[str(depth)]["decisions"] = len(rows)
    return {"ranking": ranking, "rollout_drift": drift}


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    model, vocab, cfg = load_run(args.checkpoint, args.device)
    if cfg.data.get("name", "igsm") != "igsm":
        raise ValueError("depth localization currently requires stylized iGSM")
    if getattr(model, "geo_rank_score_mode", "value") != "transition":
        raise ValueError("depth localization requires transition Energy")
    if hasattr(model.predictor, "rollout"):
        raise ValueError("this round is predeclared for MLP predictors")
    dataset = build_dataset(cfg, vocab, split=args.split, size=args.episodes)
    device = torch.device(args.device)
    cells = [
        _evaluate_cell(model, vocab, dataset, device, args, cell)
        for cell in _policy_cells()
    ]
    diagnostics = _diagnose_expert_histories(
        model, vocab, dataset, device, args
    )
    return {
        "protocol": "intent-depth-localization-v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "episodes": args.episodes,
        "diagnostic_episodes": min(args.diagnostic_episodes, args.episodes),
        "seed": args.seed,
        "information_boundary": (
            "All cells receive the same symbolic feasible-action tree. "
            "Depth above one is candidate-privileged. Teacher-forced, exact-"
            "transition, and oracle-endpoint cells are labeled diagnostics."
        ),
        "policy_cells": cells,
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--diagnostic-episodes", type=int, default=100)
    parser.add_argument("--diagnostic-cap", type=int, default=128)
    parser.add_argument("--seed", type=int, default=8051)
    args = parser.parse_args()
    result = run(args)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        cell["name"]: {
            "strict": cell["strict_success"],
            "slack2": cell["slack2_success"],
            "useful_top1": cell["useful_action_top1"],
        }
        for cell in result["policy_cells"]
    }, indent=2))


if __name__ == "__main__":
    main()
