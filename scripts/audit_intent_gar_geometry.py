"""Localize what GAR changes in intent-JEPA geometry and planning.

This audit follows the same canonical oracle state trace for every checkpoint,
but never supplies relevance labels to the model.  At each state it enumerates
the environment's feasible intent phrases and evaluates five lower-is-better
scores:

* ``gar_teacher_geometry``: true next state to terminal goal in EMA geometry;
* ``online_oracle_geometry``: true next state to goal in online geometry;
* ``predicted_geometry``: imagined next state to the EMA goal;
* ``oracle_transition_value``: learned value on the true next state;
* ``predicted_transition_value``: deployed value on the imagined next state.
* ``direct_action_score``: behavior-cloning control that scores ``(h,a)``
  without consuming an imagined successor (only for direct-ranker runs).

The successive gaps separate target geometry, online encoding, transition
drift, and value readout.  Optional forced-error branches test whether the
same geometry supports recovery away from the expert trajectory.  Symbolic
relevance is used only after scoring to calculate diagnostic metrics.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from textjepa.analysis.intent_decisions import (
    aggregate_rank_metrics,
    rank_metrics,
    transition_metrics,
)
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import prompt_sentences
from textjepa.planning.search import LatentPlanner
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, load_run


SCORES = (
    "gar_teacher_geometry",
    "online_oracle_geometry",
    "predicted_geometry",
    "oracle_transition_value",
    "predicted_transition_value",
    "direct_action_score",
)


def _ln(value: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(value, value.shape[-1:])


def _distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (_ln(left) - _ln(right)).abs().mean(-1)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left, right, dim=-1, eps=1e-8)


@torch.no_grad()
def _encode_last(
    planner: LatentPlanner,
    prompt_tokens: torch.Tensor,
    prompt_mask: torch.Tensor,
    histories: list[list[str]],
    *,
    teacher: bool,
) -> torch.Tensor:
    """Encode the final observed state for several text histories."""

    n = len(histories)
    if not n:
        raise ValueError("at least one history is required")
    if all(not history for history in histories):
        empty = torch.full(
            (n, 1, 1), planner.vocab.pad_id, dtype=torch.long,
            device=planner.device,
        )
        mask = torch.zeros(n, 1, dtype=torch.bool, device=planner.device)
        s0, _ = planner.model.encode_states(
            prompt_tokens.expand(n, -1, -1),
            prompt_mask.expand(n, -1), empty, mask, teacher=teacher,
        )
        return s0
    chunks = max(max(map(len, histories)), 1)
    width = max(
        (
            len(planner.vocab.encode(text))
            for history in histories for text in history
        ),
        default=1,
    )
    tokens = torch.full(
        (n, chunks, width), planner.vocab.pad_id, dtype=torch.long,
        device=planner.device,
    )
    mask = torch.zeros(n, chunks, dtype=torch.bool, device=planner.device)
    for row, history in enumerate(histories):
        for column, text in enumerate(history):
            ids = planner.vocab.encode(text)
            tokens[row, column, :len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=planner.device
            )
            mask[row, column] = True
    safe_mask = mask.clone()
    empty = ~safe_mask.any(1)
    safe_mask[empty, 0] = True
    _, states = planner.model.encode_states(
        prompt_tokens.expand(n, -1, -1),
        prompt_mask.expand(n, -1), tokens, safe_mask, teacher=teacher,
    )
    last = mask.sum(1).clamp(min=1) - 1
    encoded = states[torch.arange(n, device=planner.device), last]
    if empty.any():
        s0, _ = planner.model.encode_states(
            prompt_tokens.expand(int(empty.sum()), -1, -1),
            prompt_mask.expand(int(empty.sum()), -1),
            tokens[empty], mask[empty], teacher=teacher,
        )
        encoded[empty] = s0
    return encoded


@torch.no_grad()
def _predict_candidates(
    planner: LatentPlanner,
    problem,
    state: torch.Tensor,
    state_history: torch.Tensor,
    action_history: torch.Tensor,
    candidates: list[int],
) -> torch.Tensor:
    actions = planner._action_codes(problem, candidates).unsqueeze(1)
    n = len(candidates)
    if hasattr(planner.model.predictor, "rollout"):
        return planner.model.predictor.rollout(
            state.expand(n, -1), actions,
            state_history=state_history.expand(n, -1, -1),
            action_history=action_history.expand(n, -1, -1),
        )[:, -1]
    return planner.model.predictor(state.expand(n, -1), actions[:, 0])


def _complete_goal(env: SymbolicEnv, history: list[str]) -> list[str]:
    clone = env.clone()
    complete = list(history)
    while not clone.solved:
        necessary = [
            action for action in clone.feasible_actions()
            if action in clone.p.query_ancestors
        ]
        if not necessary:
            raise RuntimeError("no feasible necessary action before solution")
        complete.append(clone.step(min(necessary)))
    return complete


def _decision_rows(
    planner: LatentPlanner,
    problem,
    prompt_tokens: torch.Tensor,
    prompt_mask: torch.Tensor,
    env: SymbolicEnv,
    history: list[str],
    action_history_ids: list[int],
    *,
    episode: int,
    step: int,
    trace: str,
    group: int,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    candidates = env.feasible_actions()
    positive = np.asarray(
        [action in problem.query_ancestors for action in candidates], dtype=bool
    )
    if not positive.any():
        raise RuntimeError("competitive state has no feasible necessary action")
    current_online = _encode_last(
        planner, prompt_tokens, prompt_mask, [history], teacher=False
    )
    current_teacher = _encode_last(
        planner, prompt_tokens, prompt_mask, [history], teacher=True
    )
    goal_history = _complete_goal(env, history)
    goal_online = _encode_last(
        planner, prompt_tokens, prompt_mask, [goal_history], teacher=False
    )
    goal_teacher = _encode_last(
        planner, prompt_tokens, prompt_mask, [goal_history], teacher=True
    )
    candidate_histories = []
    exact_continuation_cost = []
    for action in candidates:
        clone = env.clone()
        candidate_histories.append(history + [clone.step(action)])
        # Exact shortest completion cost after committing to this feasible
        # action, including the action just taken.  In stylized iGSM this is
        # computable from the environment and is used only as an evaluation
        # label, never as model input.
        exact_continuation_cost.append(1 + clone.remaining_necessary())
    true_online = _encode_last(
        planner, prompt_tokens, prompt_mask, candidate_histories, teacher=False
    )
    true_teacher = _encode_last(
        planner, prompt_tokens, prompt_mask, candidate_histories, teacher=True
    )
    state_history, action_history = planner._causal_history(
        prompt_tokens, prompt_mask, history, problem, action_history_ids
    )
    predicted = _predict_candidates(
        planner, problem, current_online, state_history, action_history,
        candidates,
    )
    n = len(candidates)
    s0 = planner._s0(prompt_tokens, prompt_mask)
    action_codes = planner._action_codes(problem, candidates)
    costs = {
        "gar_teacher_geometry": _distance(
            true_teacher, goal_teacher.expand(n, -1)
        ),
        "online_oracle_geometry": _distance(
            true_online, goal_online.expand(n, -1)
        ),
        "predicted_geometry": _distance(
            predicted, goal_teacher.expand(n, -1)
        ),
        "oracle_transition_value": planner.model.value_head(
            true_online, s0.expand(n, -1)
        ),
        "predicted_transition_value": planner.model.value_head(
            predicted, s0.expand(n, -1)
        ),
    }
    if getattr(planner.model, "geo_rank_score_mode", "value") == "direct":
        costs["direct_action_score"] = planner.model.core.direct_action_rank_head(
            current_online.expand(n, -1), s0.expand(n, -1), action_codes
        )
    current_distance = float(_distance(current_teacher, goal_teacher).item())
    progress = current_distance - costs["gar_teacher_geometry"]
    true_delta = _ln(true_teacher) - _ln(current_teacher).expand(n, -1)
    predicted_delta = _ln(predicted) - _ln(current_online).expand(n, -1)
    goal_direction = _ln(goal_teacher) - _ln(current_teacher)
    transition_alignment = _cosine(predicted_delta, true_delta)
    goal_alignment = _cosine(true_delta, goal_direction.expand(n, -1))
    rows = []
    for index, action in enumerate(candidates):
        row = {
            "episode": episode,
            "step": step,
            "trace": trace,
            "group": group,
            "action": int(action),
            "operation": str(problem.vars[action].op),
            "necessary": bool(positive[index]),
            "remaining_necessary": int(env.remaining_necessary()),
            "exact_continuation_cost": int(exact_continuation_cost[index]),
            "n_candidates": n,
            "n_positive": int(positive.sum()),
            "current_goal_distance": current_distance,
            "teacher_geometry_progress": float(progress[index]),
            "transition_alignment": float(transition_alignment[index]),
            "goal_direction_alignment": float(goal_alignment[index]),
            **{name: float(value[index]) for name, value in costs.items()},
        }
        rows.append(row)
    features = {
        "current_online": current_online.expand(n, -1).cpu().numpy(),
        "current_teacher": current_teacher.expand(n, -1).cpu().numpy(),
        "goal_online": goal_online.expand(n, -1).cpu().numpy(),
        "goal_teacher": goal_teacher.expand(n, -1).cpu().numpy(),
        "true_next_online": true_online.cpu().numpy(),
        "true_next_teacher": true_teacher.cpu().numpy(),
        "predicted_next": predicted.cpu().numpy(),
        "necessary": positive,
        "group": np.full(n, group, dtype=np.int64),
        "episode": np.full(n, episode, dtype=np.int64),
        "step": np.full(n, step, dtype=np.int64),
        "action": np.asarray(candidates, dtype=np.int64),
        "forced": np.full(n, trace != "oracle", dtype=bool),
    }
    return rows, features


def _summarize(rows: list[dict], features: dict[str, np.ndarray]) -> dict:
    by_trace = {}
    for trace in sorted({row["trace"] for row in rows}):
        selected = [row for row in rows if row["trace"] == trace]
        groups = sorted({row["group"] for row in selected})
        score_summary = {}
        for score in (name for name in SCORES if name in selected[0]):
            metrics = []
            for group in groups:
                candidates = [row for row in selected if row["group"] == group]
                metrics.append(rank_metrics(
                    [row[score] for row in candidates],
                    [row["necessary"] for row in candidates],
                    [row["exact_continuation_cost"] for row in candidates],
                ))
            score_summary[score] = aggregate_rank_metrics(metrics)
        competitive = [
            row for row in selected
            if row["n_candidates"] > row["n_positive"]
        ]
        necessary = [row for row in competitive if row["necessary"]]
        distractor = [row for row in competitive if not row["necessary"]]

        def average(key: str, values: list[dict]) -> float | None:
            return (
                float(np.mean([row[key] for row in values])) if values else None
            )

        by_trace[trace] = {
            "groups": len(groups),
            "candidate_rows": len(selected),
            "scores": score_summary,
            "geometry": {
                "necessary_progress": average(
                    "teacher_geometry_progress", necessary
                ),
                "distractor_progress": average(
                    "teacher_geometry_progress", distractor
                ),
                "necessary_goal_direction_alignment": average(
                    "goal_direction_alignment", necessary
                ),
                "distractor_goal_direction_alignment": average(
                    "goal_direction_alignment", distractor
                ),
                "necessary_transition_alignment": average(
                    "transition_alignment", necessary
                ),
                "distractor_transition_alignment": average(
                    "transition_alignment", distractor
                ),
            },
        }
    by_trace_feature = features["forced"]
    transition = {}
    for name, forced in (("oracle", False), ("forced_error", True)):
        mask = by_trace_feature == forced
        if mask.any():
            transition[name] = transition_metrics(
                features["predicted_next"][mask],
                features["true_next_teacher"][mask],
                features["group"][mask],
            )
    return {"by_trace": by_trace, "transition": transition}


@torch.no_grad()
def run(args: argparse.Namespace) -> tuple[dict, dict[str, np.ndarray]]:
    seed_everything(args.seed)
    model, vocab, cfg = load_run(args.checkpoint, args.device)
    if cfg.data.get("name", "igsm") != "igsm":
        raise ValueError("GAR geometry audit currently requires stylized iGSM")
    dataset = build_dataset(cfg, vocab, split=args.split, size=args.episodes)
    planner = LatentPlanner(model, vocab, torch.device(args.device))
    all_rows: list[dict] = []
    feature_lists: dict[str, list[np.ndarray]] = defaultdict(list)
    group = 0
    for episode in range(args.episodes):
        problem, _ = dataset.problem(episode)
        env = SymbolicEnv(problem)
        prompt = prompt_sentences(problem, random.Random(args.seed + episode))
        prompt_tokens = planner._tokens(prompt)
        prompt_mask = torch.ones(
            1, len(prompt), dtype=torch.bool, device=planner.device
        )
        history: list[str] = []
        action_history: list[int] = []
        step = 0
        while not env.solved:
            rows, features = _decision_rows(
                planner, problem, prompt_tokens, prompt_mask, env, history,
                action_history, episode=episode, step=step, trace="oracle",
                group=group,
            )
            all_rows.extend(rows)
            for key, value in features.items():
                feature_lists[key].append(value)
            group += 1
            distractors = [
                action for action in env.feasible_actions()
                if action not in problem.query_ancestors
            ]
            if args.forced_errors and distractors:
                branch = env.clone()
                forced = min(distractors)
                branch_history = history + [branch.step(forced)]
                branch_actions = action_history + [forced]
                if not branch.solved:
                    branch_rows, branch_features = _decision_rows(
                        planner, problem, prompt_tokens, prompt_mask, branch,
                        branch_history, branch_actions, episode=episode,
                        step=step + 1, trace="forced_error", group=group,
                    )
                    all_rows.extend(branch_rows)
                    for key, value in branch_features.items():
                        feature_lists[key].append(value)
                    group += 1
            necessary = [
                action for action in env.feasible_actions()
                if action in problem.query_ancestors
            ]
            chosen = min(necessary)
            history.append(env.step(chosen))
            action_history.append(chosen)
            step += 1
    feature_arrays = {
        key: np.concatenate(values, axis=0) for key, values in feature_lists.items()
    }
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "episodes": args.episodes,
        "seed": args.seed,
        "forced_error_branches": bool(args.forced_errors),
        "information_boundary": (
            "Scores use feasible intent phrases and model state only. Symbolic "
            "query ancestry is used solely for post-hoc diagnostic labels. "
            "gar_teacher_geometry and oracle_transition_value encode true "
            "candidate outcomes and are explicitly oracle-transition diagnostics."
        ),
        "summary": _summarize(all_rows, feature_arrays),
        "rows": all_rows,
    }
    return result, feature_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--features-out")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7321)
    parser.add_argument("--forced-errors", action="store_true")
    args = parser.parse_args()
    result, features = run(args)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    feature_path = Path(args.features_out or destination.with_suffix(".npz"))
    np.savez_compressed(feature_path, **features)
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {destination} and {feature_path}")


if __name__ == "__main__":
    main()
