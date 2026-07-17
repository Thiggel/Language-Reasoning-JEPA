"""Decompose token planning into proposal, dynamics, geometry, and value errors.

The audit is deliberately non-symbolic.  Candidate outcomes are obtained by
appending every vocabulary token to the same text prefix and EMA-encoding the
result.  It compares teacher states with factual-action open-loop predictor
states, so the reference next token remains well defined in both conditions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.lm import LMDataset, collate_lm
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA


def normalized_l1(rows: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    return (
        F.layer_norm(rows, rows.shape[-1:])
        - F.layer_norm(goal, goal.shape[-1:])
    ).abs().mean(-1)


def one_indexed_rank(cost: torch.Tensor, factual_index: int) -> int:
    return int((cost < cost[factual_index]).sum()) + 1


def pairwise_accuracy(score: torch.Tensor, target: torch.Tensor) -> float:
    delta_score = score[:, None] - score[None, :]
    delta_target = target[:, None] - target[None, :]
    valid = torch.triu(torch.ones_like(delta_score, dtype=torch.bool), 1)
    valid &= delta_target.abs() > 1e-8
    if not valid.any():
        return float("nan")
    return float((delta_score[valid].sign() == delta_target[valid].sign()).float().mean())


def summarize(values: list[float]) -> dict[str, float]:
    finite = torch.tensor(values, dtype=torch.float)
    finite = finite[torch.isfinite(finite)]
    if not len(finite):
        return {"mean": float("nan"), "p90": float("nan"), "n": 0}
    return {
        "mean": float(finite.mean()),
        "p90": float(torch.quantile(finite, 0.9)),
        "n": int(len(finite)),
    }


def summarize_ranks(values: list[float]) -> dict[str, float]:
    ranks = torch.tensor(values, dtype=torch.float)
    return {
        "top1": float((ranks <= 1).float().mean()),
        "top5": float((ranks <= 5).float().mean()),
        "top20": float((ranks <= 20).float().mean()),
        "mean_rank": float(ranks.mean()),
        "n": int(len(ranks)),
    }


def pack_teacher_outcomes(model, prefix: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    rows = torch.cat([
        prefix.unsqueeze(0).expand(len(candidates), -1),
        candidates[:, None],
    ], dim=1)
    return model.teacher(rows)[:, -1]


def candidate_predictions(model, current, history, action_history, candidates):
    count = len(candidates)
    actions = model.token_action(candidates)[:, None]
    return model.low_predictor.rollout(
        current.expand(count, -1), actions,
        state_history=history.expand(count, -1, -1),
        action_history=action_history.expand(count, -1, -1),
    )[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--positions", type=int, default=128)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--goal-horizons", type=int, nargs="+", default=[1, 4, 16, 0])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    if model.token_prior is None:
        raise ValueError("planner-interface audit requires a trained token prior")

    dataset = LMDataset(
        vocab, size=args.examples, seed=cfg.data.val_seed + 130363,
        modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range), leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    loader = DataLoader(
        dataset, batch_size=1,
        collate_fn=lambda rows: collate_lm(rows, pad_id=vocab.pad_id),
    )
    candidates = torch.tensor(
        [i for i in range(len(vocab)) if i != vocab.pad_id],
        device=args.device,
    )
    collected: dict[str, list[float]] = defaultdict(list)
    rank_metrics: set[str] = set()

    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(args.device)
            prompt_len = batch["prompt_len"].to(args.device)
            out = model(tokens, prompt_len)
            length = int(out["valid"][0].sum())
            if length == 0:
                continue

            rolled_states = [out["prompt_state"]]
            for position in range(length):
                history = torch.stack(rolled_states, dim=1)
                action_history = out["token_actions"][:, :position]
                next_state = model.low_predictor.rollout(
                    rolled_states[-1], out["token_actions"][:, position:position + 1],
                    state_history=history, action_history=action_history,
                )[:, 0]
                rolled_states.append(next_state)

            chosen_positions = torch.linspace(
                0, length - 1, min(length, 4)
            ).long().unique().tolist()
            for position in chosen_positions:
                true_id = int(out["action_ids"][0, position])
                true_index = int((candidates == true_id).nonzero()[0])
                prefix_end = int(prompt_len[0]) + position
                actual = pack_teacher_outcomes(
                    model, tokens[0, :prefix_end], candidates
                )
                factual_actual = actual[true_index]

                teacher_history = out["prev"][:, :position + 1]
                rolled_history = torch.stack(rolled_states[:position + 1], dim=1)
                action_history = out["token_actions"][:, :position]
                conditions = {
                    "teacher": (out["prev"][:, position], teacher_history),
                    "rolled": (rolled_states[position], rolled_history),
                }
                for condition, (current, history) in conditions.items():
                    prior_cost = -model.token_prior(current)[0, candidates]
                    support = prior_cost.topk(
                        min(args.topk, len(candidates)), largest=False
                    ).indices
                    collected[f"{condition}/prior_reference_rank"].append(
                        one_indexed_rank(prior_cost, true_index)
                    )
                    rank_metrics.add(f"{condition}/prior_reference_rank")
                    predictions = candidate_predictions(
                        model, current, history, action_history, candidates
                    )
                    dynamics = normalized_l1(predictions, actual)
                    collected[f"{condition}/dynamics_all"].extend(
                        dynamics.tolist()
                    )
                    collected[f"{condition}/dynamics_factual"].append(
                        float(dynamics[true_index])
                    )

                    for horizon in args.goal_horizons:
                        target_index = length - 1 if horizon == 0 else min(
                            position + horizon - 1, length - 1
                        )
                        goal = out["target"][:, target_index]
                        actual_cost = normalized_l1(actual, goal.expand_as(actual))
                        predicted_cost = normalized_l1(
                            predictions, goal.expand_as(predictions)
                        )
                        learned_cost = model.low_goal_value(
                            predictions, goal.expand_as(predictions)
                        )
                        label = "terminal" if horizon == 0 else f"h{horizon}"
                        for name, score in {
                            "oracle_geometry": actual_cost,
                            "predicted_geometry": predicted_cost,
                            "learned_value": learned_cost,
                        }.items():
                            metric = f"{condition}/{label}/{name}"
                            collected[f"{metric}_reference_rank"].append(
                                one_indexed_rank(score, true_index)
                            )
                            rank_metrics.add(f"{metric}_reference_rank")
                            selected = support[score[support].argmin()]
                            best_supported = actual_cost[support].min()
                            collected[f"{metric}_support_regret"].append(
                                float(actual_cost[selected] - best_supported)
                            )
                            collected[f"{metric}_support_pair_accuracy"].append(
                                pairwise_accuracy(score[support], actual_cost[support])
                            )
                            collected[f"{metric}_selected_reference"].append(
                                float(int(selected) == true_index)
                            )
                        baseline = normalized_l1(
                            out["prompt_target"] if position == 0 else out["target"][:, position - 1],
                            goal,
                        )[0]
                        collected[f"{condition}/{label}/factual_advantage"].append(
                            float(actual_cost[true_index] - baseline)
                        )
                if len(collected["teacher/prior_reference_rank"]) >= args.positions:
                    break
            if len(collected["teacher/prior_reference_rank"]) >= args.positions:
                break

    result = {
        key: summarize_ranks(values) if key in rank_metrics else summarize(values)
        for key, values in sorted(collected.items())
    }
    result["metadata"] = {
        "vocabulary_candidates": len(candidates), "topk": args.topk,
        "uses_symbolic_feasibility": False, "uses_auxiliary_lm": False,
        "rolled_condition": "recursive predictor states under factual actions",
    }
    destination = Path(args.out) if args.out else Path(args.ckpt).parent / "planner_interface_audit.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
