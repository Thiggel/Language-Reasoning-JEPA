"""Diagnose whether generated macro codes and subgoals remain reachable."""

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


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: sum(row[name] for row in rows) / len(rows)
        for name in rows[0]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--method", choices=["shooting", "cem"], default="cem")
    parser.add_argument(
        "--energy", choices=["value", "macro_q", "oracle_goal"],
        default="oracle_goal",
    )
    parser.add_argument("--anchors", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--elites", type=int, default=10)
    parser.add_argument("--high-horizon", type=int, default=2)
    parser.add_argument("--max-expand", type=int, default=256)
    parser.add_argument("--density-weight", type=float, default=0.1)
    parser.add_argument("--learned-support-weight", type=float, default=0.0)
    parser.add_argument("--learned-support-threshold", type=float, default=None)
    args = parser.parse_args()
    seed_everything(321)
    model, vocab, cfg = load_run(args.ckpt, args.device)
    dataset = build_dataset(cfg, vocab, split="val", size=args.anchors * 2)
    planner = HierarchicalLatentPlanner(
        model,
        vocab,
        torch.device(args.device),
        method=args.method,
        energy=args.energy,
        high_horizon=args.high_horizon,
        n_samples=args.samples,
        cem_iters=args.iters,
        n_elites=args.elites,
        density_weight=args.density_weight,
        learned_support_weight=args.learned_support_weight,
        learned_support_threshold=args.learned_support_threshold,
    )
    rows: list[dict[str, float]] = []
    K = model.core.macro_k
    with torch.no_grad():
        for i in range(len(dataset)):
            if len(rows) >= args.anchors:
                break
            problem, _ = dataset.problem(i)
            env = SymbolicEnv(problem)
            prompt = prompt_sentences(problem, random.Random(321 + i))
            prompt_tokens = planner._tokens(prompt)
            prompt_mask = torch.ones(
                1, len(prompt), dtype=torch.bool, device=planner.device
            )
            step_texts: list[str] = []
            while not env.solved and len(rows) < args.anchors:
                seqs = _sequences(
                    problem,
                    frozenset(env.resolved_set),
                    K,
                    args.max_expand,
                )
                seqs = [seq for seq in seqs if len(seq) == K]
                if seqs:
                    state = planner._current_state(
                        prompt_tokens, prompt_mask, step_texts
                    )
                    s0 = planner._s0(prompt_tokens, prompt_mask)
                    goal = planner._oracle_goal_state(
                        problem, prompt_tokens, prompt_mask
                    ) if args.energy == "oracle_goal" else None
                    subgoal = planner._high_subgoal(state, s0, goal)
                    chosen_code = planner.last_high_plan["codes"][0]
                    true_states = planner._true_outcome_states(
                        env,
                        seqs,
                        step_texts,
                        prompt_tokens,
                        prompt_mask,
                    )
                    action = planner._action_codes(
                        problem, [a for seq in seqs for a in seq]
                    ).reshape(len(seqs), K, -1)
                    valid_codes = model.core.macro_encoder(action)
                    high_pred = model.core.hi_predictor(
                        state.expand(len(seqs), -1), valid_codes
                    )
                    cur = state.expand(len(seqs), -1).clone()
                    for depth in range(K):
                        cur = model.predictor(cur, action[:, depth])
                    manual = planner._oracle_waypoint(
                        problem,
                        env,
                        prompt_tokens,
                        prompt_mask,
                        step_texts,
                    )
                    pm, pl = model.core.macro_encoder.prior_params(state)
                    code_nll = 0.5 * (
                        pl + (chosen_code - pm).square() * (-pl).exp()
                    ).sum(-1)
                    d = planner._ln_l1
                    rows.append({
                        "generated_to_true_state": float(
                            d(true_states, subgoal.expand_as(true_states)).min()
                        ),
                        "generated_to_valid_prediction": float(
                            d(high_pred, subgoal.expand_as(high_pred)).min()
                        ),
                        "generated_to_low_rollout": float(
                            d(cur, subgoal.expand_as(cur)).min()
                        ),
                        "manual_to_low_rollout": float(
                            d(cur, manual.expand_as(cur)).min()
                        ),
                        "valid_high_prediction_error": float(
                            d(high_pred, true_states).mean()
                        ),
                        "valid_low_rollout_error": float(
                            d(cur, true_states).mean()
                        ),
                        "code_to_valid_macro_l2": float(
                            (valid_codes - chosen_code).square().mean(-1).sqrt().min()
                        ),
                        "chosen_code_nll": float(code_nll),
                        "chosen_learned_support": float(
                            planner.last_high_plan["learned_support"]
                        ),
                        "chosen_macro_q": float(
                            planner.last_high_plan["macro_q"]
                        ),
                        "generated_goal_distance": float(
                            d(subgoal, goal) if goal is not None
                            else model.core.hi_value_head(subgoal, s0)
                        ),
                        "manual_goal_distance": float(
                            d(manual, goal) if goal is not None
                            else model.core.hi_value_head(manual, s0)
                        ),
                    })
                necessary = [
                    a for a in env.feasible_actions()
                    if a in problem.query_ancestors
                ]
                step_texts.append(env.step(min(necessary)))
    result = {
        "checkpoint": args.ckpt,
        "method": args.method,
        "energy": args.energy,
        "anchors": len(rows),
        "metrics": summarize(rows),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
