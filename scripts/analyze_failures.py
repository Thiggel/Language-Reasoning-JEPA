"""Failure analysis: what do the planner's residual failures share?

Runs planning episodes, records per-episode problem features and outcome,
and reports failure rates grouped by each feature.

Usage: python scripts/analyze_failures.py ckpt=runs/X/best.pt device=cuda:0
Writes <run>/failures.json.
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.graph import OPS
from textjepa.data.igsm.render import prompt_sentences
from textjepa.planning import LatentPlanner
from textjepa.planning.search import EpisodeResult
from textjepa.utils.checkpoint import build_dataset, load_run
from textjepa.utils.seed import seed_everything


@torch.no_grad()
def _trace_one_step_episode(
    planner, problem, slack: int, seed: int, faithful: bool = False
) -> tuple[EpisodeResult, list[dict]]:
    """Run the one-step planner while retaining every root-action score.

    A positive ``necessary_margin`` means that the lowest-energy necessary
    action outranks the lowest-energy distractor.  This is analysis only: the
    planner still selects the global energy minimum and never sees the labels.
    """
    if faithful:
        from textjepa.data.faithful import FaithfulEnv

        env = FaithfulEnv(problem)
        prompt = problem.prompt_sentences
        necessary = problem.necessary
        total_necessary = len(necessary)
    else:
        env = SymbolicEnv(problem)
        prompt = prompt_sentences(problem, random.Random(seed))
        necessary = problem.query_ancestors
        total_necessary = problem.n_necessary_steps
    prompt_tokens = planner._tokens(prompt)
    prompt_mask = torch.ones(
        1, len(prompt), dtype=torch.bool, device=planner.device
    )
    step_texts: list[str] = []
    budget = total_necessary + slack
    n_distractor = 0
    decisions = []
    goal_state = (
        planner._oracle_goal_state(problem, prompt_tokens, prompt_mask)
        if not faithful and planner.energy == "oracle_goal"
        else None
    )

    while not env.solved and len(step_texts) < budget:
        if faithful:
            s = planner._state(prompt_tokens, prompt_mask, step_texts)
            s0 = planner._state(prompt_tokens, prompt_mask, [])
        else:
            s = planner._current_state(prompt_tokens, prompt_mask, step_texts)
            s0 = planner._s0(prompt_tokens, prompt_mask)
        feasible = env.feasible_actions()
        if faithful:
            action_tokens = planner._tokens(
                [env.action_text(action) for action in feasible]
            ).squeeze(0).unsqueeze(1)
            action_codes = planner.model.encode_actions(action_tokens).squeeze(1)
            predictions = planner.model.predictor(
                s.expand(len(feasible), -1), action_codes
            )
            costs = planner.model.value_head(
                predictions, s0.expand(len(feasible), -1)
            ).detach().cpu()
        else:
            costs = planner._flat_costs(
                s, s0, problem, [[a] for a in feasible], goal_state
            ).detach().cpu()
        chosen_i = int(costs.argmin().item())
        chosen = feasible[chosen_i]
        necessary_i = [i for i, a in enumerate(feasible) if a in necessary]
        distractor_i = [i for i, a in enumerate(feasible) if a not in necessary]
        best_necessary = min(float(costs[i]) for i in necessary_i)
        best_distractor = (
            min(float(costs[i]) for i in distractor_i)
            if distractor_i else None
        )
        margin = (
            best_distractor - best_necessary
            if best_distractor is not None else None
        )
        decisions.append({
            "step_index": len(step_texts),
            "history_clean": n_distractor == 0,
            "total_necessary": total_necessary,
            "remaining_necessary": len(necessary - env.resolved_set),
            "n_feasible": len(feasible),
            "n_distractor_feasible": len(distractor_i),
            "chosen_necessary": chosen in necessary,
            "chosen_energy": float(costs[chosen_i]),
            "best_necessary_energy": best_necessary,
            "best_distractor_energy": best_distractor,
            "necessary_margin": margin,
        })
        n_distractor += int(chosen not in necessary)
        step_texts.append(env.step(chosen))

    return EpisodeResult(
        env.solved, len(step_texts), total_necessary, n_distractor
    ), decisions


def _decision_summary(
    decisions: list[dict],
    total_buckets=((3, 4), (5, 6), (7, 9)),
    remaining_buckets=((1, 2), (3, 4), (5, 9)),
    step_buckets=((0, 1), (2, 3), (4, 8)),
) -> dict:
    competitive = [d for d in decisions if d["n_distractor_feasible"] > 0]
    clean = [d for d in competitive if d["history_clean"]]
    off_history = [d for d in competitive if not d["history_clean"]]
    margins = [d["necessary_margin"] for d in competitive]
    errors = [d for d in competitive if not d["chosen_necessary"]]
    correct_margins = [
        d["necessary_margin"] for d in competitive if d["chosen_necessary"]
    ]
    error_margins = [d["necessary_margin"] for d in errors]

    def accuracy(rows):
        return round(
            sum(d["chosen_necessary"] for d in rows) / max(len(rows), 1), 4
        )

    def groups(key, buckets):
        out = {}
        for lo, hi in buckets:
            rows = [d for d in decisions if lo <= d[key] <= hi]
            comp = [d for d in rows if d["n_distractor_feasible"] > 0]
            if rows:
                out[f"{key} {lo}-{hi}"] = {
                    "top1_necessary": accuracy(rows),
                    "competitive_top1": accuracy(comp),
                    "n": len(rows),
                    "competitive_n": len(comp),
                }
        return out

    return {
        "top1_necessary": accuracy(decisions),
        "top1_necessary_n": len(decisions),
        "competitive_top1": accuracy(competitive),
        "competitive_n": len(competitive),
        "clean_history_top1": accuracy(clean),
        "clean_history_n": len(clean),
        "off_history_top1": accuracy(off_history),
        "off_history_n": len(off_history),
        "n_errors": len(errors),
        "necessary_margin_mean": round(sum(margins) / max(len(margins), 1), 4),
        "necessary_margin_median": round(statistics.median(margins), 4)
        if margins else 0.0,
        "correct_margin_median": round(statistics.median(correct_margins), 4)
        if correct_margins else 0.0,
        "error_margin_median": round(statistics.median(error_margins), 4)
        if error_margins else 0.0,
        "near_tie_fraction": {
            str(threshold): round(
                sum(abs(m) < threshold for m in margins) / max(len(margins), 1),
                4,
            )
            for threshold in (0.005, 0.01, 0.02, 0.05)
        },
        "by_total_necessary": groups(
            "total_necessary", total_buckets
        ),
        "by_remaining_necessary": groups(
            "remaining_necessary", remaining_buckets
        ),
        "by_step_index": groups("step_index", step_buckets),
    }


@hydra.main(config_path="../configs", config_name="plan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    model, vocab, run_cfg = load_run(cfg.ckpt, cfg.device)
    dataset = build_dataset(run_cfg, vocab, split="val")
    device = torch.device(cfg.device)
    faithful = run_cfg.data.get("name", "igsm") == "igsm_real"
    if faithful:
        if cfg.energy != "value":
            raise ValueError("faithful failure analysis supports value energy")
        from textjepa.planning.faithful_search import FaithfulPlanner

        planner = FaithfulPlanner(
            model, vocab, device, lookahead=cfg.lookahead,
            max_expand=cfg.max_expand,
            allow_oracle_future_actions=cfg.allow_oracle_future_actions,
        )
        necessary_buckets = [(1, 4), (5, 8), (9, 15)]
        distractor_buckets = [(0, 4), (5, 9), (10, 20)]
        remaining_buckets = necessary_buckets
        step_buckets = [(0, 3), (4, 7), (8, 14)]
    else:
        planner = LatentPlanner(
            model, vocab, device, lookahead=cfg.lookahead,
            max_expand=cfg.max_expand, energy=cfg.energy,
            hierarchy=cfg.get("hierarchy", False),
            simulator=cfg.get("simulator", "latent"),
            allow_oracle_future_actions=cfg.allow_oracle_future_actions,
        )
        necessary_buckets = [(3, 4), (5, 6), (7, 9)]
        distractor_buckets = [(0, 2), (3, 5), (6, 9)]
        remaining_buckets = [(1, 2), (3, 4), (5, 9)]
        step_buckets = [(0, 1), (2, 3), (4, 8)]
    episodes = []
    decisions = []
    for i in range(cfg.n_episodes):
        p, _ = dataset.problem(i)
        if (
            cfg.lookahead == 1
            and not cfg.get("hierarchy", False)
            and cfg.get("simulator", "latent") == "latent"
        ):
            r, episode_decisions = _trace_one_step_episode(
                planner, p, cfg.slack, cfg.seed + i, faithful=faithful
            )
            for row in episode_decisions:
                row["episode"] = i
            decisions.extend(episode_decisions)
        else:
            r = planner.plan_episode(p, slack=cfg.slack, seed=cfg.seed + i)
        nec = p.necessary if faithful else p.query_ancestors
        mul_frac = (
            sum(p.op_label(j) == 3 for j in nec)
            if faithful else sum(p.vars[j].op == "mul" for j in nec)
        ) / max(len(nec), 1)
        n_vars = len(p.params) if faithful else len(p.vars)
        episodes.append({
            "solved": bool(r.solved),
            "n_necessary": len(nec) if faithful else p.n_necessary_steps,
            "n_vars": n_vars,
            "n_distractor_vars": n_vars - len(nec),
            "mul_frac": round(mul_frac, 2),
            "distractor_picks": r.n_distractor,
        })

    def group(key, buckets):
        rows = {}
        for lo, hi in buckets:
            sel = [e for e in episodes if lo <= e[key] <= hi]
            if sel:
                rows[f"{key} {lo}-{hi}"] = {
                    "fail_rate": round(
                        sum(not e["solved"] for e in sel) / len(sel), 3
                    ),
                    "n": len(sel),
                }
        return rows

    report = {
        "domain": "official iGSM" if faithful else "stylized iGSM",
        "overall_success": round(
            sum(e["solved"] for e in episodes) / len(episodes), 4
        ),
        "by_n_necessary": group("n_necessary", necessary_buckets),
        "by_n_distractor_vars": group(
            "n_distractor_vars", distractor_buckets
        ),
        "by_mul_frac": group("mul_frac", [(0.0, 0.0), (0.01, 0.4), (0.41, 1.0)]),
        "failures_with_distractor_pick": round(
            sum(
                1 for e in episodes if not e["solved"] and e["distractor_picks"] > 0
            ) / max(sum(not e["solved"] for e in episodes), 1),
            3,
        ),
    }
    if decisions:
        report["one_step_decisions"] = _decision_summary(
            decisions, total_buckets=necessary_buckets,
            remaining_buckets=remaining_buckets,
            step_buckets=step_buckets,
        )
    print(json.dumps(report, indent=2))
    out = Path(cfg.out or Path(cfg.ckpt).parent / "failures.json")
    out.write_text(json.dumps({
        "report": report, "episodes": episodes, "decisions": decisions
    }, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
