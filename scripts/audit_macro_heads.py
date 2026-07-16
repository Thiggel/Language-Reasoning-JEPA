"""Audit macro dynamics, exact value ordering, and action-manifold support."""

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


def correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.float(), y.float()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm() + 1e-8))


def pair_accuracy(score: torch.Tensor, target: torch.Tensor) -> tuple[int, int]:
    target_delta = target[:, None] - target[None, :]
    score_delta = score[:, None] - score[None, :]
    valid = target_delta != 0
    correct = (score_delta.sign() == target_delta.sign()) & valid
    return int(correct.sum()), int(valid.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--anchors", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument(
        "--support-scales",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0],
        help="Perturbation sizes, in empirical macro-code standard deviations.",
    )
    args = parser.parse_args()
    seed_everything(417)
    model, vocab, cfg = load_run(args.ckpt, args.device)
    dataset = build_dataset(cfg, vocab, split="val", size=args.anchors * 3)
    planner = HierarchicalLatentPlanner(
        model, vocab, torch.device(args.device), method="shooting"
    )
    values: dict[str, list[torch.Tensor]] = {
        name: [] for name in (
            "remaining", "macro_q", "state_value", "prior_nll",
            "geometry", "dynamics_l1", "support_pos", "support_neg",
            "q_perturbed",
        )
    }
    q_correct = q_pairs = v_correct = v_pairs = g_correct = g_pairs = 0
    top1 = {
        name: {"optimal": 0, "regret": 0.0}
        for name in ("macro_q", "state_value", "latent_goal")
    }
    support_scale_stats = {
        factor: {
            "correct": 0, "total": 0, "negative": [],
            "q_correct": 0, "q_negative": [], "q_margin": [],
        }
        for factor in args.support_scales
    }
    n_anchors = 0
    K = model.core.macro_k
    with torch.no_grad():
        for index in range(len(dataset)):
            if n_anchors >= args.anchors:
                break
            problem, _ = dataset.problem(index)
            env = SymbolicEnv(problem)
            prompt = prompt_sentences(problem, random.Random(417 + index))
            prompt_tokens = planner._tokens(prompt)
            prompt_mask = torch.ones(
                1, len(prompt), dtype=torch.bool, device=planner.device
            )
            step_texts: list[str] = []
            while not env.solved and n_anchors < args.anchors:
                seqs = _sequences(
                    problem, frozenset(env.resolved_set), K,
                    args.max_candidates,
                )
                seqs = [seq for seq in seqs if len(seq) == K]
                if len(seqs) >= 2:
                    state = planner._current_state(
                        prompt_tokens, prompt_mask, step_texts
                    )
                    s0 = planner._s0(prompt_tokens, prompt_mask)
                    goal = planner._oracle_goal_state(
                        problem, prompt_tokens, prompt_mask
                    )
                    action = planner._action_codes(
                        problem, [a for seq in seqs for a in seq]
                    ).reshape(len(seqs), K, -1)
                    codes = model.core.macro_encoder(action)
                    pred = model.core.hi_predictor(
                        state.expand(len(seqs), -1), codes
                    )
                    true = planner._true_outcome_states(
                        env, seqs, step_texts, prompt_tokens, prompt_mask
                    )
                    remaining = []
                    for seq in seqs:
                        clone = env.clone()
                        for action_id in seq:
                            clone.step(action_id)
                        remaining.append(clone.remaining_necessary())
                    remaining_t = torch.tensor(
                        remaining, dtype=pred.dtype, device=planner.device
                    )
                    q = model.core.macro_value_head(
                        state.expand(len(seqs), -1),
                        s0.expand(len(seqs), -1),
                        codes,
                    )
                    v = model.core.hi_value_head(
                        pred, s0.expand(len(seqs), -1)
                    )
                    pm, pl = model.core.macro_encoder.prior_params(
                        state.expand(len(seqs), -1)
                    )
                    nll = 0.5 * (
                        pl + (codes - pm).square() * (-pl).exp()
                    ).sum(-1)
                    pos = model.core.macro_support_head(
                        state.expand(len(seqs), -1), codes
                    )
                    scale = codes.std(0, unbiased=False).clamp_min(0.1)
                    negative_codes = codes + 3.0 * torch.randn_like(codes) * scale
                    neg = model.core.macro_support_head(
                        state.expand(len(seqs), -1), negative_codes
                    )
                    q_negative = model.core.macro_value_head(
                        state.expand(len(seqs), -1),
                        s0.expand(len(seqs), -1),
                        negative_codes,
                    )
                    geometry = planner._ln_l1(
                        true, goal.expand_as(true)
                    )
                    dynamics = planner._ln_l1(pred, true)
                    for name, tensor in (
                        ("remaining", remaining_t), ("macro_q", q),
                        ("state_value", v), ("prior_nll", nll),
                        ("geometry", geometry), ("dynamics_l1", dynamics),
                        ("support_pos", pos), ("support_neg", neg),
                        ("q_perturbed", q_negative),
                    ):
                        values[name].append(tensor.cpu())
                    c, n = pair_accuracy(q, remaining_t)
                    q_correct += c; q_pairs += n
                    c, n = pair_accuracy(v, remaining_t)
                    v_correct += c; v_pairs += n
                    c, n = pair_accuracy(geometry, remaining_t)
                    g_correct += c; g_pairs += n
                    best_remaining = float(remaining_t.min())
                    for name, score in (
                        ("macro_q", q),
                        ("state_value", v),
                        ("latent_goal", geometry),
                    ):
                        selected_remaining = float(
                            remaining_t[int(score.argmin())]
                        )
                        top1[name]["optimal"] += int(
                            selected_remaining == best_remaining
                        )
                        top1[name]["regret"] += (
                            selected_remaining - best_remaining
                        )
                    for factor, stats in support_scale_stats.items():
                        perturbed_codes = codes + (
                            factor * torch.randn_like(codes) * scale
                        )
                        negative = model.core.macro_support_head(
                            state.expand(len(seqs), -1), perturbed_codes
                        )
                        stats["correct"] += int((pos > negative).sum())
                        stats["total"] += int(pos.numel())
                        stats["negative"].append(negative.cpu())
                        perturbed_q = model.core.macro_value_head(
                            state.expand(len(seqs), -1),
                            s0.expand(len(seqs), -1),
                            perturbed_codes,
                        )
                        stats["q_correct"] += int((perturbed_q > q).sum())
                        stats["q_negative"].append(perturbed_q.cpu())
                        stats["q_margin"].append((perturbed_q - q).cpu())
                    n_anchors += 1
                necessary = [
                    action for action in env.feasible_actions()
                    if action in problem.query_ancestors
                ]
                step_texts.append(env.step(min(necessary)))
    cat = {name: torch.cat(parts) for name, parts in values.items()}
    remaining = cat["remaining"]
    result = {
        "checkpoint": args.ckpt,
        "anchors": n_anchors,
        "candidates": int(remaining.numel()),
        "metrics": {
            "macro_dynamics_l1": float(cat["dynamics_l1"].mean()),
            "macro_q_mae": float((cat["macro_q"] - remaining).abs().mean()),
            "macro_q_corr": correlation(cat["macro_q"], remaining),
            "macro_q_pair_accuracy": q_correct / max(q_pairs, 1),
            "macro_q_top1_optimal": top1["macro_q"]["optimal"] / n_anchors,
            "macro_q_top1_regret": top1["macro_q"]["regret"] / n_anchors,
            "state_value_mae": float(
                (cat["state_value"] - remaining).abs().mean()
            ),
            "state_value_corr": correlation(cat["state_value"], remaining),
            "state_value_pair_accuracy": v_correct / max(v_pairs, 1),
            "state_value_top1_optimal": (
                top1["state_value"]["optimal"] / n_anchors
            ),
            "state_value_top1_regret": (
                top1["state_value"]["regret"] / n_anchors
            ),
            "latent_goal_corr": correlation(cat["geometry"], remaining),
            "latent_goal_pair_accuracy": g_correct / max(g_pairs, 1),
            "latent_goal_top1_optimal": (
                top1["latent_goal"]["optimal"] / n_anchors
            ),
            "latent_goal_top1_regret": (
                top1["latent_goal"]["regret"] / n_anchors
            ),
            "prior_nll_corr": correlation(cat["prior_nll"], remaining),
            "support_positive_logit": float(cat["support_pos"].mean()),
            "support_perturbed_logit": float(cat["support_neg"].mean()),
            "support_pair_accuracy": float(
                (cat["support_pos"] > cat["support_neg"]).float().mean()
            ),
            "perturbed_q_margin": float(
                (cat["q_perturbed"] - cat["macro_q"]).mean()
            ),
            "perturbed_q_pair_accuracy": float(
                (cat["q_perturbed"] > cat["macro_q"]).float().mean()
            ),
            "support_scale_curve": {
                str(factor): {
                    "pair_accuracy": stats["correct"] / max(stats["total"], 1),
                    "perturbed_logit": float(torch.cat(stats["negative"]).mean()),
                    "perturbed_q": float(
                        torch.cat(stats["q_negative"]).mean()
                    ),
                    "q_margin": float(torch.cat(stats["q_margin"]).mean()),
                    "q_pair_accuracy": stats["q_correct"] / max(
                        stats["total"], 1
                    ),
                }
                for factor, stats in support_scale_stats.items()
            },
        },
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
