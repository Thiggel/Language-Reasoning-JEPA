"""Audit whether an aligned high-level ensemble detects macro OOD inputs."""

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


def predict(models: list, state: torch.Tensor, codes: torch.Tensor):
    return torch.stack([
        model.core.hi_predictor(state.expand(len(codes), -1), codes)
        for model in models
    ])


def disagreement(predictions: torch.Tensor) -> torch.Tensor:
    return predictions.var(0, unbiased=False).mean(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--ensemble", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--anchors", type=int, default=200)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument(
        "--scales", nargs="+", type=float,
        default=[0.1, 0.25, 0.5, 1.0, 2.0, 3.0],
    )
    args = parser.parse_args()
    seed_everything(881)
    model, vocab, cfg = load_run(args.ckpt, args.device)
    models = [load_run(path, args.device)[0] for path in args.ensemble]
    dataset = build_dataset(cfg, vocab, split="val", size=args.anchors * 3)
    planner = HierarchicalLatentPlanner(
        model, vocab, torch.device(args.device), method="shooting"
    )
    valid_uncertainty: list[torch.Tensor] = []
    valid_error: list[torch.Tensor] = []
    scale_stats = {
        scale: {"uncertainty": [], "support": [], "bank_distance": []}
        for scale in args.scales
    }
    anchors = 0
    K = model.core.macro_k
    with torch.no_grad():
        for index in range(len(dataset)):
            if anchors >= args.anchors:
                break
            problem, _ = dataset.problem(index)
            env = SymbolicEnv(problem)
            prompt = prompt_sentences(problem, random.Random(881 + index))
            prompt_tokens = planner._tokens(prompt)
            prompt_mask = torch.ones(
                1, len(prompt), dtype=torch.bool, device=planner.device
            )
            step_texts: list[str] = []
            while not env.solved and anchors < args.anchors:
                seqs = _sequences(
                    problem, frozenset(env.resolved_set), K,
                    args.max_candidates,
                )
                seqs = [seq for seq in seqs if len(seq) == K]
                if len(seqs) >= 2:
                    state = planner._current_state(
                        prompt_tokens, prompt_mask, step_texts
                    )
                    actions = planner._action_codes(
                        problem, [action for seq in seqs for action in seq]
                    ).reshape(len(seqs), K, -1)
                    codes = model.core.macro_encoder(actions)
                    target = planner._true_outcome_states(
                        env, seqs, step_texts, prompt_tokens, prompt_mask
                    )
                    predictions = predict(models, state, codes)
                    unc = disagreement(predictions)
                    error = planner._ln_l1(predictions.mean(0), target)
                    valid_uncertainty.append(unc.cpu())
                    valid_error.append(error.cpu())
                    code_scale = codes.std(0, unbiased=False).clamp_min(0.1)
                    for factor, stats in scale_stats.items():
                        perturbed = codes + (
                            factor * torch.randn_like(codes) * code_scale
                        )
                        perturbed_predictions = predict(
                            models, state, perturbed
                        )
                        stats["uncertainty"].append(
                            disagreement(perturbed_predictions).cpu()
                        )
                        stats["support"].append(
                            model.core.macro_support_head(
                                state.expand(len(codes), -1), perturbed
                            ).cpu()
                        )
                        stats["bank_distance"].append(
                            torch.cdist(perturbed, codes).square()
                            .min(-1).values.div(codes.shape[-1]).cpu()
                        )
                    anchors += 1
                necessary = [
                    action for action in env.feasible_actions()
                    if action in problem.query_ancestors
                ]
                step_texts.append(env.step(min(necessary)))
    valid_unc = torch.cat(valid_uncertainty)
    error = torch.cat(valid_error)
    result = {
        "checkpoint": args.ckpt,
        "ensemble": args.ensemble,
        "anchors": anchors,
        "valid_candidates": int(valid_unc.numel()),
        "metrics": {
            "valid_uncertainty": float(valid_unc.mean()),
            "valid_prediction_error": float(error.mean()),
            "uncertainty_error_correlation": correlation(valid_unc, error),
            "scale_curve": {
                str(factor): {
                    "uncertainty": float(
                        torch.cat(stats["uncertainty"]).mean()
                    ),
                    "uncertainty_pair_accuracy": float((
                        torch.cat(stats["uncertainty"]) > valid_unc
                    ).float().mean()),
                    "support_logit": float(
                        torch.cat(stats["support"]).mean()
                    ),
                    "bank_distance": float(
                        torch.cat(stats["bank_distance"]).mean()
                    ),
                }
                for factor, stats in scale_stats.items()
            },
        },
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
