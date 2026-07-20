"""Evaluate budgeted accuracy across exact necessary-step lengths.

Success without an action budget is trivial in stylized iGSM because repeatedly
executing feasible actions eventually resolves every variable.  We therefore
report success at fixed excess-action budgets, plus restricted excess cost.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from textjepa.planning import LatentPlanner, evaluate_planning
from textjepa.planning.search import validate_learned_catalogue_checkpoint
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, load_run


def variable_range(
    length: int, override: tuple[int, int] | None = None
) -> tuple[int, int]:
    if override is not None:
        return override
    # Longer ancestral closures are rare under the ordinary DAG generator.
    # More total variables preserve the generator while making rejection
    # sampling the requested exact length practical.
    return (6, 12) if length <= 9 else (2 * length, 3 * length)


class FixedProblemDataset:
    """Materialize exact-length problems once for every slack evaluation."""

    def __init__(self, dataset, size: int):
        self.problems = [dataset.problem(i)[0] for i in range(size)]

    def problem(self, index: int):
        return self.problems[index], None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lengths", type=int, nargs="+", default=[3, 6, 9, 12, 15, 18])
    parser.add_argument(
        "--n-vars-range", type=int, nargs=2, metavar=("MIN", "MAX"),
        help="hold total prompt variables fixed instead of using the legacy length-dependent range",
    )
    parser.add_argument("--slacks", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7321)
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--prior-top-m", type=int, default=0)
    parser.add_argument("--prior-only", action="store_true")
    parser.add_argument("--allow-oracle-future-actions", action="store_true")
    parser.add_argument(
        "--proposal-source",
        choices=["current_feasible", "learned_catalogue"],
        default="current_feasible",
    )
    parser.add_argument("--proposal-top-m", type=int, default=0)
    parser.add_argument("--proposal-beam-width", type=int, default=1)
    parser.add_argument("--proposal-prior-weight", type=float, default=1.0)
    parser.add_argument("--proposal-support-weight", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    n_vars_override = tuple(args.n_vars_range) if args.n_vars_range else None

    if (
        args.lookahead > 1
        and args.proposal_source == "current_feasible"
        and not args.allow_oracle_future_actions
    ):
        parser.error("lookahead > 1 is an oracle-future-action diagnostic")
    seed_everything(args.seed)
    model, vocab, checkpoint_cfg = load_run(args.ckpt, args.device)
    if args.proposal_source == "learned_catalogue":
        validate_learned_catalogue_checkpoint(checkpoint_cfg)
    payload = {
        "checkpoint": args.ckpt,
        "length_definition": "exact number of necessary actions",
        "train_steps_range": list(checkpoint_cfg.data.steps_range),
        "slacks": args.slacks,
        "lookahead": args.lookahead,
        "oracle_future_actions": bool(args.allow_oracle_future_actions),
        "prior_top_m": args.prior_top_m,
        "prior_only": args.prior_only,
        "proposal_source": args.proposal_source,
        "proposal_top_m": args.proposal_top_m,
        "proposal_beam_width": args.proposal_beam_width,
        "proposal_prior_weight": args.proposal_prior_weight,
        "proposal_support_weight": args.proposal_support_weight,
        "n_vars_range_override": list(n_vars_override) if n_vars_override else None,
        "curves": {},
    }
    for length in args.lengths:
        cfg = OmegaConf.create(OmegaConf.to_container(checkpoint_cfg, resolve=True))
        cfg.data.steps_range = [length, length]
        cfg.data.n_vars_range = list(variable_range(length, n_vars_override))
        cfg.data.test_seed = args.seed + 1000 * length
        cfg.data.test_size = args.episodes
        cfg.data.problem_max_tries = 10000
        cfg.data.strict_steps_range = True
        dataset = FixedProblemDataset(
            build_dataset(cfg, vocab, split="test", size=args.episodes),
            args.episodes,
        )
        length_rows = {}
        for slack in args.slacks:
            planner = LatentPlanner(
                model,
                vocab,
                torch.device(args.device),
                lookahead=args.lookahead,
                max_expand=256,
                energy="value",
                allow_oracle_future_actions=args.allow_oracle_future_actions,
                prior_top_m=args.prior_top_m,
                prior_only=args.prior_only,
                proposal_source=args.proposal_source,
                proposal_top_m=args.proposal_top_m,
                proposal_beam_width=args.proposal_beam_width,
                proposal_prior_weight=args.proposal_prior_weight,
                proposal_support_weight=args.proposal_support_weight,
            )
            length_rows[str(slack)] = evaluate_planning(
                planner, dataset, args.episodes, slack=slack, seed=args.seed
            )
        payload["curves"][str(length)] = {
            "n_vars_range": list(variable_range(length, n_vars_override)),
            "metrics_by_excess_budget": length_rows,
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
