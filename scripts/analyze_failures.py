"""Failure analysis: what do the planner's residual failures share?

Runs planning episodes, records per-episode problem features and outcome,
and reports failure rates grouped by each feature.

Usage: python scripts/analyze_failures.py ckpt=runs/X/best.pt device=cuda:0
Writes <run>/failures.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from textjepa.data.igsm.graph import OPS
from textjepa.planning import LatentPlanner
from textjepa.utils.checkpoint import build_dataset, load_run
from textjepa.utils.seed import seed_everything


@hydra.main(config_path="../configs", config_name="plan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    model, vocab, run_cfg = load_run(cfg.ckpt, cfg.device)
    dataset = build_dataset(run_cfg, vocab, split="val")
    device = torch.device(cfg.device)
    planner = LatentPlanner(
        model, vocab, device, lookahead=cfg.lookahead,
        max_expand=cfg.max_expand, energy=cfg.energy,
    )
    episodes = []
    for i in range(cfg.n_episodes):
        p, _ = dataset.problem(i)
        r = planner.plan_episode(p, slack=cfg.slack, seed=cfg.seed + i)
        nec = p.query_ancestors
        mul_frac = sum(
            1 for j in nec if p.vars[j].op == "mul"
        ) / max(len(nec), 1)
        episodes.append({
            "solved": bool(r.solved),
            "n_necessary": p.n_necessary_steps,
            "n_vars": len(p.vars),
            "n_distractor_vars": len(p.vars) - len(nec),
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
        "overall_success": round(
            sum(e["solved"] for e in episodes) / len(episodes), 4
        ),
        "by_n_necessary": group("n_necessary", [(3, 4), (5, 6), (7, 9)]),
        "by_n_distractor_vars": group("n_distractor_vars", [(0, 2), (3, 5), (6, 9)]),
        "by_mul_frac": group("mul_frac", [(0.0, 0.0), (0.01, 0.4), (0.41, 1.0)]),
        "failures_with_distractor_pick": round(
            sum(
                1 for e in episodes if not e["solved"] and e["distractor_picks"] > 0
            ) / max(sum(not e["solved"] for e in episodes), 1),
            3,
        ),
    }
    print(json.dumps(report, indent=2))
    out = Path(cfg.out or Path(cfg.ckpt).parent / "failures.json")
    out.write_text(json.dumps({"report": report, "episodes": episodes}, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
