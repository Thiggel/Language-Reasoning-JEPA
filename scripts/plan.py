"""Evaluate latent planning with a trained checkpoint. Usage:

    python scripts/plan.py ckpt=runs/my_run/best.pt slack=0 lookahead=1
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from textjepa.planning import LatentPlanner, evaluate_planning
from textjepa.planning.edit_search import EditPlanner, evaluate_edit_planning
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, load_run


@hydra.main(config_path="../configs", config_name="plan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    model, vocab, run_cfg = load_run(cfg.ckpt, cfg.device)
    dataset = build_dataset(run_cfg, vocab, split="val")
    device = torch.device(cfg.device)
    if run_cfg.data.get("name", "igsm") == "igsm_edit":
        planner = EditPlanner(model, vocab, device, energy=cfg.energy)
        results = evaluate_edit_planning(
            planner, dataset, cfg.n_episodes, slack=cfg.slack, seed=cfg.seed
        )
    else:
        planner = LatentPlanner(
            model, vocab, device, lookahead=cfg.lookahead,
            max_expand=cfg.max_expand, energy=cfg.energy,
            hierarchy=cfg.get("hierarchy", False),
            simulator=cfg.get("simulator", "latent"),
        )
        results = evaluate_planning(
            planner, dataset, cfg.n_episodes, slack=cfg.slack, seed=cfg.seed
        )
    for name, metrics in results.items():
        line = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        print(f"{name:16s} {line}")
    suffix = "" if cfg.energy == "value" else f"_{cfg.energy}"
    if cfg.get("hierarchy", False):
        suffix += "_hier"
    if cfg.get("simulator", "latent") == "symbolic":
        suffix += "_sym"
    out = Path(
        cfg.out
        or Path(cfg.ckpt).parent / f"plan_slack{cfg.slack}_look{cfg.lookahead}{suffix}.json"
    )
    out.write_text(json.dumps(results, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
