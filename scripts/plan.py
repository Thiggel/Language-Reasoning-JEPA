"""Evaluate latent planning with a trained checkpoint. Usage:

    python scripts/plan.py ckpt=runs/my_run/best.pt slack=0 lookahead=1
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from textjepa.planning import LatentPlanner, evaluate_planning
from textjepa.planning.search import validate_learned_catalogue_checkpoint
from textjepa.planning.edit_search import EditPlanner, evaluate_edit_planning
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import (
    apply_eval_data_overrides,
    build_dataset,
    load_run,
)


@hydra.main(config_path="../configs", config_name="plan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    model, vocab, run_cfg = load_run(cfg.ckpt, cfg.device)
    apply_eval_data_overrides(run_cfg, cfg)
    if cfg.get("candidate_interface", "feasible_menu") == "learned_catalogue":
        validate_learned_catalogue_checkpoint(run_cfg)
    split = cfg.get("split", "val")
    dataset = build_dataset(run_cfg, vocab, split=split)
    device = torch.device(cfg.device)
    if (
        run_cfg.data.get("name", "igsm") == "igsm"
        and getattr(model, "geo_rank_score_mode", "value") == "td_jepa"
    ):
        # Faithful TD-JEPA needs the task-reward projection z_r, ridge-
        # regressed from rewards on training-trace states.  Fit it here at
        # plan time from the checkpoint's own training distribution.
        from textjepa.data.igsm.dataset import collate

        fit_dataset = build_dataset(run_cfg, vocab, split="train")
        n_fit = min(
            len(fit_dataset), int(cfg.get("td_jepa_fit_examples", 256))
        )
        batches = []
        for start in range(0, n_fit, 32):
            items = [
                fit_dataset[i] for i in range(start, min(start + 32, n_fit))
            ]
            batch = collate(items, vocab.pad_id)
            batches.append({
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            })
        model.fit_td_jepa_reward_projection(batches)
    measure_flops = bool(cfg.get("measure_flops", False))
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:  # pragma: no cover - depends on the cluster torch build
        FlopCounterMode = None
    flop_counter = (
        FlopCounterMode(display=False)
        if measure_flops and FlopCounterMode is not None
        else nullcontext()
    )
    with flop_counter:
        if run_cfg.data.get("name", "igsm") == "igsm_real":
            from textjepa.planning.faithful_search import (
                FaithfulPlanner, evaluate_faithful_planning,
            )

            planner = FaithfulPlanner(
                model, vocab, device, lookahead=cfg.lookahead,
                max_expand=cfg.max_expand,
                allow_oracle_future_actions=cfg.allow_oracle_future_actions,
            )
            results = evaluate_faithful_planning(
                planner, dataset, cfg.n_episodes, slack=cfg.slack, seed=cfg.seed
            )
        elif run_cfg.data.get("name", "igsm") == "igsm_edit":
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
                allow_oracle_future_actions=cfg.allow_oracle_future_actions,
                score_control=cfg.get("score_control", "model"),
                search_algorithm=cfg.get("search_algorithm", "shooting"),
                transition_energy_composition=cfg.get(
                    "transition_energy_composition", "terminal"
                ),
                hybrid_local_pruning=cfg.get("hybrid_local_pruning", False),
                candidate_interface=cfg.get(
                    "candidate_interface", "feasible_menu"
                ),
                invalid_action_mode=cfg.get("invalid_action_mode", "noop"),
                prior_top_k=int(cfg.get("prior_top_k", 0)),
                prior_top_p=float(cfg.get("prior_top_p", 1.0)),
                prior_feasibility_gate=bool(
                    cfg.get("prior_feasibility_gate", False)
                ),
            )
            results = evaluate_planning(
                planner, dataset, cfg.n_episodes, slack=cfg.slack,
                seed=cfg.seed, slack_curve=cfg.get("slack_curve", False),
            )
    for name, metrics in results.items():
        line = "  ".join(
            f"{k}={v:.3f}" for k, v in metrics.items()
            if isinstance(v, (int, float))
        )
        print(f"{name:16s} {line}")
    suffix = "" if cfg.energy == "value" else f"_{cfg.energy}"
    if cfg.get("hierarchy", False):
        suffix += "_hier"
    if cfg.get("simulator", "latent") == "symbolic":
        suffix += "_sym"
    if cfg.lookahead > 1:
        suffix += "_oracle_actions"
    split_suffix = "" if split == "val" else f"_{split}"
    out = Path(
        cfg.out or Path(cfg.ckpt).parent
        / f"plan_slack{cfg.slack}_look{cfg.lookahead}{suffix}{split_suffix}.json"
    )
    out.write_text(json.dumps(results, indent=2))
    print(f"saved to {out}")
    if cfg.get("compute_out"):
        total_flops = (
            int(flop_counter.get_total_flops())
            if measure_flops and FlopCounterMode is not None
            else None
        )
        checkpoint = Path(cfg.ckpt)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        compute = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "lookahead": int(cfg.lookahead),
            "max_expand": int(cfg.max_expand),
            "n_episodes": int(cfg.n_episodes),
            "slack": int(cfg.slack),
            "oracle_future_action_tree": bool(cfg.allow_oracle_future_actions),
            "score_control": str(cfg.get("score_control", "model")),
            "search_algorithm": str(cfg.get("search_algorithm", "shooting")),
            "transition_energy_composition": str(cfg.get(
                "transition_energy_composition", "terminal"
            )),
            "simulator": str(cfg.get("simulator", "latent")),
            "energy": str(cfg.get("energy", "value")),
            "candidate_interface": str(cfg.get(
                "candidate_interface", "feasible_menu"
            )),
            "candidate_protocol": (
                "global-beam-v1"
                if cfg.get("search_algorithm", "shooting") == "beam"
                else "balanced-fixed-depth-absorbing-v2"
            ),
            "flop_measurement_requested": measure_flops,
            "flop_measurement_supported": FlopCounterMode is not None,
            "measured_eval_flops": total_flops,
            "measured_flops_per_episode": (
                total_flops / int(cfg.n_episodes) if total_flops is not None else None
            ),
        }
        compute_path = Path(cfg.compute_out)
        compute_path.parent.mkdir(parents=True, exist_ok=True)
        compute_path.write_text(json.dumps(compute, indent=2) + "\n")
        print(f"saved compute metadata to {compute_path}")


if __name__ == "__main__":
    main()
