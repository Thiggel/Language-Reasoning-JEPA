"""Evaluate latent planning with a trained checkpoint. Usage:

    python scripts/plan.py ckpt=runs/my_run/best.pt slack=0 lookahead=1
"""

from __future__ import annotations

from contextlib import nullcontext
import copy
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
    # Pinned copy of the checkpoint's own TRAINING data distribution, taken
    # before any opt-in eval_* distribution-shift override is applied. The
    # cem_cycle action prior must be fitted on this, not on the (possibly
    # shifted) evaluation distribution.
    train_cfg = copy.deepcopy(run_cfg)
    apply_eval_data_overrides(run_cfg, cfg)
    if cfg.get("candidate_interface", "feasible_menu") == "learned_catalogue":
        validate_learned_catalogue_checkpoint(run_cfg)
    if (
        cfg.get("candidate_interface", "feasible_menu") == "autonomous"
        and run_cfg.data.get("name", "igsm") != "igsm"
    ):
        # The self-rollout renders steps with the frozen-state sentence
        # decoder, which is trained on stylized iGSM step sentences; faithful
        # iGSM (igsm_real) needs its own decoder and step parser first.
        raise NotImplementedError(
            "candidate_interface=autonomous is implemented for the stylized "
            "iGSM domain only; data.name="
            f"{run_cfg.data.get('name', 'igsm')!r} is not supported yet"
        )
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
        if run_cfg.data.get("name", "igsm") == "observed_action":
            from textjepa.planning.observed_action_search import (
                evaluate_observed_action_planning,
            )

            from textjepa.planning.observed_action_search import (
                CANDIDATE_INTERFACES,
            )

            interface = cfg.get("candidate_interface", "feasible_menu")
            if interface not in CANDIDATE_INTERFACES:
                # The open-ended interfaces (ldad_cycle / generator_cycle /
                # cem_cycle) are implemented for the iGSM planners only; fail
                # loudly rather than silently planning with a menu.
                raise ValueError(
                    f"candidate_interface={interface!r} is not implemented "
                    "for the observed-action domain; supported interfaces: "
                    + ", ".join(sorted(CANDIDATE_INTERFACES))
                )

            results = evaluate_observed_action_planning(
                model, dataset, vocab, device,
                n_episodes=cfg.n_episodes,
                slack=cfg.slack,
                slack_curve=cfg.get("slack_curve", False),
                candidate_interface=interface,
                lookahead=cfg.lookahead,
                max_expand=cfg.max_expand,
                energy=cfg.get("energy", "value"),
                invalid_action_mode=cfg.get("invalid_action_mode", "noop"),
                score_control=cfg.get("score_control", "model"),
                seed=cfg.seed,
            )
        elif run_cfg.data.get("name", "igsm") == "igsm_real":
            from textjepa.planning.faithful_search import (
                CANDIDATE_INTERFACES, FaithfulPlanner,
                evaluate_faithful_planning,
            )

            interface = cfg.get("candidate_interface", "feasible_menu")
            if interface not in CANDIDATE_INTERFACES:
                # Never silently fall back to the feasible menu: that is
                # exactly how every faithful "full_catalogue" row before
                # 2026-08-13 became a mislabelled feasible-menu row.
                raise ValueError(
                    f"candidate_interface={interface!r} is not implemented "
                    "for faithful iGSM (igsm_real); supported interfaces: "
                    + ", ".join(sorted(CANDIDATE_INTERFACES))
                )
            planner = FaithfulPlanner(
                model, vocab, device, lookahead=cfg.lookahead,
                max_expand=cfg.max_expand,
                allow_oracle_future_actions=cfg.allow_oracle_future_actions,
                candidate_interface=interface,
                invalid_action_mode=cfg.get("invalid_action_mode", "noop"),
            )
            results = evaluate_faithful_planning(
                planner, dataset, cfg.n_episodes, slack=cfg.slack,
                seed=cfg.seed, slack_curve=cfg.get("slack_curve", False),
            )
        elif run_cfg.data.get("name", "igsm") == "igsm_edit":
            planner = EditPlanner(model, vocab, device, energy=cfg.energy)
            results = evaluate_edit_planning(
                planner, dataset, cfg.n_episodes, slack=cfg.slack, seed=cfg.seed
            )
        elif cfg.get("candidate_interface", "feasible_menu") == "autonomous":
            # Fully menu-free, oracle-free self-rollout: codebook proposals +
            # endpoint-Energy planning + the detached frozen-state decoder
            # emitting each step's text, scored on the FINAL ANSWER only.
            from textjepa.planning.autonomous import (
                AutonomousRollout, evaluate_autonomous, load_state_decoder,
            )

            if not cfg.get("state_decoder"):
                raise ValueError(
                    "candidate_interface=autonomous needs state_decoder=<path "
                    "to a decoder.pt from scripts/train_state_decoder.py>"
                )
            planner = AutonomousRollout(
                model, vocab, device,
                load_state_decoder(cfg.state_decoder, vocab, device),
                lookahead=cfg.lookahead,
                max_expand=cfg.max_expand,
                energy=cfg.energy,
                prior_top_k=int(cfg.get("prior_top_k", 0)),
                codebook_k=int(cfg.get("codebook_k", 64)),
                codebook_seed=int(cfg.get("codebook_seed", 0)),
                stop_on_claim=bool(cfg.get("autonomous_stop_on_claim", True)),
                feasibility_gate=bool(
                    cfg.get("autonomous_feasibility_gate", True)
                ),
                gate_calibration=str(
                    cfg.get("autonomous_gate_calibration", "midpoint")
                ),
                gate_quantile=float(cfg.get("autonomous_gate_quantile", 0.1)),
            )
            n_prior = int(cfg.get("cem_prior_problems", 64))
            prior_dataset = build_dataset(
                train_cfg, vocab, split="train", size=n_prior
            )
            planner.fit_action_prior([
                prior_dataset.problem(i)[0]
                for i in range(min(n_prior, len(prior_dataset)))
            ])
            if planner.feasibility_gate:
                # Calibrated on TRAINING problems only (same pinned pre-
                # override distribution as the codebook): the quantile of the
                # cycle scores of genuinely feasible training actions.
                n_gate = int(cfg.get("autonomous_gate_problems", 32))
                gate_dataset = build_dataset(
                    train_cfg, vocab, split="train", size=n_gate
                )
                threshold = planner.calibrate_feasibility_gate(
                    [
                        gate_dataset.problem(i)[0]
                        for i in range(min(n_gate, len(gate_dataset)))
                    ],
                    seed=cfg.seed,
                )
                print(f"feasibility gate threshold: {threshold:.3f}")
            results = evaluate_autonomous(
                planner, dataset, cfg.n_episodes, slack=cfg.slack,
                seed=cfg.seed, slack_curve=cfg.get("slack_curve", False),
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
                generator_samples=int(cfg.get("generator_samples", 16)),
                generator_top_p=float(cfg.get("generator_top_p", 1.0)),
                generator_temperature=float(
                    cfg.get("generator_temperature", 1.0)
                ),
                cem_population=int(cfg.get("cem_population", 64)),
                cem_elites=int(cfg.get("cem_elites", 8)),
                cem_iters=int(cfg.get("cem_iters", 3)),
                cem_offmanifold_lambda=float(
                    cfg.get("cem_offmanifold_lambda", 1.0)
                ),
                cem_prior_anchor=float(cfg.get("cem_prior_anchor", 0.1)),
                codebook_k=int(cfg.get("codebook_k", 64)),
                codebook_seed=int(cfg.get("codebook_seed", 0)),
            )
            if planner.proposer is not None:
                # The proposal distribution (the cem_cycle Gaussian or the
                # codebook_cycle k-means codebook) is fitted on TRAINING
                # problems only, from the pinned pre-override distribution.
                n_prior = int(cfg.get("cem_prior_problems", 64))
                prior_dataset = build_dataset(
                    train_cfg, vocab, split="train", size=n_prior
                )
                planner.fit_action_prior([
                    prior_dataset.problem(i)[0]
                    for i in range(min(n_prior, len(prior_dataset)))
                ])
            results = evaluate_planning(
                planner, dataset, cfg.n_episodes, slack=cfg.slack,
                seed=cfg.seed, slack_curve=cfg.get("slack_curve", False),
            )
    for name, metrics in results.items():
        line = "  ".join(
            f"{k}={v:.3f}" for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        print(f"{name:16s} {line}")
        curve = metrics.get("success_by_slack")
        if curve:
            print(
                f"{'':16s} success_by_slack  "
                + "  ".join(f"{k}={v:.3f}" for k, v in curve.items())
            )
    suffix = "" if cfg.energy == "value" else f"_{cfg.energy}"
    if cfg.get("hierarchy", False):
        suffix += "_hier"
    if cfg.get("simulator", "latent") == "symbolic":
        suffix += "_sym"
    if cfg.lookahead > 1:
        suffix += "_oracle_actions"
    if cfg.get("slack_curve", False):
        # A slack curve is scored from one generous-budget run, so it would
        # otherwise overwrite the fixed-slack run at the same nominal slack.
        suffix += "_slackcurve"
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
            "cem_population": int(cfg.get("cem_population", 64)),
            "cem_elites": int(cfg.get("cem_elites", 8)),
            "cem_iters": int(cfg.get("cem_iters", 3)),
            "cem_offmanifold_lambda": float(
                cfg.get("cem_offmanifold_lambda", 1.0)
            ),
            "cem_prior_anchor": float(cfg.get("cem_prior_anchor", 0.1)),
            "cem_prior_problems": int(cfg.get("cem_prior_problems", 64)),
            "codebook_k": int(cfg.get("codebook_k", 64)),
            "codebook_seed": int(cfg.get("codebook_seed", 0)),
            "generator_samples": int(cfg.get("generator_samples", 16)),
            "generator_top_p": float(cfg.get("generator_top_p", 1.0)),
            "generator_temperature": float(
                cfg.get("generator_temperature", 1.0)
            ),
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
