"""Evaluate deployable top-down macro planning on stylized iGSM."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from textjepa.planning.evaluate import evaluate_planning
from textjepa.planning.hierarchical_search import HierarchicalLatentPlanner
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, load_run


@hydra.main(
    config_path="../configs", config_name="hierarchical_plan", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    model, vocab, run_cfg = load_run(cfg.ckpt, cfg.device)
    if cfg.low_support_ckpt:
        payload = torch.load(
            cfg.low_support_ckpt, map_location="cpu", weights_only=False
        )
        support = {
            name: value for name, value in payload["model"].items()
            if name.startswith("core.action_support_head.")
        }
        missing, unexpected = model.load_state_dict(support, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected support keys: {unexpected}")
        if not support:
            raise RuntimeError("support checkpoint has no action support head")
        print(
            f"loaded {len(support)} action-support tensors from "
            f"{cfg.low_support_ckpt}"
        )
    if run_cfg.data.get("name", "igsm") not in {"igsm", "igsm_real"}:
        raise ValueError("hierarchical planning supports stylized or faithful iGSM")
    if cfg.eval_steps_range is not None:
        run_cfg.data.steps_range = list(cfg.eval_steps_range)
    if cfg.eval_n_vars_range is not None:
        run_cfg.data.n_vars_range = list(cfg.eval_n_vars_range)
    if cfg.eval_op_range is not None:
        run_cfg.data.op_range = list(cfg.eval_op_range)
    if cfg.eval_max_op is not None:
        run_cfg.data.max_op = int(cfg.eval_max_op)
    if cfg.eval_max_edge is not None:
        run_cfg.data.max_edge = int(cfg.eval_max_edge)
    if cfg.eval_dataset_seed is not None:
        if cfg.split == "train":
            run_cfg.data.train_seed = cfg.eval_dataset_seed
        elif cfg.split == "val":
            run_cfg.data.val_seed = cfg.eval_dataset_seed
        else:
            run_cfg.data.test_seed = cfg.eval_dataset_seed
    dataset = build_dataset(run_cfg, vocab, split=cfg.split)
    ensemble_models = [
        load_run(path, cfg.device)[0]
        for path in cfg.ensemble_ckpts
    ]
    planner = HierarchicalLatentPlanner(
        model,
        vocab,
        torch.device(cfg.device),
        energy=cfg.energy,
        method=cfg.method,
        high_horizon=cfg.high_horizon,
        n_samples=cfg.n_samples,
        cem_iters=cfg.cem_iters,
        n_elites=cfg.n_elites,
        elite_frac=cfg.elite_frac,
        mean_ema=cfg.mean_ema,
        variance_ema=cfg.variance_ema,
        scale_update=cfg.scale_update,
        cem_return=cfg.cem_return,
        cem_tolerance=cfg.cem_tolerance,
        min_std=cfg.min_std,
        cem_domain=cfg.cem_domain,
        density_weight=cfg.density_weight,
        macro_q_aux_weight=cfg.macro_q_aux_weight,
        path_value_weight=cfg.path_value_weight,
        learned_support_weight=cfg.learned_support_weight,
        learned_support_threshold=cfg.learned_support_threshold,
        macro_knn_weight=cfg.macro_knn_weight,
        macro_gmm_weight=cfg.macro_gmm_weight,
        macro_gmm_components=cfg.macro_gmm_components,
        macro_gmm_ridge=cfg.macro_gmm_ridge,
        macro_project_to_span=cfg.macro_project_to_span,
        reachability_weight=cfg.reachability_weight,
        reachability_mode=cfg.reachability_mode,
        reachability_topk=cfg.reachability_topk,
        measured_reachability_weight=cfg.measured_reachability_weight,
        measured_reachability_topk=cfg.measured_reachability_topk,
        measured_reachability_horizon=cfg.measured_reachability_horizon,
        measured_latent_goal_weight=cfg.measured_latent_goal_weight,
        measured_symbolic_goal_weight=cfg.measured_symbolic_goal_weight,
        controller_remaining_weight=cfg.controller_remaining_weight,
        controller_residual_weight=cfg.controller_residual_weight,
        collect_controller_outcomes=cfg.collect_controller_outcomes,
        ensemble_models=ensemble_models,
        epistemic_weight=cfg.epistemic_weight,
        low_subgoal_weight=cfg.low_subgoal_weight,
        low_value_weight=cfg.low_value_weight,
        low_method=cfg.low_method,
        low_action_source=cfg.low_action_source,
        low_horizon=cfg.low_horizon,
        low_max_expand=cfg.low_max_expand,
        low_cem_samples=cfg.low_cem_samples,
        low_cem_iters=cfg.low_cem_iters,
        low_cem_elites=cfg.low_cem_elites,
        low_cem_variance_ema=cfg.low_cem_variance_ema,
        low_cem_min_std=cfg.low_cem_min_std,
        low_density_weight=cfg.low_density_weight,
        low_support_weight=cfg.low_support_weight,
        low_support_threshold=cfg.low_support_threshold,
        low_cem_return=cfg.low_cem_return,
        allow_oracle_low_actions=cfg.allow_oracle_low_actions,
        subgoal_source=cfg.subgoal_source,
        discrete_execute_macro=cfg.discrete_execute_macro,
        discrete_first_value_weight=cfg.discrete_first_value_weight,
        flat_fallback_threshold=cfg.flat_fallback_threshold,
        adaptive_high_horizon=cfg.adaptive_high_horizon,
    )
    results = evaluate_planning(
        planner, dataset, cfg.n_episodes, slack=cfg.slack, seed=cfg.seed
    )
    for name, metrics in results.items():
        print(name, "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    suffix = (
        f"{cfg.subgoal_source}_{cfg.method}_hh{cfg.high_horizon}"
        f"_low{cfg.low_method}-{cfg.low_action_source}-h{cfg.low_horizon}"
        f"_n{cfg.n_samples}_i{cfg.cem_iters}"
        f"_mm{cfg.mean_ema}_vm{cfg.variance_ema}_{cfg.scale_update}"
        f"_{cfg.cem_return}_dw{cfg.density_weight}"
        f"_qaux{cfg.macro_q_aux_weight}_pathv{cfg.path_value_weight}"
        f"_domain{cfg.cem_domain}_reach{cfg.reachability_weight}"
        f"-{cfg.reachability_mode}-k{cfg.reachability_topk}"
        f"_mreach{cfg.measured_reachability_weight}"
        f"-k{cfg.measured_reachability_topk}"
        f"-h{cfg.measured_reachability_horizon}"
        f"-lg{cfg.measured_latent_goal_weight}"
        f"-sg{cfg.measured_symbolic_goal_weight}"
        f"_crem{cfg.controller_remaining_weight}"
        f"_cres{cfg.controller_residual_weight}"
        f"_ens{len(cfg.ensemble_ckpts)}_epi{cfg.epistemic_weight}"
        f"_lsw{cfg.learned_support_weight}_{cfg.energy}"
        f"_mkw{cfg.macro_knn_weight}_mproj{cfg.macro_project_to_span}"
        f"_gmm{cfg.macro_gmm_weight}-m{cfg.macro_gmm_components}"
        f"_lsgw{cfg.low_subgoal_weight}"
        f"_lst{cfg.learned_support_threshold}"
        f"_lowsw{cfg.low_support_weight}"
        f"_lowst{cfg.low_support_threshold}"
        f"_dem{cfg.discrete_execute_macro}"
        f"_dfvw{cfg.discrete_first_value_weight}"
        f"_fft{cfg.flat_fallback_threshold}"
        f"_adaptive{cfg.adaptive_high_horizon}"
    )
    out = Path(cfg.out or Path(cfg.ckpt).parent / f"plan_hier_{suffix}_s{cfg.slack}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if planner.measured_reachability_diagnostics:
        diagnostics = planner.measured_reachability_diagnostics
        results["measured_reachability"] = {
            "changed_rate": sum(row["changed"] for row in diagnostics)
            / len(diagnostics),
            "base_residual": sum(
                row["base_residual"] for row in diagnostics
            ) / len(diagnostics),
            "selected_residual": sum(
                row["selected_residual"] for row in diagnostics
            ) / len(diagnostics),
            "base_cost_increase": sum(
                row["selected_base_cost"] - row["base_cost"]
                for row in diagnostics
            ) / len(diagnostics),
            "base_remaining": sum(
                row["base_remaining"] for row in diagnostics
            ) / len(diagnostics),
            "selected_remaining": sum(
                row["selected_remaining"] for row in diagnostics
            ) / len(diagnostics),
            "base_distractors": sum(
                row["base_distractors"] for row in diagnostics
            ) / len(diagnostics),
            "selected_distractors": sum(
                row["selected_distractors"] for row in diagnostics
            ) / len(diagnostics),
            "decisions": len(diagnostics),
        }
    if planner.high_horizon_counts:
        results["high_horizon_counts"] = {
            str(horizon): count
            for horizon, count in sorted(
                planner.high_horizon_counts.items()
            )
        }
    if planner.discrete_plan_diagnostics:
        diagnostics = planner.discrete_plan_diagnostics
        results["discrete_plan_diagnostics"] = {
            name: sum(row[name] for row in diagnostics) / len(diagnostics)
            for name in diagnostics[0]
        }
        results["discrete_plan_diagnostics"]["decisions"] = len(diagnostics)
    if cfg.controller_dataset_out:
        if not planner.controller_outcome_batches:
            raise ValueError(
                "controller_dataset_out requires collect_controller_outcomes=true"
            )
        fields = planner.controller_outcome_batches[0].keys()
        dataset = {
            name: torch.cat([
                batch[name] for batch in planner.controller_outcome_batches
            ])
            for name in fields
        }
        dataset["group"] = torch.cat([
            torch.full(
                (len(batch["state"]),), group,
                dtype=torch.long,
            )
            for group, batch in enumerate(
                planner.controller_outcome_batches
            )
        ])
        dataset["source_checkpoint"] = str(cfg.ckpt)
        dataset_path = Path(cfg.controller_dataset_out)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, dataset_path)
        print(
            f"saved {len(dataset['state'])} controller outcomes to "
            f"{dataset_path}"
        )
    out.write_text(json.dumps(results, indent=2))
    if planner.cem_traces:
        trace_out = out.with_name(out.stem + "_cem_trace.json")
        trace_out.write_text(json.dumps(planner.cem_traces, indent=2))
        print(f"saved CEM trace to {trace_out}")
    if planner.low_cem_traces:
        trace_out = out.with_name(out.stem + "_low_cem_trace.json")
        trace_out.write_text(json.dumps(planner.low_cem_traces, indent=2))
        print(f"saved low-level CEM trace to {trace_out}")
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
