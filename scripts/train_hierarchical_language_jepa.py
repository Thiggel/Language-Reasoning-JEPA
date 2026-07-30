#!/usr/bin/env python3
"""Train staged JEPA modules from offline frozen-LM hidden-state batches.

Input is a ``torch.save`` dictionary with rectangular tensors:
``hidden_states [N,T,D]``, ``token_ids [N,T]``, and
``boundaries [N,J]`` padded by -1.  Generation, symbolic verification, and
exact counterfactual re-encoding intentionally happen outside this learner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from textjepa.analysis.compute import (
    ComputeLedger,
    embedding_training_ops,
    parameter_count,
    training_flops,
)
from textjepa.models.hierarchical_language_jepa import (
    HIERARCHICAL_LANGUAGE_ARCHITECTURE,
    HierarchicalLanguageJEPA,
    HierarchicalLanguageJEPAConfig,
)
from textjepa.training.hierarchical_language import (
    DenseLossWeights,
    HierarchicalLanguageLearner,
    ResearchStage,
    SparseRolloutSchedule,
    value_distillation_loss,
)
from textjepa.data.language_planning import (
    MODEL_ID,
    MODEL_REVISION,
    PRIMARY_EOS_TOKEN_ID,
    TRANSFORMERS_VERSION,
    collate_counterfactual_records,
)
from textjepa.data.provenance import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--counterfactual-features", type=Path)
    parser.add_argument("--value-replay", type=Path)
    parser.add_argument("--planner-replay", type=Path)
    parser.add_argument(
        "--init-checkpoint", type=Path,
        help="Admitted checkpoint from the preceding research stage.",
    )
    parser.add_argument(
        "--admission", type=Path,
        help="JSON validity-gate record admitting this stage.",
    )
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument(
        "--allow-unpinned-features", action="store_true",
        help="Test/debug escape hatch; production artifacts must stay pinned.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=[stage.name for stage in ResearchStage],
        default=None,
    )
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--pad-id", type=int, default=248044)
    parser.add_argument("--d-token", type=int, default=256)
    parser.add_argument("--d-sentence", type=int, default=128)
    parser.add_argument("--d-action", type=int, default=32)
    parser.add_argument("--predictor-width", type=int, default=512)
    parser.add_argument("--token-layers", type=int, default=4)
    parser.add_argument("--sentence-layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--token-context", type=int, default=64)
    parser.add_argument("--sentence-context", type=int, default=32)
    parser.add_argument("--max-span", type=int, default=64)
    parser.add_argument("--cache-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--replay-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_features(
    path: Path, *, require_provenance: bool = False
) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("feature file must be a dictionary")
    if "input_ids" not in payload and "token_ids" in payload:
        payload = {**payload, "input_ids": payload["token_ids"]}
    required = {"hidden_states", "input_ids", "boundaries"}
    if not required <= payload.keys():
        raise ValueError(f"feature file must contain {sorted(required)}")
    hidden = payload["hidden_states"]
    tokens = payload["input_ids"]
    boundaries = payload["boundaries"]
    if len(hidden) == 0:
        raise ValueError("feature dataset is empty")
    if hidden.ndim != 3 or tokens.shape != hidden.shape[:2]:
        raise ValueError("hidden_states and token_ids have incompatible shapes")
    if not hidden.is_floating_point():
        raise ValueError("hidden states must be floating point")
    if not torch.isfinite(hidden).all():
        raise ValueError("hidden states contain non-finite values")
    if tokens.dtype not in {
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    }:
        raise ValueError("token IDs must use an integer dtype")
    if boundaries.dtype not in {
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    }:
        raise ValueError("boundaries must use an integer dtype")
    if boundaries.ndim != 2 or len(boundaries) != len(hidden):
        raise ValueError("boundaries must be [examples, boundary_slots]")
    if hidden.requires_grad:
        raise ValueError("offline frozen-LM features must not require gradients")
    optional = {
        "attention_mask", "prompt_len", "solution_end",
        "step_boundary_mask", "reasoning_depth",
    }
    result = {name: payload[name] for name in required}
    result.update({
        name: payload[name] for name in optional if name in payload
    })
    if "attention_mask" in result:
        mask = result["attention_mask"]
        if mask.dtype != torch.bool or mask.shape != tokens.shape:
            raise ValueError("attention_mask must be boolean and match token IDs")
        if bool((~mask[:, :-1] & mask[:, 1:]).any()):
            raise ValueError("attention padding must be right-contiguous")
    if "prompt_len" in result or "solution_end" in result:
        if not {"prompt_len", "solution_end"} <= result.keys():
            raise ValueError("prompt_len and solution_end must be supplied together")
        prompt_len, solution_end = result["prompt_len"], result["solution_end"]
        if prompt_len.shape != (len(hidden),) or solution_end.shape != (len(hidden),):
            raise ValueError("prompt_len and solution_end must be vectors")
        for row in range(len(hidden)):
            valid_boundaries = boundaries[row][boundaries[row] >= 0]
            if len(valid_boundaries) < 2:
                raise ValueError("each example needs boundary endpoints")
            if int(valid_boundaries[0]) != int(prompt_len[row]) or int(
                valid_boundaries[-1]
            ) != int(solution_end[row]):
                raise ValueError("boundary endpoints disagree with prompt/solution")
    if require_provenance:
        expected = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise ValueError(
                    f"feature provenance mismatch for {name}: "
                    f"expected {value!r}, got {payload.get(name)!r}"
                )
        mandatory = {
            "attention_mask", "prompt_len", "solution_end",
            "step_boundary_mask", "reasoning_depth",
            "canonical_state_ids", "problem_id", "template_family",
            "graph_family", "symbolically_verified",
            "dataset_fingerprint",
        }
        missing = mandatory - payload.keys()
        if missing:
            raise ValueError(
                f"feature artifact lacks mandatory fields: {sorted(missing)}"
            )
        if not all(bool(value) for value in payload["symbolically_verified"]):
            raise ValueError("feature artifact contains unverified examples")
        for name in ("problem_id", "template_family", "graph_family"):
            if len(payload[name]) != len(hidden) or any(
                not str(value) for value in payload[name]
            ):
                raise ValueError(f"invalid {name} feature metadata")
        if len(set(map(str, payload["problem_id"]))) != len(hidden):
            raise ValueError("feature artifact contains duplicate problem IDs")
        if not isinstance(payload["dataset_fingerprint"], str) or len(
            payload["dataset_fingerprint"]
        ) != 64:
            raise ValueError("feature dataset fingerprint is invalid")
        canonical = payload["canonical_state_ids"]
        boundary_mask = payload["step_boundary_mask"]
        if canonical.shape != boundaries.shape or (
            boundary_mask.shape != boundaries.shape
        ) or boundary_mask.dtype != torch.bool:
            raise ValueError("boundary diagnostics do not align")
        for row in range(len(hidden)):
            end = int(payload["solution_end"][row])
            if end >= tokens.shape[1] or int(tokens[row, end]) != (
                PRIMARY_EOS_TOKEN_ID
            ):
                raise ValueError("primary EOS is missing at solution_end")
    return result


def _step_if_trainable(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model: HierarchicalLanguageJEPA,
    ema_momentum: float,
) -> bool:
    """Apply an update, or explicitly skip a frozen-only replay minibatch."""
    if not loss.requires_grad:
        return False
    loss.backward()
    optimizer.step()
    model.update_targets(ema_momentum)
    return True


def _validate_grounded_planner_replay(
    payload: dict,
    *,
    dataset_fingerprint: str,
    checkpoint_sha256: str,
    require_exact_worker_achievement: bool,
) -> None:
    required_header = {
        "architecture": HIERARCHICAL_LANGUAGE_ARCHITECTURE,
        "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "dataset_fingerprint": dataset_fingerprint,
        "source_checkpoint_sha256": checkpoint_sha256,
        "exact_reencoded": True,
    }
    if any(payload.get(name) != value for name, value in required_header.items()):
        raise ValueError("planner replay provenance is incompatible")
    transitions = payload.get("grounded_transitions")
    if not isinstance(transitions, list) or not transitions:
        raise ValueError(
            "closed-loop replay requires nonempty grounded_transitions"
        )
    allowed_sources = {
        "cem", "elite", "near_elite", "random_prior",
        "worker_achieved", "high_scoring_failure",
    }
    for transition in transitions:
        if transition.get("candidate_source") not in allowed_sources:
            raise ValueError("planner replay candidate source is invalid")
        iteration = transition.get("cem_iteration")
        if not isinstance(iteration, int) or iteration < 0:
            raise ValueError("planner replay CEM iteration is invalid")
        if transition.get("exact_reencoded") is not True:
            raise ValueError("planner replay transition is not exact-grounded")
        predicted = transition.get("predicted_endpoint")
        achieved = transition.get("achieved_endpoint")
        if not isinstance(predicted, torch.Tensor) or not isinstance(
            achieved, torch.Tensor
        ) or predicted.shape != achieved.shape or not torch.isfinite(
            predicted
        ).all() or not torch.isfinite(achieved).all():
            raise ValueError("planner replay endpoints are invalid")
        if require_exact_worker_achievement and transition.get(
            "worker_achieved"
        ) is not True:
            raise ValueError("planner replay lacks exact worker achievement")
    records = payload.get("counterfactual_records")
    if not isinstance(records, list) or not records:
        raise ValueError(
            "closed-loop replay must contain planner-selected transitions"
        )


def _record_dense_flops(
    ledger: ComputeLedger,
    model: HierarchicalLanguageJEPA,
    stage: ResearchStage,
    batch: dict[str, torch.Tensor],
) -> None:
    prompt = batch.get("prompt_len")
    end = batch.get("solution_end")
    if prompt is not None and end is not None:
        token_items = int((end - prompt).sum())
    else:
        token_items = int(batch.get(
            "attention_mask", batch["input_ids"].ne(model.config.pad_id)
        ).sum())
    boundaries = batch.get("step_boundary_mask")
    sentence_items = (
        int(boundaries.sum() - len(boundaries))
        if boundaries is not None else 0
    )
    modules: list[tuple[str, torch.nn.Module, int]] = []
    if stage in {
        ResearchStage.TOKEN_JEPA, ResearchStage.CLOSED_LOOP_REANALYSIS
    }:
        modules.extend([
            ("dense_e0", model.e0, token_items),
            ("dense_token_action", model.token_action, token_items),
            ("dense_p0", model.p0, token_items),
        ])
    if (
        ResearchStage.SENTENCE_JEPA <= stage < ResearchStage.VALUE_DISTILLATION
    ) or stage == ResearchStage.CLOSED_LOOP_REANALYSIS:
        modules.extend([
            ("dense_e0_to_1", model.e0_to_1, sentence_items + len(batch[
                "input_ids"
            ])),
            ("dense_a1", model.a1, token_items),
            ("dense_p1", model.p1, sentence_items),
        ])
    if stage >= ResearchStage.MACRO_ACTION and model.pi1 is not None:
        modules.append(("dense_pi1", model.pi1, sentence_items))
    for name, module, items in modules:
        if any(parameter.requires_grad for parameter in module.parameters()):
            estimated = (
                embedding_training_ops(module.embedding_dim, items)
                if isinstance(module, torch.nn.Embedding)
                else training_flops(
                    parameter_count(module, trainable_only=True), items
                )
            )
            ledger.add(
                name,
                estimated_flops=estimated,
                calls=0,
                items=items,
            )


def _record_counterfactual_flops(
    ledger: ComputeLedger,
    model: HierarchicalLanguageJEPA,
    branch: dict[str, torch.Tensor],
    rollout_horizon: int = 1,
) -> None:
    token_items = int(branch["lengths"].sum())
    sentence_mask = branch["sentence_eligible"]
    sentence_items = int(sentence_mask.sum())
    sentence_tokens = int(
        (branch["lengths"] * sentence_mask.long()).sum()
    )
    for name, module, items in (
        ("replay_e0", model.e0, token_items),
        ("replay_token_action", model.token_action, token_items),
        ("replay_p0", model.p0, token_items),
        ("replay_e0_to_1", model.e0_to_1, sentence_items),
        ("replay_a1", model.a1, sentence_tokens),
        ("replay_p1", model.p1, sentence_items),
    ):
        if items and any(
            parameter.requires_grad for parameter in module.parameters()
        ):
            estimated = (
                embedding_training_ops(module.embedding_dim, items)
                if isinstance(module, torch.nn.Embedding)
                else training_flops(
                    parameter_count(module, trainable_only=True), items
                )
            )
            ledger.add(
                name,
                estimated_flops=estimated,
                calls=0,
                items=items,
            )
    if rollout_horizon > 1:
        eligible = int((branch["lengths"] >= rollout_horizon).sum())
        rollout_items = eligible * rollout_horizon
        if rollout_items and any(
            parameter.requires_grad for parameter in model.p0.parameters()
        ):
            ledger.add(
                "replay_p0_rollout",
                estimated_flops=training_flops(
                    parameter_count(model.p0, trainable_only=True),
                    rollout_items,
                ),
                calls=0,
                items=rollout_items,
            )
            ledger.add(
                "replay_token_action_rollout",
                estimated_flops=embedding_training_ops(
                    model.token_action.embedding_dim, rollout_items
                ),
                calls=0,
                items=rollout_items,
            )


def _load_admission(
    path: Path | None, stage: ResearchStage, *, required: bool = True
) -> dict | None:
    if stage <= ResearchStage.FLAT_ORACLE_TOKEN or not required:
        return None
    if path is None:
        raise ValueError(
            f"{stage.name} requires --admission from the preceding validity gate"
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("passed") is not True:
        raise ValueError("admission record did not pass")
    required = {
        ResearchStage.SENTENCE_JEPA: "FLAT_ORACLE_TOKEN",
        ResearchStage.ORACLE_WAYPOINT: "SENTENCE_JEPA",
        ResearchStage.DYNAMIC_COMMUTATION: "ORACLE_WAYPOINT",
        ResearchStage.MACRO_ACTION: "DYNAMIC_COMMUTATION",
        ResearchStage.ORACLE_HIGH_LEVEL: "MACRO_ACTION",
        ResearchStage.VALUE_DISTILLATION: "ORACLE_HIGH_LEVEL",
        ResearchStage.FULL_HIERARCHY: "VALUE_DISTILLATION",
        ResearchStage.CLOSED_LOOP_REANALYSIS: "FULL_HIERARCHY",
    }
    admitted = ResearchStage[record["admitted_stage"]]
    if admitted < ResearchStage[required[stage]]:
        raise ValueError(
            f"admission only reaches {admitted.name}; {stage.name} requires "
            f"{required[stage]}"
        )
    if not isinstance(record.get("metrics"), dict):
        raise ValueError("admission record must contain measured metrics")
    required_metrics = {
        ResearchStage.SENTENCE_JEPA: {
            "exact_oracle_gain", "model_oracle_gap",
        },
        ResearchStage.ORACLE_WAYPOINT: {
            "heldout_sentence_dynamics", "sentence_effective_rank",
            "symbolic_state_purity",
        },
        ResearchStage.DYNAMIC_COMMUTATION: {
            "exact_symbolic_waypoint_success",
            "latent_symbolic_disagreement",
        },
        ResearchStage.MACRO_ACTION: {
            "commutation_error", "worker_success",
        },
        ResearchStage.ORACLE_HIGH_LEVEL: {
            "macro_prior_nll", "worker_executability",
        },
        ResearchStage.VALUE_DISTILLATION: {
            "oracle_success_gain", "optimizer_curse_regret",
        },
        ResearchStage.FULL_HIERARCHY: {
            "top_one_regret", "rank_correlation",
        },
        ResearchStage.CLOSED_LOOP_REANALYSIS: {
            "no_terminal_success", "predicted_realized_gap",
        },
    }[stage]
    missing_metrics = required_metrics - record["metrics"].keys()
    if missing_metrics:
        raise ValueError(
            f"admission metrics are incomplete: {sorted(missing_metrics)}"
        )
    if any(
        not isinstance(record["metrics"][name], (int, float))
        or not torch.isfinite(torch.tensor(record["metrics"][name]))
        for name in required_metrics
    ):
        raise ValueError("admission metrics must be finite scalars")
    if not record.get("dataset_fingerprint") or not record.get(
        "admitted_checkpoint_sha256"
    ):
        raise ValueError(
            "admission must bind a dataset and exact checkpoint"
        )
    return record


def _initialize_from_checkpoint(
    model: HierarchicalLanguageJEPA,
    learner: HierarchicalLanguageLearner,
    checkpoint_path: Path | None,
    stage: ResearchStage,
    *,
    required: bool = True,
    admission: dict | None = None,
) -> dict | None:
    if required and stage >= ResearchStage.SENTENCE_JEPA and checkpoint_path is None:
        raise ValueError(
            f"{stage.name} requires --init-checkpoint from an admitted prior stage"
        )
    if checkpoint_path is None:
        return None
    if admission is not None:
        digest = hashlib.sha256()
        with checkpoint_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != admission["admitted_checkpoint_sha256"]:
            raise ValueError(
                "admission record does not bind the initialization checkpoint"
            )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if checkpoint.get("architecture") != HIERARCHICAL_LANGUAGE_ARCHITECTURE:
        raise ValueError("initial checkpoint uses an incompatible architecture")
    prior_stage = ResearchStage[checkpoint["stage"]]
    required_predecessor = {
        ResearchStage.SENTENCE_JEPA: ResearchStage.TOKEN_JEPA,
        ResearchStage.DYNAMIC_COMMUTATION: ResearchStage.SENTENCE_JEPA,
        ResearchStage.MACRO_ACTION: ResearchStage.DYNAMIC_COMMUTATION,
        ResearchStage.VALUE_DISTILLATION: ResearchStage.MACRO_ACTION,
        ResearchStage.CLOSED_LOOP_REANALYSIS:
            ResearchStage.VALUE_DISTILLATION,
    }.get(stage)
    if required_predecessor is not None and prior_stage != required_predecessor:
        raise ValueError(
            f"{stage.name} requires a {required_predecessor.name} checkpoint, "
            f"got {prior_stage.name}"
        )
    if required_predecessor is None and prior_stage >= stage:
        raise ValueError("initial checkpoint must come from an earlier stage")
    prior_config = checkpoint["config"]
    immutable = (
        "d_backbone", "vocab_size", "pad_id", "d_token", "d_sentence",
        "d_action", "d_task", "predictor_width", "token_layers",
        "sentence_layers", "n_heads", "token_context",
        "sentence_context", "max_span",
    )
    for name in immutable:
        if prior_config[name] != getattr(model.config, name):
            raise ValueError(f"checkpoint configuration mismatch: {name}")
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = []
    if model.pi1 is not None:
        allowed_missing.append("pi1.")
    if model.v is not None:
        allowed_missing.append("v.")
    unexpected_missing = [
        name for name in incompatible.missing_keys
        if not any(name.startswith(prefix) for prefix in allowed_missing)
    ]
    if incompatible.unexpected_keys or unexpected_missing:
        raise ValueError(
            "checkpoint parameters are incompatible: "
            f"missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    # Preserve trained Mahalanobis geometry without reloading the embedded
    # duplicate model state from learner.state_dict().
    learner_state = checkpoint.get("learner", {})
    geometry = {
        name: value for name, value in learner_state.items()
        if name.startswith(("token_metric.", "sentence_metric."))
    }
    learner.load_state_dict(geometry, strict=False)
    return checkpoint


def _configure_stage_trainability(
    model: HierarchicalLanguageJEPA, stage: ResearchStage
) -> list[str]:
    model.requires_grad_(False)
    active: list[tuple[str, torch.nn.Module | None]]
    if stage == ResearchStage.TOKEN_JEPA:
        active = [
            ("e0", model.e0), ("p0", model.p0),
            ("token_action", model.token_action),
        ]
    elif stage == ResearchStage.SENTENCE_JEPA:
        active = [
            ("e0_to_1", model.e0_to_1), ("a1", model.a1),
            ("p1", model.p1),
        ]
    elif stage == ResearchStage.DYNAMIC_COMMUTATION:
        active = [("e0_to_1", model.e0_to_1), ("p1", model.p1)]
    elif stage == ResearchStage.MACRO_ACTION:
        active = [
            ("p1", model.p1), ("pi1", model.pi1),
            ("task_projection", model.task_projection),
        ]
    elif stage == ResearchStage.VALUE_DISTILLATION:
        active = [
            ("v", model.v), ("task_projection", model.task_projection)
        ]
    elif stage == ResearchStage.CLOSED_LOOP_REANALYSIS:
        active = [
            ("e0", model.e0), ("p0", model.p0),
            ("token_action", model.token_action),
            ("e0_to_1", model.e0_to_1), ("a1", model.a1),
            ("p1", model.p1), ("pi1", model.pi1),
            ("v", model.v), ("task_projection", model.task_projection),
        ]
    else:
        active = []
    names = []
    for name, module in active:
        if module is not None:
            module.requires_grad_(True)
            names.append(name)
    return names


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    data = load_features(
        args.features,
        require_provenance=(
            args.epochs > 0 and not args.allow_unpinned_features
        ),
    )
    feature_header = torch.load(
        args.features, map_location="cpu", weights_only=True
    )
    experiment_config = None
    if args.experiment_config is not None:
        from omegaconf import OmegaConf
        if args.experiment_config.parent.name == "experiment":
            from hydra import compose, initialize_config_dir
            config_root = args.experiment_config.parent.parent.resolve()
            with initialize_config_dir(
                config_dir=str(config_root), version_base=None
            ):
                composed = compose(
                    config_name=f"experiment/{args.experiment_config.stem}"
                )
            experiment_config = OmegaConf.to_container(
                composed, resolve=True
            )
        else:
            experiment_config = OmegaConf.to_container(
                OmegaConf.load(args.experiment_config), resolve=True
            )
        configured_stage = experiment_config.get("research", {}).get("stage")
        if args.stage is not None and configured_stage is not None and (
            configured_stage != args.stage
        ):
            raise ValueError(
                f"CLI stage {args.stage} disagrees with experiment config "
                f"{configured_stage}"
            )
    configured_stage = (
        experiment_config.get("research", {}).get("stage")
        if experiment_config else None
    )
    stage = ResearchStage[
        args.stage or configured_stage or ResearchStage.TOKEN_JEPA.name
    ]
    evaluation_only = {
        ResearchStage.FLAT_ORACLE_TOKEN,
        ResearchStage.ORACLE_WAYPOINT,
        ResearchStage.ORACLE_HIGH_LEVEL,
        ResearchStage.FULL_HIERARCHY,
    }
    if stage in evaluation_only:
        raise ValueError(
            f"{stage.name} is evaluation-only; use the oracle evaluator "
            "instead of the trainer"
        )
    admission = _load_admission(
        args.admission, stage,
        required=stage >= ResearchStage.SENTENCE_JEPA,
    )
    if admission is not None:
        if feature_header.get("dataset_fingerprint") != admission[
            "dataset_fingerprint"
        ]:
            raise ValueError(
                "admission record does not bind the training dataset"
            )
    model_config = (
        experiment_config.get("model", {}).get("config", {})
        if experiment_config else {}
    )
    def configured(name: str, cli_value):
        return model_config.get(name, cli_value)
    vocab_size = configured("vocab_size", args.vocab_size)
    if vocab_size is None:
        raise ValueError("vocab size is required by CLI or experiment config")
    token_values = data["input_ids"]
    if bool((token_values < 0).any()) or bool((token_values >= vocab_size).any()):
        raise ValueError("token ID lies outside configured vocabulary")
    configured_backbone = model_config.get("d_backbone")
    if configured_backbone is not None and configured_backbone != (
        data["hidden_states"].shape[-1]
    ):
        raise ValueError("feature hidden width disagrees with experiment config")
    counterfactual = None
    if args.counterfactual_features is not None:
        raw = torch.load(
            args.counterfactual_features, map_location="cpu", weights_only=True
        )
        if (
            raw.get("model_id") != MODEL_ID
            or raw.get("model_revision") != MODEL_REVISION
            or raw.get("transformers_version") != TRANSFORMERS_VERSION
            or raw.get("dataset_fingerprint") != feature_header.get(
                "dataset_fingerprint"
            )
            or raw.get("input_fingerprint") != feature_header.get(
                "input_fingerprint"
            )
            or raw.get("shard_index") != feature_header.get("shard_index")
            or raw.get("num_shards") != feature_header.get("num_shards")
        ):
            raise ValueError("counterfactual provenance does not match backbone")
        accounting = raw.get("candidate_accounting")
        if not isinstance(accounting, dict) or (
            accounting.get("empty_or_failed_slots") != 0
        ):
            raise ValueError(
                "counterfactual shard must account for all eight candidate slots"
            )
        counterfactual = collate_counterfactual_records(raw["records"])
    value_replay = None
    if args.value_replay is not None:
        value_replay = torch.load(
            args.value_replay, map_location="cpu", weights_only=True
        )
    replay_payload = None
    if args.planner_replay is not None:
        replay_payload = torch.load(
            args.planner_replay, map_location="cpu", weights_only=True
        )
        if counterfactual is None and "counterfactual_records" in replay_payload:
            counterfactual = collate_counterfactual_records(
                replay_payload["counterfactual_records"]
            )
        if value_replay is None and "value_replay" in replay_payload:
            value_replay = replay_payload["value_replay"]
    if (
        stage >= ResearchStage.VALUE_DISTILLATION
        and args.epochs > 0
        and value_replay is None
    ):
        raise ValueError("value stages require --value-replay for training")
    if (
        stage >= ResearchStage.CLOSED_LOOP_REANALYSIS
        and args.planner_replay is None
    ):
        raise ValueError(
            "CLOSED_LOOP_REANALYSIS requires grounded --planner-replay"
        )
    config = HierarchicalLanguageJEPAConfig(
        d_backbone=data["hidden_states"].shape[-1],
        vocab_size=vocab_size,
        pad_id=configured("pad_id", args.pad_id),
        d_token=configured("d_token", args.d_token),
        d_sentence=configured("d_sentence", args.d_sentence),
        d_action=configured("d_action", args.d_action),
        d_task=configured("d_task", 256),
        predictor_width=configured("predictor_width", args.predictor_width),
        token_layers=configured("token_layers", args.token_layers),
        sentence_layers=configured("sentence_layers", args.sentence_layers),
        n_heads=configured("n_heads", args.heads),
        token_context=configured("token_context", args.token_context),
        sentence_context=configured(
            "sentence_context", args.sentence_context
        ),
        cache_dropout=configured("cache_dropout", args.cache_dropout),
        max_span=configured("max_span", args.max_span),
        enable_macro_actions=stage >= ResearchStage.MACRO_ACTION,
        enable_value=stage >= ResearchStage.VALUE_DISTILLATION,
    )
    model = HierarchicalLanguageJEPA(config).to(args.device)
    loss_config = experiment_config.get("loss", {}) if experiment_config else {}
    weights = DenseLossWeights(**{
        name: loss_config[name]
        for name in DenseLossWeights.__dataclass_fields__
        if name in loss_config
    })
    learner = HierarchicalLanguageLearner(
        model, stage, weights=weights
    ).to(args.device)
    initialized_from = _initialize_from_checkpoint(
        model, learner, args.init_checkpoint, stage,
        required=stage >= ResearchStage.SENTENCE_JEPA,
        admission=admission,
    )
    checkpoint_sha256 = (
        sha256_file(args.init_checkpoint)
        if args.init_checkpoint is not None else ""
    )
    if value_replay is not None:
        required_value_provenance = {
            "architecture": HIERARCHICAL_LANGUAGE_ARCHITECTURE,
            "model_revision": MODEL_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "dataset_fingerprint": feature_header.get("dataset_fingerprint"),
            "source_checkpoint_sha256": checkpoint_sha256,
        }
        if any(
            value_replay.get(name) != value
            for name, value in required_value_provenance.items()
        ):
            raise ValueError("value replay provenance is incompatible")
    if stage == ResearchStage.CLOSED_LOOP_REANALYSIS:
        validity = (
            experiment_config.get("validity_gates", {})
            if experiment_config else {}
        )
        _validate_grounded_planner_replay(
            replay_payload,
            dataset_fingerprint=feature_header["dataset_fingerprint"],
            checkpoint_sha256=checkpoint_sha256,
            require_exact_worker_achievement=bool(
                validity.get("require_exact_worker_achievement", True)
            ),
        )
    trainable_modules = _configure_stage_trainability(model, stage)
    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = (
        torch.optim.AdamW(trainable_parameters, lr=args.learning_rate)
        if trainable_parameters else None
    )
    main_count = len(data["hidden_states"])
    history = []
    compute = ComputeLedger()
    rollout_config = (
        experiment_config.get("rollout", {}) if experiment_config else {}
    )
    rollout_schedule = SparseRolloutSchedule(**{
        name: rollout_config[name]
        for name in SparseRolloutSchedule.__dataclass_fields__
        if name in rollout_config
    })
    rollout_generator = torch.Generator().manual_seed(args.seed + 17)
    for epoch in range(args.epochs):
        order = (
            torch.randperm(main_count)
            if stage != ResearchStage.VALUE_DISTILLATION
            else torch.empty(0, dtype=torch.long)
        )
        totals: dict[str, float] = {}
        examples = 0
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            batch = {
                name: value[index].to(args.device)
                for name, value in data.items()
            }
            batch["hidden_states"] = batch["hidden_states"].to(
                dtype=next(model.parameters()).dtype
            )
            with compute.measure(
                "dense_training_wall", estimated_flops=0.0,
                items=len(index),
            ):
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                loss, losses = learner(
                    batch["hidden_states"], batch["input_ids"],
                    batch["boundaries"],
                    attention_mask=batch.get("attention_mask"),
                    prompt_len=batch.get("prompt_len"),
                    solution_end=batch.get("solution_end"),
                )
                if stage != ResearchStage.DATA_VALIDATION:
                    assert optimizer is not None
                    loss.backward()
                    optimizer.step()
                    model.update_targets(args.ema_momentum)
            _record_dense_flops(compute, model, stage, batch)
            examples += len(index)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * len(index)
        replay_totals: dict[str, float] = {}
        replay_examples = 0
        replay_skipped_batches = 0
        if counterfactual is not None:
            replay_count = len(counterfactual["token_ids"])
            replay_order = torch.randperm(
                replay_count, generator=rollout_generator
            )
            for replay_start in range(0, replay_count, args.replay_batch_size):
                replay_index = replay_order[
                    replay_start:replay_start + args.replay_batch_size
                ]
                branch = {
                    name: value[replay_index].to(args.device)
                    for name, value in counterfactual.items()
                    if isinstance(value, torch.Tensor)
                }
                for name in (
                    "root_hidden", "suffix_hidden",
                    "root_token_history_hidden",
                    "root_sentence_history_hidden",
                ):
                    branch[name] = branch[name].to(
                        dtype=next(model.parameters()).dtype
                    )
                with compute.measure(
                    "counterfactual_replay_wall",
                    estimated_flops=0.0,
                    items=len(branch["lengths"]),
                ):
                    assert optimizer is not None
                    optimizer.zero_grad(set_to_none=True)
                    rollout_horizon = rollout_schedule.sample_horizon(
                        rollout_generator
                    )
                    cf_loss, cf_losses = learner.counterfactual_loss(
                        branch["root_hidden"], branch["suffix_hidden"],
                        branch["token_ids"], branch["lengths"],
                        sentence_eligible=branch["sentence_eligible"],
                        rollout_horizon=rollout_horizon,
                        rollout_truncate_bptt=(
                            rollout_schedule.truncate_bptt(rollout_horizon)
                        ),
                        root_token_history_hidden=branch[
                            "root_token_history_hidden"
                        ],
                        root_token_history_action_ids=branch[
                            "root_token_history_action_ids"
                        ],
                        root_token_history_lengths=branch[
                            "root_token_history_lengths"
                        ],
                        root_sentence_history_hidden=branch[
                            "root_sentence_history_hidden"
                        ],
                        root_sentence_history_lengths=branch[
                            "root_sentence_history_lengths"
                        ],
                        root_sentence_history_span_ids=branch[
                            "root_sentence_history_span_ids"
                        ],
                        root_sentence_history_span_mask=branch[
                            "root_sentence_history_span_mask"
                        ],
                    )
                    updated = _step_if_trainable(
                        cf_loss, optimizer, model, args.ema_momentum
                    )
                    replay_skipped_batches += int(not updated)
                _record_counterfactual_flops(
                    compute, model, branch, rollout_horizon
                )
                replay_examples += len(branch["lengths"])
                for name, value in cf_losses.items():
                    replay_totals[name] = replay_totals.get(
                        name, 0.0
                    ) + float(value.detach()) * len(branch["lengths"])
        value_total = 0.0
        value_roots = 0
        if value_replay is not None:
            replay = {
                name: (
                    value.to(args.device)
                    if isinstance(value, torch.Tensor) else value
                )
                for name, value in value_replay.items()
            }
            replay["task_hidden"] = replay["task_hidden"].to(
                dtype=next(model.parameters()).dtype
            )
            if "task_hidden" not in replay:
                raise ValueError(
                    "value replay must store frozen task_hidden=H(P), not a "
                    "stale projected task_embedding"
                )
            required_shapes = {
                "successor_state": replay["teacher_cost"].shape
                + (config.d_sentence,),
                "successor_context": replay["teacher_cost"].shape
                + (config.predictor_width,),
                "first_action_log_probability": replay["teacher_cost"].shape,
                "action_mask": replay["teacher_cost"].shape,
            }
            for name, shape in required_shapes.items():
                if replay[name].shape != shape:
                    raise ValueError(f"value replay {name} has invalid shape")
            if replay["task_hidden"].shape != (
                replay["teacher_cost"].shape[0], config.d_backbone
            ):
                raise ValueError("value replay task_hidden has invalid shape")
            tensor_values = [
                value for value in replay.values()
                if isinstance(value, torch.Tensor) and value.is_floating_point()
            ]
            if any(not torch.isfinite(value).all() for value in tensor_values):
                raise ValueError("value replay contains non-finite values")
            root_count = len(replay["teacher_cost"])
            value_order = torch.randperm(
                root_count, generator=rollout_generator
            ).to(replay["teacher_cost"].device)
            for value_start in range(
                0, root_count, args.replay_batch_size
            ):
                index = value_order[
                    value_start:value_start + args.replay_batch_size
                ]
                task_embedding = model.task_projection(
                    replay["task_hidden"][index]
                )
                value_items = int(replay["action_mask"][index].sum())
                with compute.measure(
                    "value_distillation_wall",
                    estimated_flops=0.0,
                    items=value_items,
                ):
                    assert optimizer is not None
                    optimizer.zero_grad(set_to_none=True)
                    value_loss = value_distillation_loss(
                        model,
                        replay["successor_state"][index],
                        replay["successor_context"][index],
                        task_embedding,
                        replay["first_action_log_probability"][index],
                        replay["teacher_cost"][index],
                        replay["action_mask"][index],
                        step_cost=float(replay.get("step_cost", 0.0)),
                        prior_weight=float(replay.get("prior_weight", 1.0)),
                        teacher_temperature=float(
                            replay.get("teacher_temperature", 1.0)
                        ),
                        value_temperature=float(
                            replay.get("value_temperature", 1.0)
                        ),
                    )
                    value_loss.backward()
                    optimizer.step()
                    model.update_targets(args.ema_momentum)
                for component, module in (
                    ("value_task_projection", model.task_projection),
                    ("value_head", model.v),
                ):
                    compute.add(
                        component,
                        estimated_flops=training_flops(
                            parameter_count(module, trainable_only=True),
                            value_items,
                        ),
                        calls=0,
                        items=value_items,
                    )
                batch_roots = len(index)
                value_total += float(value_loss.detach()) * batch_roots
                value_roots += batch_roots
        row = {"epoch": epoch}
        if examples:
            row.update({
                name: value / examples for name, value in totals.items()
            })
        if replay_examples:
            row.update({
                name: value / replay_examples
                for name, value in replay_totals.items()
            })
            row["counterfactual_skipped_batches"] = replay_skipped_batches
        if value_roots:
            row["value_distillation"] = value_total / value_roots
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "architecture": HIERARCHICAL_LANGUAGE_ARCHITECTURE,
        "model": model.state_dict(),
        "learner": learner.state_dict(),
        "config": config.__dict__,
        "stage": stage.name,
        "experiment_config": experiment_config,
        "initialized_from": (
            str(args.init_checkpoint) if initialized_from is not None else None
        ),
        "admission": admission,
        "trainable_modules": trainable_modules,
        "frozen_modules": [
            name for name in (
                "e0", "p0", "token_action", "e0_to_1", "a1", "p1",
                "pi1", "v", "task_projection",
            )
            if name not in trainable_modules
        ],
    }, args.output / "model.pt")
    (args.output / "metrics.json").write_text(
        json.dumps({
            "history": history,
            "compute": compute.summary(),
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
