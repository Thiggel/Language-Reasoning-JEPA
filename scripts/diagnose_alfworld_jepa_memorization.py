"""Localize ALFWorld JEPA training-game failure without retraining.

This is a candidate-privileged diagnostic, not a deployment evaluation.  It
separates:

1. executed-transition latent fit;
2. observed-counterfactual transition fit and three-way GAR ordering;
3. teacher-forced full-catalogue expert ranking.

The existing closed-loop train metrics supply the fourth stage.  For GAR we
report both the historical single-token alternative path used by ``_geo_rank``
and the full-prefix path used by the causal planner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from statistics import mean

import torch
import torch.nn.functional as F

from textjepa.data.igsm.dataset import collate
from textjepa.data.observed_action import ObservedActionDataset
from textjepa.planning.catalogue import CatalogueLatentPlanner
from textjepa.utils.checkpoint import build_dataset, load_run


def _ln_l1(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (
        F.layer_norm(left, left.shape[-1:])
        - F.layer_norm(right, right.shape[-1:])
    ).abs().mean(-1)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _rate(values: list[bool]) -> float:
    return sum(values) / max(len(values), 1)


def _pairwise_accuracy(energy: torch.Tensor, label: torch.Tensor,
                       gap: float = 0.02) -> tuple[int, int]:
    """Return correct and eligible unordered pairs."""
    correct = total = 0
    for first in range(energy.numel()):
        for second in range(first + 1, energy.numel()):
            delta = float(label[first] - label[second])
            if abs(delta) <= gap:
                continue
            total += 1
            predicted = float(energy[first] - energy[second])
            correct += int((predicted < 0) == (delta < 0))
    return correct, total


def _summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()) if len(values) else 0.0,
        "median": float(tensor.median()) if len(values) else 0.0,
        "count": len(values),
    }


def _selected_counterfactual_actions(
    episode, transition, seed: int, k: int,
) -> set[str]:
    """Reproduce the dataset's deterministic per-anchor alternative sample."""
    alternatives = list(transition.counterfactuals)
    random.Random(f"{seed}:{episode.episode_id}:geo").shuffle(alternatives)
    return {value.action for value in alternatives[:max(0, int(k))]}


@torch.no_grad()
def _target_state(model, planner: CatalogueLatentPlanner, prompt,
                  outcomes: list[str]) -> torch.Tensor:
    prompt_tokens = planner._chunks(prompt)
    prompt_mask = torch.ones(
        1, len(prompt), dtype=torch.bool, device=planner.device
    )
    step_tokens = planner._chunks(outcomes)
    step_mask = torch.ones(
        1, len(outcomes), dtype=torch.bool, device=planner.device
    )
    _, states = model.encode_states(
        prompt_tokens, prompt_mask, step_tokens, step_mask, teacher=True
    )
    return states[0, len(outcomes) - 1]


@torch.no_grad()
def diagnose(checkpoint: Path, device_name: str, split: str,
             episodes: int) -> dict:
    model, vocab, cfg = load_run(checkpoint, device_name)
    model.eval()
    device = torch.device(device_name)
    source = build_dataset(cfg, vocab, split)
    if not hasattr(source, "episodes"):
        raise ValueError("diagnostic requires a recorded observed-action dataset")
    selected = source.episodes[:episodes]
    plain = ObservedActionDataset(selected, vocab, geo_rank_k=0)
    planner = CatalogueLatentPlanner(
        model, vocab, device, simulation_depth=1, proposal_top_m=1,
        beam_width=1,
    )

    executed_error: list[float] = []
    executed_persistence: list[float] = []
    executed_retrieval: list[bool] = []
    counterfactual_error: list[float] = []
    counterfactual_trained_error: list[float] = []
    counterfactual_heldout_error: list[float] = []
    counterfactual_legacy_error: list[float] = []
    counterfactual_persistence: list[float] = []
    gar_full_top1: list[bool] = []
    gar_legacy_top1: list[bool] = []
    gar_full_pairs = [0, 0]
    gar_legacy_pairs = [0, 0]
    configured_full_top1: list[bool] = []
    configured_model_top1: list[bool] = []
    configured_full_pairs = [0, 0]
    configured_model_pairs = [0, 0]
    full_expert_top1: list[bool] = []
    full_top1_feasible: list[bool] = []
    full_expert_ranks: list[float] = []
    full_expert_percentiles: list[float] = []
    catalogue_sizes: list[float] = []
    state_count = counterfactual_count = 0

    for episode_index, episode in enumerate(selected):
        batch = _to_device(
            collate([plain[episode_index]], pad_id=vocab.pad_id), device
        )
        out = model(batch)
        length = len(episode.transitions)
        targets = out.step_states_tgt[0, :length]
        predictions = out.preds[0, :length]
        previous = out.prev_states[0, :length]
        actions = out.actions[0, :length]
        target_distance = _ln_l1(
            predictions.unsqueeze(1), targets.unsqueeze(0)
        )
        for step in range(length):
            executed_error.append(float(target_distance[step, step]))
            executed_persistence.append(float(_ln_l1(
                previous[step], targets[step]
            )))
            executed_retrieval.append(
                int(target_distance[step].argmin()) == step
            )

            transition = episode.transitions[step]
            catalogue = tuple(transition.catalogue)
            expert_index = catalogue.index(transition.action)
            codes = planner._action_codes(catalogue)
            prefix_states = out.prev_states[:, :step + 1].expand(
                len(catalogue), -1, -1
            )
            prefix_actions = actions[:step + 1].unsqueeze(0).expand(
                len(catalogue), -1, -1
            ).clone()
            prefix_actions[:, step] = codes
            full_predictions = model.core.predictor(
                prefix_states, prefix_actions
            )[:, step]
            full_energy = model.value_head(
                full_predictions,
                out.s0.expand(len(catalogue), -1),
            )
            order = full_energy.argsort()
            rank = int((order == expert_index).nonzero()[0]) + 1
            selected_action = catalogue[int(order[0])]
            full_expert_top1.append(rank == 1)
            full_top1_feasible.append(selected_action in transition.available)
            full_expert_ranks.append(float(rank))
            full_expert_percentiles.append(
                (rank - 1) / max(len(catalogue) - 1, 1)
            )
            catalogue_sizes.append(float(len(catalogue)))
            state_count += 1

            alternatives = list(transition.counterfactuals)
            if not alternatives:
                continue
            selected_actions = _selected_counterfactual_actions(
                episode,
                transition,
                int(cfg.seed),
                int(cfg.data.get("geo_rank_k", 2)),
            )
            alt_codes = planner._action_codes(
                tuple(value.action for value in alternatives)
            )
            count = len(alternatives)
            cf_states = out.prev_states[:, :step + 1].expand(
                count, -1, -1
            )
            cf_actions = actions[:step + 1].unsqueeze(0).expand(
                count, -1, -1
            ).clone()
            cf_actions[:, step] = alt_codes
            cf_predictions = model.core.predictor(
                cf_states, cf_actions
            )[:, step]
            # Historical GAR path: alternatives are evaluated as length-one
            # sequences, unlike the executed action and the planner.
            legacy_predictions = model.core.predictor(
                previous[step].expand(count, -1), alt_codes
            )
            true_cf = torch.stack([
                _target_state(
                    model, planner, episode.prompt,
                    [
                        value.outcome
                        for value in episode.transitions[:step]
                    ] + [alternative.outcome],
                )
                for alternative in alternatives
            ])
            alternative_error = _ln_l1(
                cf_predictions, true_cf
            ).cpu().tolist()
            counterfactual_error.extend(alternative_error)
            for alternative, error in zip(alternatives, alternative_error):
                target = (
                    counterfactual_trained_error
                    if alternative.action in selected_actions
                    else counterfactual_heldout_error
                )
                target.append(error)
            counterfactual_legacy_error.extend(
                _ln_l1(legacy_predictions, true_cf).cpu().tolist()
            )
            counterfactual_persistence.extend(
                _ln_l1(previous[step].expand_as(true_cf), true_cf)
                .cpu().tolist()
            )

            candidates_full = torch.cat([
                predictions[step:step + 1], cf_predictions
            ])
            candidates_legacy = torch.cat([
                predictions[step:step + 1], legacy_predictions
            ])
            candidate_targets = torch.cat([
                targets[step:step + 1], true_cf
            ])
            goal = targets[length - 1]
            labels = _ln_l1(candidate_targets, goal.expand_as(candidate_targets))
            energy_full = model.value_head(
                candidates_full, out.s0.expand(count + 1, -1)
            )
            energy_legacy = model.value_head(
                candidates_legacy, out.s0.expand(count + 1, -1)
            )
            best_label = int(labels.argmin())
            gar_full_top1.append(int(energy_full.argmin()) == best_label)
            gar_legacy_top1.append(int(energy_legacy.argmin()) == best_label)
            good, total = _pairwise_accuracy(energy_full, labels)
            gar_full_pairs[0] += good
            gar_full_pairs[1] += total
            good, total = _pairwise_accuracy(energy_legacy, labels)
            gar_legacy_pairs[0] += good
            gar_legacy_pairs[1] += total
            counterfactual_count += count

    # Repeat every dense GAR anchor exactly as configured so this block tests
    # fit to the actual horizon labels, not the one-step geometry above.
    dense = ObservedActionDataset(
        selected,
        vocab,
        geo_rank_k=int(cfg.data.get("geo_rank_k", 2)),
        geo_rank_horizon=int(cfg.data.get("geo_rank_horizon", 1)),
        dense_geo_anchors=True,
        seed=int(cfg.seed),
    )
    for index in range(len(dense)):
        batch = _to_device(collate([dense[index]], pad_id=vocab.pad_id), device)
        out = model(batch)
        # Recorded domains can contain deterministic episodes with no
        # alternative action at any prefix.  They remain factual-transition
        # examples but contribute no configured GAR label.
        if "ga_valid" not in out.extras:
            continue
        valid = out.extras["ga_valid"][0]
        label = out.extras["ga_label"][0, valid]
        configured_energy = out.extras["ga_energy"][0, valid]
        step = int(batch["ga_t"][0])
        alt_valid = batch["ga_valid"][0]
        alt_tokens = batch["ga_alt_action_tokens"][0, alt_valid]
        alt_codes = model.encode_actions(alt_tokens.unsqueeze(1)).squeeze(1)
        count = alt_codes.shape[0]
        prefix_states = out.prev_states[:, :step + 1].expand(
            count, -1, -1
        )
        prefix_actions = out.actions[:, :step + 1].expand(
            count, -1, -1
        ).clone()
        prefix_actions[:, step] = alt_codes
        alt_predictions = model.core.predictor(
            prefix_states, prefix_actions
        )[:, step]
        full_predictions = torch.cat([
            out.preds[0, step:step + 1], alt_predictions
        ])
        full_energy = model.value_head(
            full_predictions, out.s0.expand(count + 1, -1)
        )
        # Every valid row begins with the executed candidate.
        configured_full_top1.append(
            int(full_energy.argmin()) == int(label.argmin())
        )
        configured_model_top1.append(
            int(configured_energy.argmin()) == int(label.argmin())
        )
        good, total = _pairwise_accuracy(full_energy, label)
        configured_full_pairs[0] += good
        configured_full_pairs[1] += total
        good, total = _pairwise_accuracy(configured_energy, label)
        configured_model_pairs[0] += good
        configured_model_pairs[1] += total

    return {
        "checkpoint": str(checkpoint),
        "split": split,
        "episodes": len(selected),
        "states": state_count,
        "counterfactuals": counterfactual_count,
        "information_boundary": (
            "candidate-privileged training-state diagnostic; stored expert, "
            "counterfactual, and feasibility labels are used for localization"
        ),
        "executed_transition": {
            "latent_ln_l1": _summary(executed_error),
            "persistence_ln_l1": _summary(executed_persistence),
            "within_episode_target_retrieval_top1": _rate(executed_retrieval),
        },
        "observed_counterfactual_transition": {
            "planner_consistent_full_prefix_ln_l1": _summary(
                counterfactual_error
            ),
            "training_selected_ln_l1": _summary(
                counterfactual_trained_error
            ),
            "heldout_stored_ln_l1": _summary(
                counterfactual_heldout_error
            ),
            "legacy_gar_single_token_ln_l1": _summary(
                counterfactual_legacy_error
            ),
            "persistence_ln_l1": _summary(counterfactual_persistence),
        },
        "seen_one_step_geometry": {
            "planner_consistent_top1": _rate(gar_full_top1),
            "legacy_training_path_top1": _rate(gar_legacy_top1),
            "planner_consistent_pair_accuracy": (
                gar_full_pairs[0] / max(gar_full_pairs[1], 1)
            ),
            "legacy_training_path_pair_accuracy": (
                gar_legacy_pairs[0] / max(gar_legacy_pairs[1], 1)
            ),
            "eligible_pairs": gar_full_pairs[1],
        },
        "configured_horizon_gar_labels": {
            "label_construction": (
                "candidate rollout target includes the factual outcomes "
                "strictly before the action anchor"
            ),
            "planner_consistent_top1": _rate(configured_full_top1),
            "model_forward_top1": _rate(configured_model_top1),
            "planner_consistent_pair_accuracy": (
                configured_full_pairs[0] / max(configured_full_pairs[1], 1)
            ),
            "model_forward_pair_accuracy": (
                configured_model_pairs[0] / max(configured_model_pairs[1], 1)
            ),
            "eligible_pairs": configured_full_pairs[1],
        },
        "teacher_forced_full_catalogue": {
            "expert_top1": _rate(full_expert_top1),
            "selected_action_feasible": _rate(full_top1_feasible),
            "expert_rank": _summary(full_expert_ranks),
            "expert_rank_percentile": _summary(full_expert_percentiles),
            "catalogue_size": _summary(catalogue_sizes),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        checkpoint.parents[1].name: diagnose(
            checkpoint, args.device, args.split, args.episodes
        )
        for checkpoint in args.checkpoint
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
