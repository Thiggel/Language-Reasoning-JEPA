"""Representation and drift audit for the non-symbolic faithful token editor.

Terminal-buffer geometry is intentionally reported only as a privileged
diagnostic: the true repaired buffer is unavailable to a deployed editor.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.data import DataLoader

from textjepa.utils.checkpoint import build_dataset, collate_for, load_run
from textjepa.utils.metrics import effective_rank, feature_std


OPERATION_NAMES = {0: "delete", 1: "insert", 2: "replace"}


def normalized_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-item L1 after independently normalizing each latent vector."""
    norm = lambda value: F.layer_norm(value, value.shape[-1:])
    return (norm(prediction) - norm(target)).abs().mean(-1)


def masked_mean(values, mask):
    return float(values[mask].mean()) if mask.any() else 0.0


def _summary(values: torch.Tensor, mask: torch.Tensor) -> dict:
    selected = values[mask]
    return {
        "ln_l1": float(selected.mean()) if selected.numel() else None,
        "steps": int(selected.numel()),
    }


def pad_concat_2d(tensors: list[torch.Tensor], fill: float | int | bool = 0):
    """Concatenate batches whose padded trajectory widths may differ."""
    width = max(tensor.shape[1] for tensor in tensors)
    padded = [
        F.pad(tensor, (0, width - tensor.shape[1]), value=fill)
        for tensor in tensors
    ]
    return torch.cat(padded)


def depth_summary(values: torch.Tensor, mask: torch.Tensor) -> dict[str, dict]:
    """Summarize open-loop error at each one-indexed trajectory depth."""
    return {
        str(depth + 1): _summary(values[:, depth], mask[:, depth])
        for depth in range(values.shape[1])
        if bool(mask[:, depth].any())
    }


def operation_summary(
    values: torch.Tensor, mask: torch.Tensor, operations: torch.Tensor
) -> dict[str, dict]:
    result = {}
    for label in sorted(int(value) for value in operations[mask].unique()):
        name = OPERATION_NAMES.get(label, f"unknown_{label}")
        result[name] = _summary(values, mask & operations.eq(label))
    return result


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else None


def _same_state_assignment_stats(pred, target, valid) -> dict | None:
    if pred is None or target is None or valid is None or pred.shape[2] < 2:
        return None
    candidates = pred.shape[2]
    eye = torch.eye(candidates, dtype=torch.bool, device=pred.device)
    pair_mask = (
        valid.unsqueeze(-1) & valid.unsqueeze(-2) & ~eye.view(1, 1, candidates, candidates)
    )
    target_pair = normalized_l1(target.unsqueeze(-2), target.unsqueeze(-3))
    prediction_pair = normalized_l1(pred.unsqueeze(-2), pred.unsqueeze(-3))

    # [B,T,prediction candidate,target candidate]
    assignment_error = normalized_l1(pred.unsqueeze(-2), target.unsqueeze(-3))
    assignment_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    masked_assignment = assignment_error.masked_fill(~assignment_valid, float("inf"))
    nearest = masked_assignment.argmin(-1)
    identity = torch.arange(candidates, device=pred.device).view(1, 1, candidates)
    accuracy_mask = valid & torch.isfinite(masked_assignment).any(-1)

    matched = assignment_error.diagonal(dim1=-2, dim2=-1)
    wrong_mask = pair_mask
    return {
        "target_pair_sum": float(target_pair[pair_mask].sum()),
        "prediction_pair_sum": float(prediction_pair[pair_mask].sum()),
        "pair_count": int(pair_mask.sum()),
        "matched_sum": float(matched[valid].sum()),
        "matched_count": int(valid.sum()),
        "wrong_sum": float(assignment_error[wrong_mask].sum()),
        "wrong_count": int(wrong_mask.sum()),
        "assignment_correct": int((nearest.eq(identity) & accuracy_mask).sum()),
        "assignment_count": int(accuracy_mask.sum()),
    }


def same_state_action_sensitivity(out) -> dict | None:
    """Test whether distinct actions produce correspondingly distinct states.

    Pointwise next-state loss can be minimized by predicting the local mean
    when every token edit moves a long-buffer embedding only slightly. Exact
    same-state alternatives let us measure this failure directly at both the
    global buffer and exact changed-step granularities.
    """
    global_stats = _same_state_assignment_stats(
        out.extras.get("cf_chunk_pred"),
        out.extras.get("cf_chunk_tgt"),
        out.extras.get("cf_valid"),
    )
    if global_stats is None:
        return None
    local_stats = _same_state_assignment_stats(
        out.extras.get("cf_slot_pred"),
        out.extras.get("cf_slot_tgt"),
        out.extras.get("cf_slot_valid"),
    )
    return {"global": global_stats, "local": local_stats}


def _combine_assignment_parts(parts: list[dict]) -> dict | None:
    if not parts:
        return None
    totals = {key: sum(part[key] for part in parts) for key in parts[0]}
    target_pair = totals["target_pair_sum"] / max(totals["pair_count"], 1)
    prediction_pair = totals["prediction_pair_sum"] / max(totals["pair_count"], 1)
    matched = totals["matched_sum"] / max(totals["matched_count"], 1)
    wrong = totals["wrong_sum"] / max(totals["wrong_count"], 1)
    return {
        "exact_target_pairwise_ln_l1": target_pair,
        "predicted_pairwise_ln_l1": prediction_pair,
        "prediction_to_target_separation_ratio": (
            prediction_pair / target_pair if target_pair else None
        ),
        "matched_assignment_ln_l1": matched,
        "wrong_assignment_ln_l1": wrong,
        "correct_over_wrong_assignment_margin": wrong - matched,
        "nearest_outcome_assignment_accuracy": (
            totals["assignment_correct"] / max(totals["assignment_count"], 1)
        ),
        "candidates": totals["assignment_count"],
        "interpretation": "same observed buffer; exact mechanically executed outcomes; no quality labels",
    }


def combine_action_sensitivity(parts: list[dict]) -> dict | None:
    if not parts:
        return None
    return {
        "global_buffer": _combine_assignment_parts([
            part["global"] for part in parts if part.get("global") is not None
        ]),
        "exact_changed_step": _combine_assignment_parts([
            part["local"] for part in parts if part.get("local") is not None
        ]),
    }


def probe(x, y, classification=False):
    split = max(1, int(0.7 * len(x)))
    if split >= len(x) or len(np.unique(y[:split])) < 2:
        return None
    model = (
        LogisticRegression(max_iter=500, class_weight="balanced")
        if classification else Ridge(alpha=1.0)
    )
    model.fit(x[:split], y[:split])
    prediction = model.predict(x[split:])
    return float(
        accuracy_score(y[split:], prediction)
        if classification else r2_score(y[split:], prediction)
    )


def shuffled_action_prediction(model, out, batch):
    """Predict with a deranged current action and each sample's true prefix.

    ``LatentDynamicsCore._predict_counterfactuals`` constructs an independent
    causal sequence for every (trajectory, step) pair.  Thus a replacement at
    step t cannot leak another candidate or alter the observed actions before
    t. Attention-over-buffer predictors use the same deranged action codes
    with fixed observed current-step embeddings, so only the current action
    changes in either predictor family.
    """
    valid = out.step_mask
    count = int(valid.sum())
    if count < 2:
        return None, "fewer_than_two_valid_actions", None
    core = getattr(model, "core", None)
    if core is None:
        return None, "independent_causal_prefix_api_unavailable", None
    alternatives = out.actions.detach().clone()
    observed = alternatives[valid]
    best, best_changed = None, None
    for shift in range(1, count):
        candidate = observed.roll(shift, 0)
        changed = ~torch.isclose(candidate, observed).all(-1)
        if best_changed is None or int(changed.sum()) > int(best_changed.sum()):
            best, best_changed = candidate, changed
    if not bool(best_changed.any()):
        return None, "fewer_than_two_distinct_action_codes", None
    alternatives[valid] = best
    changed_mask = valid.clone()
    changed_mask[valid] = best_changed
    if getattr(model, "token_pred", None) is not None:
        # Structured actions are a coupled (operation, pointer, content) tuple.
        # Derange the raw tuple while keeping each observed current token state
        # fixed; rebuilding only an already-compressed action code would bypass
        # the pointer-conditioned transition interface under test.
        indices = torch.arange(count, device=valid.device).roll(1)
        shuffled_batch = dict(batch)
        for key in ("op", "edit_position", "edit_content_token"):
            value = batch[key].clone()
            value[valid] = batch[key][valid][indices]
            shuffled_batch[key] = value
        shuffled_out = model(shuffled_batch)
        tuple_changed = (
            shuffled_batch["op"].ne(batch["op"])
            | shuffled_batch["edit_position"].ne(batch["edit_position"])
            | shuffled_batch["edit_content_token"].ne(batch["edit_content_token"])
        ) & valid
        return shuffled_out.preds, None, tuple_changed
    if getattr(model, "attn_pred", None) is not None:
        batch_size, states, chunks, length = batch["buffer_tokens"].shape
        steps = states - 1
        sentence_embeddings = model.encode_chunks(
            batch["buffer_tokens"][:, :-1].reshape(
                batch_size * steps, chunks, length
            )
        )
        sentence_mask = batch["buffer_mask"][:, :-1].reshape(
            batch_size * steps, chunks
        )
        prediction = model.attn_pred(
            sentence_embeddings,
            sentence_mask,
            out.prev_states.reshape(batch_size * steps, -1),
            alternatives.reshape(batch_size * steps, -1),
        ).reshape(batch_size, steps, -1)
        return prediction, None, changed_mask
    if not hasattr(core, "_predict_counterfactuals"):
        return None, "independent_causal_prefix_api_unavailable", None
    prediction = core._predict_counterfactuals(
        out.prev_states, out.actions, alternatives.unsqueeze(2), valid
    )
    return prediction[:, :, 0], None, changed_mask


def recursive_horizon_errors(
    model, out, horizons: tuple[int, ...], max_origins: int = 4
) -> dict:
    """Measure true h-step rollouts from evenly spaced observed prefixes.

    Causal-transformer rollout cost grows with both prefix and horizon.  The
    explicit origin cap keeps the post-training audit bounded; zero requests
    every feasible prefix.
    """
    result = {}
    batch, width = out.step_mask.shape
    if getattr(model, "token_pred", None) is not None:
        for horizon in sorted(set(horizons)):
            if 1 <= horizon <= width:
                mask = out.step_mask[:, :horizon].all(1)
                error = normalized_l1(
                    out.rollout[:, horizon - 1],
                    out.step_states_tgt[:, horizon - 1],
                )
                result[str(horizon)] = _summary(error, mask)
        return result
    for horizon in sorted(set(horizons)):
        if horizon < 1 or horizon > width:
            continue
        predictions, targets, masks = [], [], []
        n_origins = width - horizon + 1
        if max_origins and n_origins > max_origins:
            origin_indices = torch.linspace(
                0, n_origins - 1, max_origins
            ).round().long().unique().tolist()
        else:
            origin_indices = range(n_origins)
        for origin in origin_indices:
            codes = out.actions[:, origin:origin + horizon]
            if getattr(model.predictor, "causal_sequence", False):
                rollout = model.predictor.rollout(
                    out.prev_states[:, origin], codes,
                    state_history=out.prev_states[:, :origin + 1],
                    action_history=out.actions[:, :origin],
                )
            else:
                rollout = model.core._rollout(out.prev_states[:, origin], codes)
            predictions.append(rollout[:, -1])
            targets.append(out.step_states_tgt[:, origin + horizon - 1])
            masks.append(out.step_mask[:, origin:origin + horizon].all(1))
        error = normalized_l1(torch.stack(predictions, 1), torch.stack(targets, 1))
        mask = torch.stack(masks, 1)
        result[str(horizon)] = _summary(error, mask)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument(
        "--counterfactual-k", type=int,
        help="override evaluation alternatives per state, including for K=0 checkpoints",
    )
    parser.add_argument(
        "--corruption-mode",
        choices=("mixed", "mask", "replace", "remove"),
        help="override the validation corruption family for a matched cross-regime audit",
    )
    parser.add_argument(
        "--horizons", default="1,2,4,8",
        help="comma-separated recursive horizons",
    )
    parser.add_argument(
        "--max-horizon-origins", type=int, default=4,
        help="evenly spaced causal prefixes per horizon and batch; 0 uses all",
    )
    parser.add_argument("--out")
    args = parser.parse_args()
    horizons = tuple(int(value) for value in args.horizons.split(",") if value)
    if any(value < 1 for value in horizons):
        parser.error("--horizons must contain positive integers")
    if args.max_horizon_origins < 0:
        parser.error("--max-horizon-origins must be nonnegative")

    model, vocab, cfg = load_run(args.ckpt, args.device)
    if args.counterfactual_k is not None:
        if args.counterfactual_k < 0:
            parser.error("--counterfactual-k must be nonnegative")
        cfg.data.counterfactual_k = args.counterfactual_k
        cfg.data.counterfactual_source = "uniform_local"
    if args.corruption_mode is not None:
        cfg.data.corruption_mode = args.corruption_mode
    dataset = build_dataset(cfg, vocab, "val", size=args.examples)
    loader = DataLoader(
        dataset, batch_size=min(16, cfg.train.batch_size),
        collate_fn=partial(collate_for(cfg), pad_id=vocab.pad_id),
    )
    direct, recursive, persistence = [], [], []
    masks, all_ops, states, remaining, ops, delta = [], [], [], [], [], []
    horizon_totals: dict[str, list[dict]] = {}
    high, ldad_correct, ldad_total, goal_distance, goal_remaining = [], 0, 0, [], []
    shuffled, shuffled_masks, shuffle_unavailable = [], [], set()
    counterfactual_errors = []
    action_sensitivity_parts = []
    recursive_available = getattr(model, "attn_pred", None) is None
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(args.device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            out = model(batch)
            mask = out.step_mask
            direct_error = normalized_l1(out.preds, out.step_states_tgt)
            direct.append(direct_error.cpu())
            if recursive_available:
                recursive_error = normalized_l1(out.rollout, out.step_states_tgt)
                recursive.append(recursive_error.cpu())
            persistence.append(normalized_l1(out.prev_states, out.step_states_tgt).cpu())
            masks.append(mask.cpu())
            all_ops.append(batch["op"].cpu())
            states.append(out.step_states[mask].cpu())
            remaining.append(batch["remaining"][mask].cpu())
            ops.append(batch["op"][mask].cpu())
            delta.append((out.step_states - out.prev_states)[mask].cpu())
            if "cf_chunk_pred" in out.extras:
                cf_valid = out.extras["cf_valid"]
                cf_error = normalized_l1(
                    out.extras["cf_chunk_pred"], out.extras["cf_chunk_tgt"]
                )
                counterfactual_errors.append(cf_error[cf_valid].cpu())
                sensitivity = same_state_action_sensitivity(out)
                if sensitivity is not None:
                    action_sensitivity_parts.append(sensitivity)

            if recursive_available:
                for horizon, summary in recursive_horizon_errors(
                    model, out, horizons, args.max_horizon_origins
                ).items():
                    horizon_totals.setdefault(horizon, []).append(summary)

            shuffled_prediction, reason, changed_mask = shuffled_action_prediction(
                model, out, batch
            )
            if shuffled_prediction is None:
                shuffle_unavailable.add(reason)
            else:
                shuffled.append(normalized_l1(shuffled_prediction, out.step_states_tgt).cpu())
                shuffled_masks.append(changed_mask.cpu())

            last = mask.sum(1).clamp_min(1) - 1
            goal = out.step_states_tgt[torch.arange(mask.shape[0], device=mask.device), last]
            distance = normalized_l1(out.step_states_tgt, goal.unsqueeze(1))
            goal_distance.append(distance[mask].cpu())
            goal_remaining.append(batch["remaining"][mask].cpu())
            if out.hi_preds is not None:
                high.append(normalized_l1(out.hi_preds, out.hi_targets)[out.hi_mask].cpu())
            logits = out.extras.get("observed_action_logits")
            if logits is not None:
                length = min(logits.shape[-2], batch["action_tokens"].shape[-1])
                target = batch["action_tokens"][..., :length]
                valid = mask.unsqueeze(-1) & target.ne(vocab.pad_id)
                ldad_correct += int(logits[..., :length, :].argmax(-1)[valid].eq(target[valid]).sum())
                ldad_total += int(valid.sum())

    direct_error = pad_concat_2d(direct)
    recursive_error = pad_concat_2d(recursive) if recursive else None
    persistence_error = pad_concat_2d(persistence)
    step_mask = pad_concat_2d(masks, False)
    operation = pad_concat_2d(all_ops, -1)
    state = torch.cat(states)
    rem = torch.cat(remaining).numpy()
    op = torch.cat(ops).numpy()
    displacement = torch.cat(delta).numpy()
    geometry = torch.cat(goal_distance).numpy()
    geometry_remaining = torch.cat(goal_remaining).numpy()

    # Combine batch summaries without biasing toward a short final batch.
    horizon_payload = {}
    for horizon, summaries in horizon_totals.items():
        count = sum(item["steps"] for item in summaries)
        horizon_payload[horizon] = {
            "ln_l1": (
                sum(item["ln_l1"] * item["steps"] for item in summaries if item["ln_l1"] is not None)
                / count if count else None
            ),
            "steps": count,
        }

    matched_mean = _summary(direct_error, step_mask)["ln_l1"]
    if shuffled:
        shuffled_error = pad_concat_2d(shuffled)
        shuffled_mask = pad_concat_2d(shuffled_masks, False)
        shuffled_summary = _summary(shuffled_error, shuffled_mask)
        shuffled_control = {
            **shuffled_summary,
            "over_matched_error_ratio": (
                shuffled_summary["ln_l1"] / matched_mean
                if matched_mean and shuffled_summary["ln_l1"] is not None else None
            ),
            "construction": "current action deranged across valid steps; unchanged duplicate codes excluded; independent true causal prefix",
        }
    else:
        shuffled_control = {
            "ln_l1": None, "steps": 0, "over_matched_error_ratio": None,
            "unavailable_reasons": sorted(reason for reason in shuffle_unavailable if reason),
        }

    privileged_geometry = {
        "uses_privileged_true_terminal_buffer": True,
        "available_at_deployment": False,
        "purpose": "diagnostic_only_not_a_planning_metric",
        "distance_remaining_edit_correlation": safe_correlation(
            geometry, geometry_remaining
        ),
    }
    payload = {
        "examples": len(dataset),
        "evaluation_corruption_mode": dataset.corruption_mode,
        "one_step_ln_l1": matched_mean,
        "recursive_ln_l1": (
            _summary(recursive_error, step_mask)["ln_l1"]
            if recursive_error is not None else None
        ),
        "counterfactual_one_step_ln_l1": (
            float(torch.cat(counterfactual_errors).mean())
            if counterfactual_errors else None
        ),
        "counterfactual_candidates": (
            int(sum(values.numel() for values in counterfactual_errors))
        ),
        "counterfactual_contract": {
            "mechanically_executed_exact_outcomes": True,
            "target_relative_quality_labels_used": False,
            "candidate_tokens_drawn_from_current_buffer_only": True,
        },
        "same_state_action_sensitivity": combine_action_sensitivity(
            action_sensitivity_parts
        ),
        "persistence_no_change_ln_l1": _summary(persistence_error, step_mask)["ln_l1"],
        "shuffled_action_causal_falsifier": shuffled_control,
        "recursive_ln_l1_by_horizon": horizon_payload,
        "recursive_horizon_origin_sampling": (
            {
                "method": "evenly_spaced_observed_causal_prefixes",
                "maximum_origins_per_horizon_per_batch": args.max_horizon_origins,
                "zero_means_all": True,
            }
            if recursive_available else {
                "unavailable_reason": (
                    "attention-over-buffer predictor has no recursive rollout API; "
                    "the dormant core predictor is not a valid substitute"
                )
            }
        ),
        "open_loop_ln_l1_by_trajectory_depth": (
            depth_summary(recursive_error, step_mask)
            if recursive_error is not None else {}
        ),
        "open_loop_ln_l1_by_operation": (
            operation_summary(recursive_error, step_mask, operation)
            if recursive_error is not None else {}
        ),
        "macro_ln_l1": float(torch.cat(high).mean()) if high else None,
        "state_std": float(feature_std(state)),
        "state_effective_rank": float(effective_rank(state[:4096])),
        "remaining_edit_probe_r2": probe(state.numpy(), rem),
        "operation_from_displacement_accuracy": probe(displacement, op, True),
        "privileged_terminal_goal_diagnostics": privileged_geometry,
        # Backward-compatible key, now explicitly duplicated under the labelled block.
        "terminal_geometry_remaining_correlation": privileged_geometry[
            "distance_remaining_edit_correlation"
        ],
        "ldad_token_accuracy": ldad_correct / max(ldad_total, 1),
        "symbolic_reasoning_labels_used": False,
    }
    destination = Path(args.out or Path(args.ckpt).parent / "token_edit_audit.json")
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
