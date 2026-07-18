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


def gar_candidate_stats(out) -> dict | None:
    """Held-out same-state ranking for expert plus mechanical alternatives."""
    expert_prediction = out.extras.get("gar_action_value")
    alt_prediction = out.extras.get("gar_alt_action_value")
    if expert_prediction is None or alt_prediction is None:
        return None
    prediction = torch.cat([expert_prediction.unsqueeze(-1), alt_prediction], -1)
    target = torch.cat([
        out.extras["gar_action_target"].unsqueeze(-1),
        out.extras["gar_alt_action_target"],
    ], -1)
    valid = torch.cat([
        out.step_mask.unsqueeze(-1), out.extras["gar_alt_action_valid"]
    ], -1)
    pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    target_difference = target.unsqueeze(-1) - target.unsqueeze(-2)
    prediction_difference = prediction.unsqueeze(-1) - prediction.unsqueeze(-2)
    pair_valid &= target_difference.abs().gt(1e-8)
    pair_correct = (
        target_difference.sign().eq(prediction_difference.sign()) & pair_valid
    )
    masked_target = target.masked_fill(~valid, -float("inf"))
    masked_prediction = prediction.masked_fill(~valid, -float("inf"))
    state_valid = valid.sum(-1).ge(2)
    return {
        "absolute_error_sum": float((prediction - target).abs()[valid].sum()),
        "candidate_count": int(valid.sum()),
        "pair_correct": int(pair_correct.sum()),
        "pair_count": int(pair_valid.sum()),
        "top1_correct": int((
            masked_prediction.argmax(-1).eq(masked_target.argmax(-1)) & state_valid
        ).sum()),
        "state_count": int(state_valid.sum()),
        "prediction": prediction[valid].detach().cpu(),
        "target": target[valid].detach().cpu(),
    }


def combine_gar_candidate_stats(parts: list[dict]) -> dict | None:
    if not parts:
        return None
    prediction = torch.cat([part["prediction"] for part in parts]).numpy()
    target = torch.cat([part["target"] for part in parts]).numpy()
    candidate_count = sum(part["candidate_count"] for part in parts)
    pair_count = sum(part["pair_count"] for part in parts)
    state_count = sum(part["state_count"] for part in parts)
    return {
        "candidate_privileged_training_diagnostic": True,
        "goal_available_to_learned_head": False,
        "mean_absolute_error": sum(
            part["absolute_error_sum"] for part in parts
        ) / max(candidate_count, 1),
        "pairwise_ranking_accuracy": sum(
            part["pair_correct"] for part in parts
        ) / max(pair_count, 1),
        "top1_action_accuracy": sum(
            part["top1_correct"] for part in parts
        ) / max(state_count, 1),
        "prediction_target_correlation": safe_correlation(prediction, target),
        "candidates": candidate_count,
        "states": state_count,
    }


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


def shuffled_action_prediction(model, out, batch=None, return_output: bool = False):
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
    if (
        getattr(model, "token_pred", None) is None
        and getattr(model, "attn_pred", None) is None
        and not hasattr(core, "_predict_counterfactuals")
    ):
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
        if batch is None:
            return None, "structured_action_batch_unavailable", None
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
        return (
            shuffled_out if return_output else shuffled_out.preds,
            None,
            tuple_changed,
        )
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


def structured_component_prediction(model, batch, valid, key: str):
    """Change one raw action component while holding the other two fixed."""
    eligible = valid.clone()
    if key == "edit_content_token":
        eligible &= batch["op"].ne(0)  # deletion has no content argument
    observed = batch[key][eligible]
    if observed.numel() < 2:
        return None, None, None
    best, best_changed = None, None
    for shift in range(1, observed.numel()):
        candidate = observed.roll(shift, 0)
        changed = candidate.ne(observed)
        if best_changed is None or int(changed.sum()) > int(best_changed.sum()):
            best, best_changed = candidate, changed
    if not bool(best_changed.any()):
        return None, None, None
    changed_batch = dict(batch)
    values = batch[key].clone()
    values[eligible] = best
    changed_batch[key] = values
    changed_mask = torch.zeros_like(valid)
    changed_mask[eligible] = best_changed
    return model(changed_batch), changed_mask, changed_batch


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
    parser.add_argument(
        "--component-falsifiers", action="store_true",
        help="separately derange operation, current-buffer pointer, and content",
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
    token_direct, token_shuffled, token_masks, token_shuffle_masks = [], [], [], []
    token_recursive, token_recursive_masks = [], []
    component_parts = {
        key: {
            "matched": [], "shuffled": [],
            "matched_mask": [], "shuffled_mask": [],
            "matched_local_mask": [], "shuffled_local_mask": [],
        }
        for key in ("op", "edit_position", "edit_content_token")
    }
    masks, all_ops, states, remaining, ops, delta = [], [], [], [], [], []
    horizon_totals: dict[str, list[dict]] = {}
    high, ldad_correct, ldad_total, goal_distance, goal_remaining = [], 0, 0, [], []
    shuffled, shuffled_masks, shuffle_unavailable = [], [], set()
    counterfactual_errors = []
    action_sensitivity_parts = []
    gar_candidate_parts = []
    recursive_available = getattr(model, "attn_pred", None) is None
    with torch.no_grad():
        for batch in loader:
            batch = {
                key: value.to(args.device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            out = model(batch)
            gar_stats = gar_candidate_stats(out)
            if gar_stats is not None:
                gar_candidate_parts.append(gar_stats)
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

            if args.component_falsifiers and getattr(model, "token_pred", None) is not None:
                for key, parts in component_parts.items():
                    component_out, component_changed, component_batch = structured_component_prediction(
                        model, batch, mask, key
                    )
                    if component_out is None:
                        continue
                    target_tokens = out.extras["token_targets"]
                    parts["matched"].append(normalized_l1(
                        out.extras["token_predictions"], target_tokens
                    ).cpu())
                    parts["shuffled"].append(normalized_l1(
                        component_out.extras["token_predictions"],
                        component_out.extras["token_targets"],
                    ).cpu())
                    matched_mask = (
                        out.extras["token_prediction_mask"]
                        & out.extras["token_target_mask"]
                        & component_changed.unsqueeze(-1)
                    )
                    shuffled_mask_component = (
                        component_out.extras["token_prediction_mask"]
                        & component_out.extras["token_target_mask"]
                        & component_changed.unsqueeze(-1)
                    )
                    width = matched_mask.shape[-1]
                    coordinate = torch.arange(width, device=mask.device).view(1, 1, -1)
                    original_position = batch["edit_position"].unsqueeze(-1)
                    changed_position = component_batch["edit_position"].unsqueeze(-1)
                    local = (
                        coordinate.sub(original_position).abs().le(2)
                        | coordinate.sub(changed_position).abs().le(2)
                    )
                    parts["matched_mask"].append(matched_mask.cpu())
                    parts["shuffled_mask"].append(shuffled_mask_component.cpu())
                    parts["matched_local_mask"].append((matched_mask & local).cpu())
                    parts["shuffled_local_mask"].append((
                        shuffled_mask_component & local
                    ).cpu())

            if recursive_available:
                for horizon, summary in recursive_horizon_errors(
                    model, out, horizons, args.max_horizon_origins
                ).items():
                    horizon_totals.setdefault(horizon, []).append(summary)

            shuffled_result, reason, changed_mask = shuffled_action_prediction(
                model, out, batch,
                return_output=getattr(model, "token_pred", None) is not None,
            )
            if shuffled_result is None:
                shuffle_unavailable.add(reason)
            else:
                shuffled_prediction = (
                    shuffled_result.preds
                    if getattr(model, "token_pred", None) is not None
                    else shuffled_result
                )
                shuffled.append(normalized_l1(shuffled_prediction, out.step_states_tgt).cpu())
                shuffled_masks.append(changed_mask.cpu())
                if getattr(model, "token_pred", None) is not None:
                    target_tokens = out.extras["token_targets"]
                    recursive_token_mask = (
                        out.extras["token_rollout_mask"]
                        & out.extras["token_target_mask"]
                        & out.step_mask.unsqueeze(-1)
                    )
                    token_recursive.append(normalized_l1(
                        out.extras["token_rollout_predictions"], target_tokens
                    ).cpu())
                    token_recursive_masks.append(recursive_token_mask.cpu())
                    direct_token_mask = (
                        out.extras["token_prediction_mask"]
                        & out.extras["token_target_mask"]
                        & changed_mask.unsqueeze(-1)
                    )
                    shuffled_token_mask = (
                        shuffled_result.extras["token_prediction_mask"]
                        & shuffled_result.extras["token_target_mask"]
                        & changed_mask.unsqueeze(-1)
                    )
                    token_direct.append(normalized_l1(
                        out.extras["token_predictions"], target_tokens
                    ).cpu())
                    token_shuffled.append(normalized_l1(
                        shuffled_result.extras["token_predictions"],
                        shuffled_result.extras["token_targets"],
                    ).cpu())
                    token_masks.append(direct_token_mask.cpu())
                    token_shuffle_masks.append(shuffled_token_mask.cpu())

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

    token_causal_control = None
    if token_direct:
        direct_token_error = torch.cat([
            values[mask] for values, mask in zip(token_direct, token_masks)
        ])
        shuffled_token_error = torch.cat([
            values[mask] for values, mask in zip(token_shuffled, token_shuffle_masks)
        ])
        direct_token_mean = float(direct_token_error.mean())
        shuffled_token_mean = float(shuffled_token_error.mean())
        token_causal_control = {
            "matched_ln_l1": direct_token_mean,
            "shuffled_ln_l1": shuffled_token_mean,
            "over_matched_error_ratio": (
                shuffled_token_mean / direct_token_mean
                if direct_token_mean else None
            ),
            "matched_tokens": int(direct_token_error.numel()),
            "shuffled_tokens": int(shuffled_token_error.numel()),
            "recursive_matched_ln_l1": (
                float(torch.cat([
                    values[mask] for values, mask in zip(
                        token_recursive, token_recursive_masks
                    )
                ]).mean()) if token_recursive else None
            ),
            "recursive_matched_tokens": int(sum(
                mask.sum().item() for mask in token_recursive_masks
            )),
            "construction": (
                "token-aligned EMA targets; current structured action tuple "
                "deranged; only valid predicted and target tokens scored"
            ),
        }

    component_controls = None
    if args.component_falsifiers:
        component_controls = {}
        for key, parts in component_parts.items():
            if not parts["matched"]:
                component_controls[key] = {"available": False}
                continue
            matched = torch.cat([
                values[mask] for values, mask in zip(
                    parts["matched"], parts["matched_mask"]
                )
            ])
            perturbed = torch.cat([
                values[mask] for values, mask in zip(
                    parts["shuffled"], parts["shuffled_mask"]
                )
            ])
            matched_mean_component = float(matched.mean())
            perturbed_mean = float(perturbed.mean())
            matched_local = torch.cat([
                values[mask] for values, mask in zip(
                    parts["matched"], parts["matched_local_mask"]
                )
            ])
            perturbed_local = torch.cat([
                values[mask] for values, mask in zip(
                    parts["shuffled"], parts["shuffled_local_mask"]
                )
            ])
            matched_local_mean = float(matched_local.mean())
            perturbed_local_mean = float(perturbed_local.mean())
            component_controls[key] = {
                "available": True,
                "matched_ln_l1": matched_mean_component,
                "shuffled_ln_l1": perturbed_mean,
                "over_matched_error_ratio": (
                    perturbed_mean / matched_mean_component
                    if matched_mean_component else None
                ),
                "matched_tokens": int(matched.numel()),
                "shuffled_tokens": int(perturbed.numel()),
                "local_radius_tokens": 2,
                "local_matched_ln_l1": matched_local_mean,
                "local_shuffled_ln_l1": perturbed_local_mean,
                "local_over_matched_error_ratio": (
                    perturbed_local_mean / matched_local_mean
                    if matched_local_mean else None
                ),
                "local_matched_tokens": int(matched_local.numel()),
                "local_shuffled_tokens": int(perturbed_local.numel()),
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
        "gar_candidate_ranking": combine_gar_candidate_stats(
            gar_candidate_parts
        ),
        "persistence_no_change_ln_l1": _summary(persistence_error, step_mask)["ln_l1"],
        "shuffled_action_causal_falsifier": shuffled_control,
        "token_aligned_shuffled_action_falsifier": token_causal_control,
        "token_aligned_component_falsifiers": component_controls,
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
