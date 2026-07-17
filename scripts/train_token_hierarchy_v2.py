"""Train the multilevel causal token-to-span JEPA."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.lm import LMDataset, collate_lm
from textjepa.data.sampling import FreshEpochSampler
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA
from textjepa.planning.token_hierarchy import macro_codes
from textjepa.training.loggers import MetricLogger
from textjepa.training.optim import build_optimizer, cosine_warmup, ema_momentum
from textjepa.utils import seed_everything
from textjepa.utils.metrics import effective_rank, feature_std


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1)


def normalized_mse(pred, target, mask):
    pred = F.layer_norm(pred, pred.shape[-1:])
    target = F.layer_norm(target, target.shape[-1:])
    return masked_mean((pred - target).square().mean(-1), mask.float())


def dense_loss(predictions, targets, masks, discount):
    losses, weights = [], []
    for horizon, (prediction, target, mask) in enumerate(
        zip(predictions, targets, masks)
    ):
        weight = discount ** horizon
        losses.append(weight * normalized_mse(prediction, target, mask))
        weights.append(weight)
    return torch.stack(losses).sum() / sum(weights)


def vicreg(states, covariance):
    centered = states - states.mean(0)
    std = torch.sqrt(centered.var(0, unbiased=False) + 1e-4)
    variance = F.relu(1.0 - std).mean()
    cov = centered.T @ centered / max(states.shape[0] - 1, 1)
    cov = cov - torch.diag_embed(torch.diagonal(cov))
    return variance + covariance * cov.square().sum() / states.shape[-1]


def geometric_rank_loss(energy, distance, margin, label_gap):
    """Same-state hinge: lower energy for the EMA outcome nearer the goal."""
    delta = distance.unsqueeze(2) - distance.unsqueeze(1)
    energy_delta = energy.unsqueeze(2) - energy.unsqueeze(1)
    better = delta < -float(label_gap)
    loss = F.relu(float(margin) + energy_delta)
    return masked_mean(loss, better.float())


def geometric_preference_loss(energy, distance, objective, margin, label_gap,
                              temperature):
    """Non-symbolic objectives whose targets are EMA geometric distances."""
    if objective == "pairwise":
        return geometric_rank_loss(energy, distance, margin, label_gap)
    if objective == "listwise":
        teacher = (-distance / float(temperature)).softmax(-1)
        student = (-energy / float(temperature)).log_softmax(-1)
        return -(teacher * student).sum(-1).mean()
    if objective == "regression":
        return F.mse_loss(energy, distance)
    raise ValueError(f"unknown geometric preference objective: {objective}")


def geometric_rank_metrics(energy, distance, label_gap):
    delta = distance.unsqueeze(2) - distance.unsqueeze(1)
    energy_delta = energy.unsqueeze(2) - energy.unsqueeze(1)
    valid = delta.abs() > float(label_gap)
    correct = (energy_delta.sign() == delta.sign()) & valid
    pair = correct.sum() / valid.sum().clamp_min(1)
    chosen = energy.argmin(1)
    selected = distance.gather(1, chosen[:, None]).squeeze(1)
    best = distance.min(1).values
    return pair, (selected <= best + 1e-7).float().mean(), (selected - best).mean()


@torch.no_grad()
def counterfactual_distances(
    model, batch_tokens, prompt_len, root_chunks, continuation_chunks,
    anchor_tokens, goal, level_index,
):
    """EMA-encode same-prefix root/continuation chunks and score geometry.

    ``root_chunks`` is [B,C,L]. ``continuation_chunks`` is [B,M,S,L] or
    ``None``, where M continuation proposals each contain S future chunks.
    Every alternative is appended to the *same* factual prefix;
    chunks borrowed from another batch row are proposals only, never outcome
    labels. This is the token analogue of intent GAR's environment-rendered
    counterfactual next states.
    """
    device = batch_tokens.device
    batch, candidates, width = root_chunks.shape
    continuations = 1 if continuation_chunks is None else continuation_chunks.shape[1]
    rows, owners, actual_lengths = [], [], []
    for b in range(batch):
        prefix_end = int(prompt_len[b]) + int(anchor_tokens)
        prefix = batch_tokens[b, :prefix_end]
        for c in range(candidates):
            for m in range(continuations):
                pieces = [prefix, root_chunks[b, c]]
                if continuation_chunks is not None:
                    pieces.extend(continuation_chunks[b, m].unbind(0))
                sequence = torch.cat(pieces)
                rows.append(sequence)
                owners.append(b)
                actual_lengths.append(len(sequence))
    maximum = max(actual_lengths)
    packed = batch_tokens.new_full((len(rows), maximum), model.pad_id)
    for i, sequence in enumerate(rows):
        packed[i, :len(sequence)] = sequence
    encoded = model.teacher(packed)
    continuation_steps = (
        0 if continuation_chunks is None else continuation_chunks.shape[2]
    )
    reasoning_length = anchor_tokens + width * (1 + continuation_steps) + 1
    base_paths = []
    for i, b in enumerate(owners):
        start = int(prompt_len[b]) - 1
        base_paths.append(encoded[i, start:start + reasoning_length])
    base_paths = torch.stack(base_paths)
    if level_index is None:
        outcome = base_paths[:, -1]
    else:
        outcome = model.lift_state_path(
            base_paths, through_level=level_index, teacher=True
        )[level_index][:, -1]
    owner = torch.tensor(owners, device=device, dtype=torch.long)
    distance = (
        F.layer_norm(outcome, outcome.shape[-1:])
        - F.layer_norm(goal.index_select(0, owner), goal.shape[-1:])
    ).abs().mean(-1)
    return distance.reshape(batch, candidates, continuations).amin(-1)


def primitive_candidates(factual, k, vocabulary_size):
    """Observed token plus K non-symbolic full-vocabulary alternatives."""
    alternatives = torch.randint(
        1, int(vocabulary_size), (len(factual), int(k)),
        device=factual.device,
    )
    # Avoid a silent factual duplicate without introducing any semantic rule.
    duplicate = alternatives.eq(factual[:, None])
    alternatives[duplicate] = alternatives[duplicate].remainder(
        int(vocabulary_size) - 1
    ).add(1)
    return torch.cat([factual[:, None], alternatives], 1)


def macro_chunk_candidates(level, anchor, k, mode="global", conditional_k=32):
    """Factual chunk plus observed, optionally state-conditioned proposals."""
    factual = level["raw_action_ids"][:, anchor]
    pool = level["raw_action_ids"][level["valid"]]
    if mode == "global":
        ids = torch.randint(len(pool), (len(factual), int(k)), device=factual.device)
    elif mode == "conditional":
        pool_states = level["prev"][level["valid"]].detach()
        roots = level["prev"][:, anchor].detach()
        neighbours = torch.cdist(roots, pool_states).topk(
            min(int(conditional_k), len(pool)), largest=False
        ).indices
        sampled = torch.randint(
            neighbours.shape[1], (len(factual), int(k)), device=factual.device
        )
        ids = neighbours.gather(1, sampled)
    else:
        raise ValueError(f"unknown macro proposal mode: {mode}")
    return torch.cat([factual[:, None], pool[ids]], 1)


def end_to_end_geometric_preferences(model, batch, out, cfg):
    """Intent-style GAR at the primitive and every macro hierarchy level."""
    obj = cfg.objective
    if float(obj.geo_rank_low) == 0 and float(obj.geo_rank_high) == 0:
        zero = out["low_pred"].sum() * 0
        return zero, {}, zero.detach()
    k = int(obj.geo_rank_k)
    horizon = int(obj.geo_rank_horizon)
    if horizon not in (1, 2, 4):
        raise ValueError("end-to-end GAR supports horizons 1, 2, or 4")
    tokens = batch["tokens"].to(out["low_pred"].device)
    prompt_len = batch["prompt_len"].to(tokens.device)
    items, total, selection = {}, out["low_pred"].sum() * 0, out["low_pred"].sum() * 0

    if float(obj.geo_rank_low) > 0:
        available = int(out["valid"].sum(1).min())
        anchor = torch.randint(max(1, available - (horizon - 1)), ()).item()
        root_ids = primitive_candidates(
            out["action_ids"][:, anchor], k, model.token_action.num_embeddings
        )
        root_chunks = root_ids.unsqueeze(-1)
        continuation = None
        if horizon > 1:
            continuation = torch.stack([
                primitive_candidates(
                    out["action_ids"][:, anchor + step],
                    int(obj.geo_rank_continuations) - 1,
                    model.token_action.num_embeddings,
                )
                for step in range(1, horizon)
            ], 2).unsqueeze(-1)
        goal = out["final_target"].detach()
        distance = counterfactual_distances(
            model, tokens, prompt_len, root_chunks, continuation,
            anchor, goal, None,
        )
        baseline = (
            out["prompt_target"] if anchor == 0 else out["target"][:, anchor - 1]
        )
        baseline_distance = (
            F.layer_norm(baseline, baseline.shape[-1:])
            - F.layer_norm(goal, goal.shape[-1:])
        ).abs().mean(-1)
        # A cost advantage: negative means that the candidate moved closer to
        # the terminal goal.  Pairwise ordering is unchanged, while the MSE
        # term now calibrates the magnitude of progress across states.
        target = distance - baseline_distance[:, None]
        batch_size, candidates = root_ids.shape
        histories = out["prev"][:, :anchor + 1].repeat_interleave(candidates, 0)
        previous = model.token_action(out["action_ids"][:, :anchor])
        previous = previous.repeat_interleave(candidates, 0)
        actions = model.token_action(root_ids.reshape(-1))[:, None]
        action_history = torch.cat([previous, actions], 1)
        predicted = model.low_predictor(histories, action_history)[:, -1]
        if bool(obj.geo_rank_detach_prediction):
            predicted = predicted.detach()
        goal_rows = goal.repeat_interleave(candidates, 0)
        energy = model.low_goal_value(predicted, goal_rows).reshape(
            batch_size, candidates
        )
        rank = geometric_preference_loss(
            energy, target, obj.geo_rank_objective,
            obj.geo_rank_margin, obj.geo_rank_label_gap,
            obj.geo_rank_temperature,
        )
        regression = F.mse_loss(energy, target)
        low_total = rank + float(obj.geo_rank_regression) * regression
        total = total + float(obj.geo_rank_low) * low_total
        selection = selection + float(obj.geo_rank_low) * low_total.detach()
        pair, top1, regret = geometric_rank_metrics(
            energy.detach(), target, obj.geo_rank_label_gap
        )
        items.update({
            "geo_low_rank": rank, "geo_low_regression": regression,
            "geo_low_pair": pair, "geo_low_top1": top1,
            "geo_low_regret": regret,
        })

    level_weights = list(obj.geo_rank_level_weights)
    if len(level_weights) == 1:
        level_weights *= len(out["levels"])
    if len(level_weights) != len(out["levels"]):
        raise ValueError("geo_rank_level_weights must broadcast or match levels")
    if float(obj.geo_rank_high) > 0:
        for level, configured_weight in zip(out["levels"], level_weights):
            level_weight = float(configured_weight)
            if level_weight == 0:
                continue
            available = int(level["valid"].sum(1).min())
            if available < horizon:
                continue
            anchor = torch.randint(max(1, available - (horizon - 1)), ()).item()
            roots = macro_chunk_candidates(
                level, anchor, k, obj.geo_rank_macro_proposals,
                obj.geo_rank_conditional_k,
            )
            continuation = None
            if horizon > 1:
                continuation = torch.stack([
                    macro_chunk_candidates(
                        level, anchor + step,
                        int(obj.geo_rank_continuations) - 1,
                        obj.geo_rank_macro_proposals,
                        obj.geo_rank_conditional_k,
                    )
                    for step in range(1, horizon)
                ], 2)
            counts = level["valid"].sum(1).long() - 1
            goal = level["target"][
                torch.arange(len(tokens), device=tokens.device), counts
            ].detach()
            distance = counterfactual_distances(
                model, tokens, prompt_len, roots, continuation,
                anchor * int(level["span"]), goal, int(level["index"]),
            )
            baseline = level["teacher_prev"][:, anchor]
            baseline_distance = (
                F.layer_norm(baseline, baseline.shape[-1:])
                - F.layer_norm(goal, goal.shape[-1:])
            ).abs().mean(-1)
            target = distance - baseline_distance[:, None]
            batch_size, candidates = roots.shape[:2]
            flat_roots = roots.reshape(-1, roots.shape[-1])
            codes = macro_codes(
                model, flat_roots, through_level=int(level["index"])
            )[int(level["index"])][:, 0]
            state_history = level["prev"][:, :anchor + 1].repeat_interleave(
                candidates, 0
            )
            previous = level["codes"][:, :anchor].repeat_interleave(candidates, 0)
            action_history = torch.cat([previous, codes[:, None]], 1)
            predicted = model.levels[int(level["index"])].predictor(
                state_history, action_history
            )[:, -1]
            if bool(obj.geo_rank_detach_prediction):
                predicted = predicted.detach()
            goal_rows = goal.repeat_interleave(candidates, 0)
            energy = model.levels[int(level["index"])].goal_value(
                predicted, goal_rows
            ).reshape(batch_size, candidates)
            rank = geometric_preference_loss(
                energy, target, obj.geo_rank_objective,
                obj.geo_rank_margin, obj.geo_rank_label_gap,
                obj.geo_rank_temperature,
            )
            regression = F.mse_loss(energy, target)
            level_total = rank + float(obj.geo_rank_regression) * regression
            weight = float(obj.geo_rank_high) * level_weight
            total = total + weight * level_total
            selection = selection + weight * level_total.detach()
            pair, top1, regret = geometric_rank_metrics(
                energy.detach(), target, obj.geo_rank_label_gap
            )
            prefix = f"geo_level{int(level['index']) + 1}"
            items.update({
                f"{prefix}_rank": rank, f"{prefix}_regression": regression,
                f"{prefix}_pair": pair, f"{prefix}_top1": top1,
                f"{prefix}_regret": regret,
            })
    return total, items, selection


def compute_losses(out, cfg, model=None, batch=None):
    obj = cfg.objective
    low = normalized_mse(out["low_pred"], out["target"], out["valid"])
    low_dense = dense_loss(
        out["low_dense_predictions"], out["low_dense_targets"],
        out["low_dense_masks"], obj.dense_discount,
    )
    low_value = masked_mean(
        F.smooth_l1_loss(
            out["low_value"], out["low_remaining_target"], reduction="none"
        ), out["valid"].float(),
    )
    goal = F.smooth_l1_loss(
        F.layer_norm(out["goal_pred"], out["goal_pred"].shape[-1:]),
        F.layer_norm(out["final_target"].detach(), out["final_target"].shape[-1:]),
    )
    # Regularize the online causal states. ``target`` comes from the EMA
    # teacher under no_grad, so applying VICReg there changes the scalar loss
    # without producing any encoder gradient.
    state_flat = out["prev"].reshape(-1, out["prev"].shape[-1])[
        out["valid"].reshape(-1)
    ]
    regularizer = vicreg(state_flat, obj.covariance)
    total = (
        obj.low_prediction * low
        + obj.low_dense * low_dense
        + obj.low_value * low_value
        + obj.goal_prediction * goal
        + obj.vicreg * regularizer
    )
    items = {
        "low_prediction": low,
        "low_dense": low_dense,
        "low_value": low_value,
        "goal_prediction": goal,
        "vicreg": regularizer,
    }
    selection = (
        obj.low_prediction * low.detach()
        + obj.low_dense * low_dense.detach()
        + obj.goal_prediction * goal.detach()
    )
    if out["token_prior_logits"] is not None:
        logits = out["token_prior_logits"][out["valid"]]
        labels = out["action_ids"][out["valid"]]
        token_prior = F.cross_entropy(
            logits,
            labels,
            label_smoothing=float(obj.token_prior_label_smoothing),
        )
        token_prior_accuracy = logits.argmax(-1).eq(labels).float().mean()
        token_prior_entropy = torch.distributions.Categorical(
            logits=logits
        ).entropy().mean()
        total = total + obj.token_prior * token_prior
        selection = selection + obj.token_prior * token_prior.detach()
        items.update({
            "token_prior": token_prior,
            "token_prior_accuracy": token_prior_accuracy,
            "token_prior_entropy": token_prior_entropy,
        })
        rollout_losses = []
        rollout_weights = []
        for horizon, rollout_logits in enumerate(
            out["token_prior_rollout_logits"], 1
        ):
            mask = out["valid"][:, horizon:]
            labels = out["action_ids"][:, horizon:]
            if not mask.any():
                continue
            loss_at_horizon = F.cross_entropy(
                rollout_logits[mask], labels[mask],
                label_smoothing=float(obj.token_prior_label_smoothing),
            )
            weight = float(obj.token_prior_rollout_discount) ** (horizon - 1)
            rollout_losses.append(weight * loss_at_horizon)
            rollout_weights.append(weight)
            items[f"token_prior_rollout_h{horizon}"] = loss_at_horizon
        if rollout_losses:
            token_prior_rollout = torch.stack(rollout_losses).sum() / sum(
                rollout_weights
            )
            total = total + obj.token_prior_rollout * token_prior_rollout
            selection = (
                selection
                + obj.token_prior_rollout * token_prior_rollout.detach()
            )
            items["token_prior_rollout"] = token_prior_rollout
    configured_level_weights = list(
        getattr(obj, "high_level_weights", [1.0])
    )
    if len(configured_level_weights) == 1:
        configured_level_weights *= len(out["levels"])
    if len(configured_level_weights) != len(out["levels"]):
        raise ValueError(
            "objective.high_level_weights must have length one or match "
            "the number of hierarchy levels"
        )
    for level, level_weight in zip(out["levels"], configured_level_weights):
        level_weight = float(level_weight)
        mask = level["valid"]
        high = normalized_mse(level["pred"], level["target"], mask)
        high_dense = dense_loss(
            level["dense_predictions"], level["dense_targets"],
            level["dense_masks"], obj.dense_discount,
        )
        reachability = normalized_mse(
            level["pred"], level["recursive_low_endpoint"].detach(), mask
        )
        value = masked_mean(F.smooth_l1_loss(
            level["value"], level["remaining_target"], reduction="none"
        ), mask.float())
        prior = masked_mean(level["prior_nll"], mask.float())
        support = masked_mean(
            F.softplus(-level["support_pos"]) + F.softplus(level["support_neg"]),
            mask.float(),
        )
        level_total = (
            obj.high_prediction * high
            + obj.high_dense * high_dense
            + obj.reachability * reachability
            + obj.high_value * value
            + obj.macro_prior * prior
            + obj.support * support
        )
        total = total + level_weight * level_total
        prefix = f"level{level['index'] + 1}"
        items.update({
            f"{prefix}_prediction": high,
            f"{prefix}_dense": high_dense,
            f"{prefix}_reachability": reachability,
            f"{prefix}_value": value,
            f"{prefix}_prior": prior,
            f"{prefix}_support": support,
            f"{prefix}_weight": high.new_tensor(level_weight),
        })
        selection = selection + level_weight * (
            obj.high_prediction * high.detach()
            + obj.high_dense * high_dense.detach()
            + obj.reachability * reachability.detach()
        )
    if model is not None and batch is not None:
        geo_total, geo_items, geo_selection = end_to_end_geometric_preferences(
            model, batch, out, cfg
        )
        total = total + geo_total
        selection = selection + geo_selection
        items.update(geo_items)
    items["loss"] = total
    items["selection"] = selection
    return total, items


def make_dataset(cfg, vocab, size, seed):
    return LMDataset(
        vocab, size=size, seed=seed, modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range), leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )


@hydra.main(config_path="../configs", config_name="token_hierarchy_v2", version_base="1.3")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    print(OmegaConf.to_yaml(cfg))
    vocab = build_vocab(cfg.data.modulus)
    train_ds = make_dataset(cfg, vocab, cfg.data.train_size, cfg.data.train_seed)
    val_ds = make_dataset(cfg, vocab, cfg.data.val_size, cfg.data.val_seed)
    sampler = FreshEpochSampler(train_ds, seed=cfg.seed)
    collate = partial(collate_lm, pad_id=vocab.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, sampler=sampler,
        num_workers=cfg.train.num_workers, collate_fn=collate, drop_last=True,
        persistent_workers=cfg.train.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        collate_fn=collate,
    )
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(cfg.device)
    optimizer = build_optimizer(model, cfg.train.lr, cfg.train.weight_decay)
    total_steps = cfg.train.epochs * len(train_loader)
    logger = MetricLogger(out_dir)
    step, best = 0, float("inf")
    for epoch in range(cfg.train.epochs):
        sampler.set_epoch(epoch)
        model.train()
        for batch in train_loader:
            for group in optimizer.param_groups:
                group["lr"] = cfg.train.lr * cosine_warmup(
                    step, total_steps, cfg.train.warmup_steps
                )
            out = model(batch["tokens"].to(cfg.device), batch["prompt_len"].to(cfg.device))
            loss, items = compute_losses(out, cfg, model=model, batch=batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            model.update_teacher(ema_momentum(
                step, total_steps, cfg.train.ema_start, cfg.train.ema_end
            ))
            if step % cfg.train.log_every == 0:
                logger.log(step, {k: float(v.detach()) for k, v in items.items()}, prefix="train/")
            step += 1
        model.eval()
        sums, count, state_features, level_features = {}, 0, [], None
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                if index >= cfg.train.eval_batches:
                    break
                out = model(batch["tokens"].to(cfg.device), batch["prompt_len"].to(cfg.device))
                _, items = compute_losses(out, cfg, model=model, batch=batch)
                for name, value in items.items():
                    sums[name] = sums.get(name, 0.0) + float(value)
                state_features.append(out["target"][out["valid"]])
                if level_features is None:
                    level_features = [[] for _ in out["levels"]]
                for i, level in enumerate(out["levels"]):
                    level_features[i].append(level["codes"][level["valid"]])
                count += 1
        metrics = {name: value / count for name, value in sums.items()}
        states = torch.cat(state_features)
        metrics.update(state_std=feature_std(states), state_rank=effective_rank(states[:4096]))
        for i, features in enumerate(level_features or []):
            codes = torch.cat(features)
            metrics[f"level{i + 1}_action_std"] = feature_std(codes)
            metrics[f"level{i + 1}_action_rank"] = effective_rank(codes[:4096])
        logger.log(step, metrics, prefix="val/")
        payload = {
            "model": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True),
            "epoch": epoch, "metrics": metrics,
        }
        torch.save(payload, out_dir / "last.pt")
        if metrics["selection"] < best:
            best = metrics["selection"]
            torch.save(payload, out_dir / "best.pt")
        print(f"[epoch {epoch}] " + "  ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items())), flush=True)
    logger.close()


if __name__ == "__main__":
    main()
