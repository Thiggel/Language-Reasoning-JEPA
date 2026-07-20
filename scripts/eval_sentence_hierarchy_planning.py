"""Non-symbolic two-level planning with categorical/continuous CEM.

The high level plans only in learned macro-action support. The low level then
generates a sentence token by token toward the first predicted waypoint. After
execution, the actual generated prefix is re-encoded and the complete high
plan is rebuilt; imagined waypoints are never carried across MPC steps.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.semantic_lm import SemanticBoundaryLMDataset, collate_semantic_lm
from textjepa.models.sentence_hierarchy import SentenceHierarchyJEPA


def normalized_distance(left, right):
    return F.mse_loss(
        F.layer_norm(left, left.shape[-1:]),
        F.layer_norm(right, right.shape[-1:]), reduction="none",
    ).mean(-1)


@torch.no_grad()
def encode_macro_sentences(model, sentence_ids, pad_id):
    width = max(map(len, sentence_ids))
    ids = torch.full(
        (len(sentence_ids), width), pad_id, dtype=torch.long,
        device=next(model.parameters()).device,
    )
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for row, sentence in enumerate(sentence_ids):
        ids[row, :len(sentence)] = torch.tensor(sentence, device=ids.device)
        valid[row, :len(sentence)] = True
    return model.macro_action.encoder(model.token_action(ids), valid)


@torch.no_grad()
def build_codebook(model, dataset, examples, pad_id):
    sentences, seen = [], set()
    for index in range(min(examples, len(dataset))):
        item = dataset[index]
        start = 0
        reasoning = item["tokens"][item["prompt_len"]:]
        for end in item["sentence_ends"]:
            sentence = tuple(reasoning[start:end])
            start = end
            if sentence and sentence not in seen:
                seen.add(sentence)
                sentences.append(list(sentence))
    return encode_macro_sentences(model, sentences, pad_id), sentences


@torch.no_grad()
def encode_generated(model, prefix, prompt_length, period_id):
    device = next(model.parameters()).device
    tokens = torch.tensor(prefix, device=device).unsqueeze(0)
    low = model.encoder(tokens)
    high = model.high_encoder(tokens)
    prompt_low = low[:, prompt_length - 1]
    prompt_high = high[:, prompt_length - 1]
    reasoning = prefix[prompt_length:]
    completed = [index + 1 for index, token in enumerate(reasoning) if token == period_id]
    # A failed low-level execution may end without punctuation. It is still
    # the actual state from which the next MPC step must replan, so represent
    # that partial span as an executed macro action instead of silently
    # reverting to the preceding imagined boundary.
    if reasoning and (not completed or completed[-1] != len(reasoning)):
        completed.append(len(reasoning))
    boundary_positions = [prompt_length - 1] + [
        prompt_length + end - 1 for end in completed
    ]
    high_history = high[:, boundary_positions]
    sentence_ids, start = [], 0
    for end in completed:
        sentence_ids.append(reasoning[start:end])
        start = end
    if sentence_ids:
        action_history = encode_macro_sentences(model, sentence_ids, model.pad_id).unsqueeze(0)
    else:
        action_history = high.new_zeros(1, 0, model.d_macro)
    low_history = low[:, prompt_length - 1:]
    if reasoning:
        ids = torch.tensor(reasoning, device=device).unsqueeze(0)
        low_actions = model.token_action(ids)
    else:
        low_actions = low.new_zeros(1, 0, model.token_action.embedding_dim)
    return {
        "tokens": tokens, "low_history": low_history,
        "low_action_history": low_actions, "low_state": low[:, -1],
        "high_history": high_history, "high_action_history": action_history,
        "high_state": high[:, -1], "prompt_low": prompt_low,
        "prompt_high": prompt_high,
        "high_goal": model.high_goal_head(prompt_high),
    }


@torch.no_grad()
def rollout_score(model, context, codes, goal_weight, value_weight,
                  prior_weight, support_weight, reachability_weight):
    count, horizon, _ = codes.shape
    predicted = model.high_predictor.rollout(
        context["high_state"].expand(count, -1), codes,
        state_history=context["high_history"].expand(count, -1, -1),
        action_history=context["high_action_history"].expand(count, -1, -1),
    )
    states = torch.cat([
        context["high_state"].expand(count, 1, -1), predicted[:, :-1]
    ], 1)
    prompt = context["prompt_high"].expand(count, horizon, -1)
    advantage = model.macro_value(states, prompt, codes)
    p_mu, p_logvar = model.macro_action.prior_params(states)
    prior_nll = 0.5 * (
        p_logvar + (codes - p_mu).square() * (-p_logvar).exp()
    ).mean(-1)
    support = model.macro_support(states, codes)
    goal = normalized_distance(
        predicted[:, -1], context["high_goal"].expand(count, -1)
    )
    reachability = model.reachability(
        context["low_state"].expand(count, -1), predicted[:, 0]
    )
    score = (
        goal_weight * goal - value_weight * advantage.sum(-1)
        + prior_weight * prior_nll.mean(-1)
        - support_weight * support.mean(-1)
        - reachability_weight * reachability
    )
    return score, predicted


@torch.no_grad()
def categorical_cem(
    model, context, codebook, horizon=2, candidates=256, updates=10,
    elite=32, pool_size=64, smoothing=0.25, goal_weight=1.0,
    value_weight=1.0, prior_weight=0.1, support_weight=0.0,
    reachability_weight=0.0,
):
    p_mu, p_logvar = model.macro_action.prior_params(context["high_state"])
    nll = 0.5 * (
        p_logvar + (codebook - p_mu).square() * (-p_logvar).exp()
    ).mean(-1)
    pool_index = nll.topk(min(pool_size, len(codebook)), largest=False).indices
    pool = codebook[pool_index]
    logits = pool.new_zeros(horizon, len(pool))
    logits[0] = -nll[pool_index]
    last_indices = None
    for _ in range(updates):
        probability = logits.softmax(-1)
        indices = torch.stack([
            torch.multinomial(probability[step], candidates, replacement=True)
            for step in range(horizon)
        ], 1)
        # Guarantee direct coverage of every available first action.
        cover = min(candidates, len(pool))
        indices[:cover, 0] = torch.arange(cover, device=indices.device)
        codes = pool[indices]
        score, _ = rollout_score(
            model, context, codes, goal_weight, value_weight, prior_weight,
            support_weight, reachability_weight,
        )
        chosen = score.topk(min(elite, candidates), largest=False).indices
        for step in range(horizon):
            counts = torch.bincount(
                indices[chosen, step], minlength=len(pool)
            ).to(logits.dtype)
            new_logits = (counts + 1e-3).log()
            logits[step].lerp_(new_logits, 1.0 - smoothing)
        last_indices = indices
    best_local = logits.argmax(-1)
    best_codes = pool[best_local].unsqueeze(0)
    score, predicted = rollout_score(
        model, context, best_codes, goal_weight, value_weight, prior_weight,
        support_weight, reachability_weight,
    )
    return {
        "code": best_codes[:, 0], "subgoal": predicted[:, 0],
        "score": score, "pool_indices": pool_index,
        "selected_codebook_index": pool_index[best_local[0]],
        "refine_codes": pool[logits[0].topk(min(len(pool), 16)).indices],
    }


@torch.no_grad()
def continuous_prior_cem(
    model, context, horizon=2, candidates=256, updates=10, elite=32,
    smoothing=0.25, goal_weight=1.0, value_weight=1.0,
    prior_weight=0.1, support_weight=0.0, reachability_weight=0.0,
):
    mean, logvar = model.macro_action.prior_params(context["high_state"])
    mean = mean[:, None].expand(1, horizon, -1).clone()
    std = (0.5 * logvar)[:, None].exp().expand_as(mean).clone()
    for _ in range(updates):
        codes = mean + torch.randn(
            candidates, horizon, model.d_macro, device=mean.device
        ) * std
        score, _ = rollout_score(
            model, context, codes, goal_weight, value_weight, prior_weight,
            support_weight, reachability_weight,
        )
        selected = codes[score.topk(min(elite, candidates), largest=False).indices]
        mean.lerp_(selected.mean(0, keepdim=True), 1.0 - smoothing)
        std.lerp_(selected.std(0, keepdim=True).clamp_min(1e-3), 1.0 - smoothing)
    score, predicted = rollout_score(
        model, context, mean, goal_weight, value_weight, prior_weight,
        support_weight, reachability_weight,
    )
    return {"code": mean[:, 0], "subgoal": predicted[:, 0], "score": score}


@torch.no_grad()
def execute_low_level(
    model, prefix, prompt_length, period_id, subgoal, token_topk=20,
    max_sentence_tokens=48, low_prior_weight=1.0,
):
    generated = list(prefix)
    target = model.high_to_low(subgoal)
    for _ in range(max_sentence_tokens):
        context = encode_generated(model, generated, prompt_length, period_id)
        logits = model.token_prior(context["low_state"])
        logits[:, model.pad_id] = -torch.inf
        candidate_ids = logits.topk(min(token_topk, logits.shape[-1] - 1)).indices[0]
        if not candidate_ids.eq(period_id).any():
            candidate_ids = torch.cat([
                candidate_ids[:-1], candidate_ids.new_tensor([period_id])
            ])
        actions = model.token_action(candidate_ids).unsqueeze(1)
        count = len(candidate_ids)
        predicted = model.low_predictor.rollout(
            context["low_state"].expand(count, -1), actions,
            state_history=context["low_history"].expand(count, -1, -1),
            action_history=context["low_action_history"].expand(count, -1, -1),
        )[:, 0]
        distance = normalized_distance(predicted, target.expand(count, -1))
        prior_cost = -logits.log_softmax(-1)[0, candidate_ids]
        selected = int(candidate_ids[(distance + low_prior_weight * prior_cost).argmin()])
        generated.append(selected)
        if selected == period_id:
            break
    reached = encode_generated(model, generated, prompt_length, period_id)["high_state"]
    return generated, reached, normalized_distance(reached, subgoal)


@torch.no_grad()
def plan_trace(model, prompt, period_id, codebook, args):
    prefix = list(prompt)
    diagnostics = []
    for _ in range(args.max_sentences):
        if len(prefix) - len(prompt) >= args.max_tokens:
            break
        context = encode_generated(model, prefix, len(prompt), period_id)
        kwargs = dict(
            horizon=args.high_horizon, candidates=args.cem_candidates,
            updates=args.cem_updates, elite=args.cem_elite,
            smoothing=args.cem_smoothing, goal_weight=args.goal_weight,
            value_weight=args.value_weight, prior_weight=args.macro_prior_weight,
            support_weight=args.support_weight,
            reachability_weight=args.reachability_weight,
        )
        if args.macro_support == "codebook":
            proposal = categorical_cem(
                model, context, codebook, pool_size=args.codebook_pool, **kwargs
            )
        else:
            proposal = continuous_prior_cem(model, context, **kwargs)
        before = len(prefix)
        execution_kwargs = dict(
            token_topk=args.token_topk,
            max_sentence_tokens=min(
                args.max_sentence_tokens, args.max_tokens - (len(prefix) - len(prompt))
            ), low_prior_weight=args.low_prior_weight,
        )
        if args.refine_top > 0 and "refine_codes" in proposal:
            refine_codes = proposal["refine_codes"][:args.refine_top]
            refine_score, refine_pred = rollout_score(
                model, context, refine_codes[:, None], args.goal_weight,
                args.value_weight, args.macro_prior_weight,
                args.support_weight, args.reachability_weight,
            )
            executions = []
            for candidate in range(len(refine_codes)):
                candidate_prefix, candidate_reached, candidate_residual = execute_low_level(
                    model, prefix, len(prompt), period_id,
                    refine_pred[candidate:candidate + 1, 0], **execution_kwargs,
                )
                executions.append((
                    float(refine_score[candidate])
                    + args.refine_weight * float(candidate_residual),
                    candidate_prefix, candidate_reached, candidate_residual,
                ))
            _, prefix, reached, residual = min(executions, key=lambda item: item[0])
        else:
            prefix, reached, residual = execute_low_level(
                model, prefix, len(prompt), period_id, proposal["subgoal"],
                **execution_kwargs,
            )
        diagnostics.append({
            "tokens": len(prefix) - before,
            "reachability_residual": float(residual),
            "high_score": float(proposal["score"]),
        })
        # The complete plan is deliberately discarded here. The next loop
        # re-encodes the actual reached prefix and replans from it.
        if len(prefix) == before:
            break
    return prefix[len(prompt):], diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--codebook-examples", type=int, default=256)
    parser.add_argument("--macro-support", choices=["codebook", "prior"], default="codebook")
    parser.add_argument("--high-horizon", type=int, default=2)
    parser.add_argument("--cem-candidates", type=int, default=256)
    parser.add_argument("--cem-updates", type=int, default=10)
    parser.add_argument("--cem-elite", type=int, default=32)
    parser.add_argument("--cem-smoothing", type=float, default=0.25)
    parser.add_argument("--codebook-pool", type=int, default=64)
    parser.add_argument("--token-topk", type=int, default=20)
    parser.add_argument("--max-sentence-tokens", type=int, default=48)
    parser.add_argument("--max-sentences", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--goal-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--macro-prior-weight", type=float, default=0.1)
    parser.add_argument("--support-weight", type=float, default=0.0)
    parser.add_argument("--reachability-weight", type=float, default=0.0)
    parser.add_argument("--refine-top", type=int, default=0)
    parser.add_argument("--refine-weight", type=float, default=1.0)
    parser.add_argument("--low-prior-weight", type=float, default=1.0)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = SentenceHierarchyJEPA(
        len(vocab), vocab.pad_id, **cfg.model
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    train = SemanticBoundaryLMDataset(
        vocab, size=args.codebook_examples, seed=cfg.data.train_seed,
        boundary_mode="semantic", modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range), leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    codebook, _ = build_codebook(model, train, args.codebook_examples, vocab.pad_id)
    test = SemanticBoundaryLMDataset(
        vocab, size=args.examples, seed=cfg.data.val_seed + 104729,
        boundary_mode="semantic", modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range), leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    period = vocab.token_to_id["."]
    exact, correct, total, valid_sentences, residuals, records = 0, 0, 0, 0, [], []
    for index in range(args.examples):
        item = test[index]
        prompt = item["tokens"][:item["prompt_len"]]
        reference = item["tokens"][item["prompt_len"]:]
        generated, diagnostics = plan_trace(model, prompt, period, codebook, args)
        exact += int(generated == reference)
        length = min(len(generated), len(reference))
        correct += sum(generated[position] == reference[position] for position in range(length))
        total += max(len(generated), len(reference))
        valid_sentences += sum(token == period for token in generated)
        residuals.extend(step["reachability_residual"] for step in diagnostics)
        records.append({
            "episode": index, "exact": generated == reference,
            "generated_tokens": len(generated), "reference_tokens": len(reference),
            "generated": vocab.decode(generated),
            "reference": vocab.decode(reference), "steps": diagnostics,
        })
    result = {
        "exact_trace_success": exact / args.examples,
        "token_accuracy": correct / max(total, 1),
        "completed_sentence_count": valid_sentences,
        "mean_reachability_residual": sum(residuals) / max(len(residuals), 1),
        "macro_support": args.macro_support,
        "high_horizon": args.high_horizon,
        "cem_candidates": args.cem_candidates, "cem_updates": args.cem_updates,
        "uses_symbolic_feasibility": False, "uses_auxiliary_lm": False,
        "uses_oracle_goal": False, "iterative_actual_state_replanning": True,
        "episodes": records,
    }
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    destination = Path(args.ckpt).parent / (
        f"sentence_planning_{args.macro_support}_h{args.high_horizon}{suffix}.json"
    )
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
