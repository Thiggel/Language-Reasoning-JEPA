"""MPC and open-loop beam evaluation for the pooled-prefix JEPA."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.semantic_lm import SemanticBoundaryLMDataset
from textjepa.models.pooled_sentence_jepa import PooledSentenceJEPA


def summarize_examples(generated, references, boundary_ids, bootstrap_seed=0):
    """Aggregate trace-level metrics without hiding the tiny-sample variance."""
    if len(generated) != len(references) or not references:
        raise ValueError("generated and reference traces must be nonempty and matched")
    per_trace, first_error, exact = [], [], 0
    quartile_correct = [0] * 4
    quartile_total = [0] * 4
    boundary_correct = boundary_total = 0
    for predicted, reference in zip(generated, references):
        if len(predicted) != len(reference) or not reference:
            raise ValueError("each generated trace must match its nonempty oracle length")
        matches = [int(a == b) for a, b in zip(predicted, reference)]
        per_trace.append(sum(matches) / len(matches))
        exact += int(all(matches))
        first = next((index for index, match in enumerate(matches) if not match), len(matches))
        first_error.append(first / len(matches))
        for index, (match, token) in enumerate(zip(matches, reference)):
            bucket = min(3, (4 * index) // len(reference))
            quartile_correct[bucket] += match
            quartile_total[bucket] += 1
            if token in boundary_ids:
                boundary_correct += match
                boundary_total += 1
    rng = random.Random(int(bootstrap_seed))
    samples = []
    for _ in range(2000):
        draw = [per_trace[rng.randrange(len(per_trace))] for _ in per_trace]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    return {
        "token_accuracy": sum(per_trace) / len(per_trace),
        "token_accuracy_ci95": [samples[49], samples[1949]],
        "exact_trace_success": exact / len(references),
        "mean_first_error_fraction": sum(first_error) / len(first_error),
        "boundary_token_accuracy": boundary_correct / max(boundary_total, 1),
        "position_quartile_accuracy": [
            correct / max(total, 1)
            for correct, total in zip(quartile_correct, quartile_total)
        ],
        "per_trace_token_accuracy": per_trace,
    }


def normalized_distance(left, right):
    return F.mse_loss(
        F.layer_norm(left, left.shape[-1:]),
        F.layer_norm(right, right.shape[-1:]), reduction="none",
    ).mean(-1)


@torch.no_grad()
def _context(model, prefix, prompt_length):
    device = next(model.parameters()).device
    ids = torch.tensor(prefix, device=device).unsqueeze(0)
    states = model.state_encoder(ids)
    start = prompt_length - 1
    reasoning = ids[:, prompt_length:]
    actions = model.token_action(reasoning) if reasoning.numel() else states.new_zeros(
        1, 0, model.d_action
    )
    return {
        "state": states[:, -1], "states": states[:, start:],
        "actions": actions, "prompt": states[:, start],
    }


@torch.no_grad()
def beam_plan(
    model, prefix, goal, depth=4, width=8, score_mode="value",
    proposal_mode="prior", proposal_topk=20, prompt_length=None,
    prior_score_weight=1.0,
):
    if score_mode not in {"oracle", "value", "prior"}:
        raise ValueError(f"unknown score mode: {score_mode}")
    if proposal_mode not in {"prior", "all"}:
        raise ValueError(f"unknown proposal mode: {proposal_mode}")
    if proposal_mode == "prior" and model.token_prior is None:
        raise ValueError("prior proposals require a trained token prior")
    if proposal_mode == "all" and prior_score_weight:
        raise ValueError("prior score weight requires prior proposals")
    prompt_length = len(prefix) if prompt_length is None else int(prompt_length)
    context = _context(model, prefix, prompt_length)
    beams = [([], context["states"], context["actions"], 0.0, 0.0, 0.0)]
    expanded = 0
    for _ in range(int(depth)):
        beam_count = len(beams)
        state_history = torch.cat([item[1] for item in beams], 0)
        action_history = torch.cat([item[2] for item in beams], 0)
        current = state_history[:, -1]
        proposal_logits = None
        if proposal_mode == "prior":
            logits = model.token_prior(current)
            logits[:, model.pad_id] = -torch.inf
            ids = logits.topk(min(proposal_topk, model.vocab_size - 1), dim=-1).indices
            proposal_logits = logits.log_softmax(-1).gather(1, ids)
        else:
            vocabulary = torch.arange(model.vocab_size, device=current.device)
            vocabulary = vocabulary[vocabulary.ne(model.pad_id)]
            ids = vocabulary[None].expand(beam_count, -1)
        branch = ids.shape[1]
        expanded += beam_count * branch
        flat_ids = ids.reshape(-1)
        actions = model.token_action(flat_ids)
        repeated_states = state_history[:, None].expand(-1, branch, -1, -1).reshape(
            beam_count * branch, state_history.shape[1], -1
        )
        repeated_actions = action_history[:, None].expand(-1, branch, -1, -1).reshape(
            beam_count * branch, action_history.shape[1], model.d_action
        )
        repeated_current = current[:, None].expand(-1, branch, -1).reshape(
            beam_count * branch, -1
        )
        predicted = model.predictor.rollout(
            repeated_current, actions[:, None], state_history=repeated_states,
            action_history=repeated_actions,
        )[:, 0]
        prompt = context["prompt"].expand(beam_count * branch, -1)
        advantage = model.token_value(repeated_current, prompt, actions)
        cumulative = torch.tensor([item[3] for item in beams], device=current.device)
        cumulative = cumulative[:, None].expand(-1, branch).reshape(-1)
        cumulative_prior = torch.tensor([item[4] for item in beams], device=current.device)
        cumulative_prior = cumulative_prior[:, None].expand(-1, branch).reshape(-1)
        updated_prior = cumulative_prior + (
            proposal_logits.reshape(-1) if proposal_logits is not None else 0.0
        )
        if score_mode == "oracle":
            updated = cumulative
            rank_score = -normalized_distance(
                predicted, goal.expand(beam_count * branch, -1)
            ) + prior_score_weight * updated_prior
        elif score_mode == "value":
            updated = cumulative + advantage
            rank_score = updated + prior_score_weight * updated_prior
        else:
            updated = cumulative
            rank_score = updated_prior
        keep = rank_score.topk(min(width, len(rank_score))).indices
        next_beams = []
        for flat in keep.tolist():
            parent, candidate = divmod(flat, branch)
            next_beams.append((
                beams[parent][0] + [int(ids[parent, candidate])],
                torch.cat([state_history[parent:parent + 1], predicted[flat:flat + 1, None]], 1),
                torch.cat([action_history[parent:parent + 1], actions[flat:flat + 1, None]], 1),
                float(updated[flat]), float(updated_prior[flat]), float(rank_score[flat]),
            ))
        beams = next_beams
    return {"tokens": beams[0][0], "score": beams[0][5], "expanded": expanded}


@torch.no_grad()
def generate(model, prompt, goal, length, args):
    generated, expanded = list(prompt), 0
    if args.depth == 0:
        if model.token_prior is None:
            raise ValueError("depth zero is the token-prior control and requires a prior")
        for _ in range(length):
            context = _context(model, generated, len(prompt))
            logits = model.token_prior(context["state"])
            logits[:, model.pad_id] = -torch.inf
            generated.append(int(logits.argmax(-1)))
        return generated[len(prompt):], expanded
    while len(generated) - len(prompt) < length:
        remaining = length - (len(generated) - len(prompt))
        planning_depth = min(args.depth, remaining)
        plan = beam_plan(
            model, generated, goal, depth=planning_depth, width=args.width,
            score_mode=args.score, proposal_mode=args.proposals,
            proposal_topk=args.proposal_topk, prompt_length=len(prompt),
            prior_score_weight=args.prior_score_weight,
        )
        take = 1 if args.planner == "mpc" else planning_depth
        generated.extend(plan["tokens"][:take])
        expanded += plan["expanded"]
    return generated[len(prompt):], expanded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--depth", type=int, choices=(0, 1, 2, 4, 8, 16), required=True)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--planner", choices=("mpc", "beam"), required=True)
    parser.add_argument("--score", choices=("oracle", "value", "prior"), required=True)
    parser.add_argument("--proposals", choices=("prior", "all"), required=True)
    parser.add_argument("--proposal-topk", type=int, default=20)
    parser.add_argument("--prior-score-weight", type=float, default=1.0)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = PooledSentenceJEPA(
        len(vocab), vocab.pad_id, period_id=vocab.token_to_id["."],
        question_id=vocab.token_to_id["?"], **cfg.model,
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    eval_seed = cfg.data.val_seed + 104729 if args.eval_seed is None else args.eval_seed
    dataset = SemanticBoundaryLMDataset(
        vocab, size=args.examples, seed=eval_seed,
        boundary_mode="semantic", modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range), leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    generated_traces, reference_traces, expanded = [], [], 0
    for index in range(args.examples):
        item = dataset[index]
        prompt = item["tokens"][:item["prompt_len"]]
        reference = item["tokens"][item["prompt_len"]:][:args.max_tokens]
        full = torch.tensor(item["tokens"], device=args.device).unsqueeze(0)
        goal = model.teacher(full)[:, len(item["tokens"]) - 1]
        generated, work = generate(model, prompt, goal, len(reference), args)
        generated_traces.append(generated)
        reference_traces.append(reference)
        expanded += work
    result = {
        **summarize_examples(
            generated_traces, reference_traces,
            {vocab.token_to_id["."], vocab.token_to_id["?"]}, eval_seed,
        ),
        "examples": args.examples, "eval_seed": eval_seed,
        "planner": args.planner, "depth": args.depth, "width": args.width,
        "score": args.score, "proposals": args.proposals,
        "proposal_topk": args.proposal_topk,
        "prior_score_weight": args.prior_score_weight,
        "mean_expanded_candidates": expanded / args.examples,
        "uses_oracle_goal": args.score == "oracle", "uses_oracle_length": True,
        "uses_symbolic_feasibility": False, "uses_auxiliary_lm": False,
    }
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.ckpt).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.output_tag}" if args.output_tag else ""
    destination = output_dir / (
        f"pooled{tag}_{args.planner}_{args.proposals}_{args.score}_d{args.depth}_w{args.width}"
        f"_pw{args.prior_score_weight:g}.json"
    )
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
