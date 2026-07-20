"""Receding-horizon token beam search in a single causal JEPA state space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.semantic_lm import SemanticBoundaryLMDataset
from textjepa.models.flat_sentence_jepa import FlatSentenceJEPA


def normalized_distance(left, right):
    return F.mse_loss(
        F.layer_norm(left, left.shape[-1:]),
        F.layer_norm(right, right.shape[-1:]), reduction="none",
    ).mean(-1)


@torch.no_grad()
def _context(model, prefix, prompt_length):
    device = next(model.parameters()).device
    ids = torch.tensor(prefix, device=device).unsqueeze(0)
    states = model.encoder(ids)
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
    model, prefix, goal, depth=4, width=8, score_mode="oracle",
    proposal_mode="prior", proposal_topk=20, prompt_length=None,
):
    """Return one open-loop beam; no grammar or feasibility filter is used."""
    if score_mode not in {"oracle", "value", "prior"}:
        raise ValueError(f"unknown score mode: {score_mode}")
    if proposal_mode not in {"prior", "all"}:
        raise ValueError(f"unknown proposal mode: {proposal_mode}")
    if proposal_mode == "prior" and model.token_prior is None:
        raise ValueError("prior proposals require a trained token prior")
    prompt_length = len(prefix) if prompt_length is None else int(prompt_length)
    context = _context(model, prefix, prompt_length)
    # tokens, states, actions, cumulative GAR score, ranking score
    beams = [([], context["states"], context["actions"], 0.0, 0.0)]
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
        repeated_states = state_history[:, None].expand(
            -1, branch, -1, -1
        ).reshape(beam_count * branch, state_history.shape[1], -1)
        repeated_actions = action_history[:, None].expand(
            -1, branch, -1, -1
        ).reshape(beam_count * branch, action_history.shape[1], model.d_action)
        repeated_current = current[:, None].expand(-1, branch, -1).reshape(
            beam_count * branch, -1
        )
        predicted = model.predictor.rollout(
            repeated_current, actions[:, None], state_history=repeated_states,
            action_history=repeated_actions,
        )[:, 0]
        prompt = context["prompt"].expand(beam_count * branch, -1)
        advantage = model.token_value(repeated_current, prompt, actions)
        cumulative = torch.tensor(
            [item[3] for item in beams], device=current.device
        )[:, None].expand(-1, branch).reshape(-1)
        if score_mode == "oracle":
            rank_score = -normalized_distance(
                predicted, goal.expand(beam_count * branch, -1)
            )
            updated = cumulative
        elif score_mode == "value":
            updated = advantage + cumulative
            rank_score = updated
        else:
            if proposal_logits is None:
                raise ValueError("prior scoring requires prior proposals")
            updated = proposal_logits.reshape(-1) + cumulative
            rank_score = updated
        keep = rank_score.topk(min(width, len(rank_score))).indices
        next_beams = []
        for flat in keep.tolist():
            parent, candidate = divmod(flat, branch)
            next_beams.append((
                beams[parent][0] + [int(ids[parent, candidate])],
                torch.cat([
                    state_history[parent:parent + 1],
                    predicted[flat:flat + 1, None],
                ], 1),
                torch.cat([
                    action_history[parent:parent + 1],
                    actions[flat:flat + 1, None],
                ], 1),
                float(updated[flat]), float(rank_score[flat]),
            ))
        beams = next_beams
    best = beams[0]
    return {
        "tokens": best[0], "score": best[4], "expanded": expanded,
        "uses_symbolic_feasibility": False,
    }


@torch.no_grad()
def generate(model, prompt, goal, length, args):
    generated = list(prompt)
    expanded = 0
    for _ in range(length):
        plan = beam_plan(
            model, generated, goal, depth=args.depth, width=args.width,
            score_mode=args.score, proposal_mode=args.proposals,
            proposal_topk=args.proposal_topk, prompt_length=len(prompt),
        )
        generated.append(plan["tokens"][0])
        expanded += plan["expanded"]
    return generated[len(prompt):], expanded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--depth", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--score", choices=("oracle", "value", "prior"), required=True)
    parser.add_argument("--proposals", choices=("prior", "all"), required=True)
    parser.add_argument("--proposal-topk", type=int, default=20)
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = FlatSentenceJEPA(len(vocab), vocab.pad_id, **cfg.model).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    dataset = SemanticBoundaryLMDataset(
        vocab, size=args.examples, seed=cfg.data.val_seed + 104729,
        boundary_mode="semantic", modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range), leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    exact = correct = total = expanded = 0
    for index in range(args.examples):
        item = dataset[index]
        prompt = item["tokens"][:item["prompt_len"]]
        reference = item["tokens"][item["prompt_len"]:][:args.max_tokens]
        full = torch.tensor(item["tokens"], device=args.device).unsqueeze(0)
        # Oracle terminal representation is used only by the explicitly
        # diagnostic oracle scorer. The deployable GAR scorer never reads it.
        goal = model.teacher(full)[:, len(item["tokens"]) - 1]
        generated, work = generate(model, prompt, goal, len(reference), args)
        exact += int(generated == reference)
        correct += sum(a == b for a, b in zip(generated, reference))
        total += len(reference)
        expanded += work
    result = {
        "exact_trace_success": exact / args.examples,
        "token_accuracy": correct / max(total, 1),
        "examples": args.examples, "depth": args.depth, "width": args.width,
        "score": args.score, "proposals": args.proposals,
        "proposal_topk": args.proposal_topk,
        "mean_expanded_candidates": expanded / args.examples,
        "uses_oracle_goal": args.score == "oracle",
        "uses_oracle_length": True,
        "uses_symbolic_feasibility": False,
        "uses_auxiliary_lm": False,
    }
    destination = Path(args.ckpt).parent / (
        f"flat_sentence_beam_{args.proposals}_{args.score}_d{args.depth}_w{args.width}.json"
    )
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
