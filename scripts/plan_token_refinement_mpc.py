"""Target-free receding-horizon beam MPC for replacement-only refinement."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from scripts.plan_faithful_token_edits import (
    Buffer, Edit, _buffer_tensor, buffer_distance, copy_buffer,
)
from textjepa.data.faithful_token_edits import MASK_TOKEN, OPS, _apply
from textjepa.utils.checkpoint import build_dataset, load_run


@dataclass
class BeamNode:
    buffer: Buffer
    states: torch.Tensor
    mask: torch.Tensor
    score: float
    actions: tuple[Edit, ...]
    root: Edit | None
    root_gar: float | None


@torch.no_grad()
def encode_prompt(model, prompt: Buffer, pad_id: int, device: str) -> torch.Tensor:
    raw = _buffer_tensor(prompt, pad_id, device)[:, 0]
    chunks = model.encode_chunks(raw)
    return chunks.mean(1)


@torch.no_grad()
def prior_candidates(
    model, states: torch.Tensor, mask: torch.Tensor, prompt: torch.Tensor,
    buffer: Buffer, top_positions: int, top_tokens: int,
    max_candidates: int, excluded_tokens: set[int],
) -> list[tuple[Edit, float]]:
    dummy = torch.zeros(1, dtype=torch.long, device=states.device)
    position_logits, _ = model.refinement_prior(states, mask, prompt, dummy)
    position_logp = position_logits.log_softmax(-1)
    count = min(top_positions, int(mask.sum().item()))
    positions = position_logp.topk(count, -1).indices[0]
    repeated_states = states.expand(count, -1, -1)
    repeated_mask = mask.expand(count, -1)
    repeated_prompt = prompt.expand(count, -1)
    _, content_logits = model.refinement_prior(
        repeated_states, repeated_mask, repeated_prompt, positions
    )
    content_logp = content_logits.log_softmax(-1)
    current = [token for sentence in buffer for token in sentence]
    candidates = []
    for row, position in enumerate(positions.tolist()):
        blocked = set(excluded_tokens)
        blocked.add(current[position])
        ranked = content_logp[row].argsort(descending=True).tolist()
        accepted = 0
        for token in ranked:
            if token in blocked:
                continue
            candidates.append((
                ("replace", position, token),
                float(position_logp[0, position] + content_logp[row, token]),
            ))
            accepted += 1
            if accepted >= top_tokens:
                break
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:max_candidates]


@torch.no_grad()
def expand_node(
    model, node: BeamNode, prompt: torch.Tensor,
    top_positions: int, top_tokens: int, max_candidates: int,
    prior_weight: float, gar_weight: float, excluded_tokens: set[int],
) -> list[BeamNode]:
    proposed = prior_candidates(
        model, node.states, node.mask, prompt, node.buffer,
        top_positions, top_tokens, max_candidates, excluded_tokens,
    )
    if not proposed:
        return []
    count = len(proposed)
    operations = torch.full(
        (count,), OPS["replace"], dtype=torch.long, device=node.states.device
    )
    positions = torch.tensor(
        [action[1] for action, _ in proposed], device=node.states.device
    )
    content_ids = torch.tensor(
        [int(action[2]) for action, _ in proposed], device=node.states.device
    )
    content = model.chunk_encoder.tok(content_ids)
    states = node.states.expand(count, -1, -1)
    mask = node.mask.expand(count, -1)
    next_states, next_mask, action_codes = model.token_pred(
        states, mask, operations, positions, content,
        prompt.expand(count, -1), return_action=True,
    )
    pooled = model._pool_tokens(states, mask)
    gar = model.gar_head(torch.cat([pooled, action_codes], -1)).squeeze(-1)
    children = []
    for index, ((action, log_prior), value) in enumerate(zip(proposed, gar)):
        outcome = copy_buffer(node.buffer)
        _apply(outcome, action)
        root = node.root if node.root is not None else action
        root_gar = node.root_gar if node.root_gar is not None else float(value)
        children.append(BeamNode(
            outcome, next_states[index:index + 1], next_mask[index:index + 1],
            node.score + prior_weight * log_prior + gar_weight * float(value),
            (*node.actions, action), root, root_gar,
        ))
    return children


@torch.no_grad()
def search_first_action(
    model, prompt_tokens: Buffer, current: Buffer, pad_id: int, device: str,
    horizon: int, beam_width: int, top_positions: int, top_tokens: int,
    max_candidates: int, prior_weight: float, gar_weight: float,
    excluded_tokens: set[int],
) -> tuple[Edit, dict[Edit, float], float]:
    """Plan H latent transitions without accepting a target/goal argument."""
    prompt = encode_prompt(model, prompt_tokens, pad_id, device)
    raw = _buffer_tensor(current, pad_id, device)
    states, mask = model.encode_token_buffers(raw, mode="online")
    beam = [BeamNode(
        copy_buffer(current), states[:, 0], mask[:, 0], 0.0, (), None, None,
    )]
    for _ in range(horizon):
        expanded = []
        for node in beam:
            expanded.extend(expand_node(
                model, node, prompt, top_positions, top_tokens,
                max_candidates, prior_weight, gar_weight, excluded_tokens,
            ))
        if not expanded:
            break
        expanded.sort(key=lambda node: node.score, reverse=True)
        beam = expanded[:beam_width]
    if not beam or beam[0].root is None:
        raise RuntimeError("refinement prior produced no executable candidate")
    root_scores: dict[Edit, float] = {}
    for node in beam:
        root_scores[node.root] = max(root_scores.get(node.root, -math.inf), node.score)
    if beam[0].root_gar is None:
        raise RuntimeError("selected beam has no root GAR value")
    return beam[0].root, root_scores, beam[0].root_gar


def posterior_metrics(scores: dict[Edit, float], current: Buffer,
                      target: Buffer) -> tuple[float, float, float]:
    actions = list(scores)
    logits = torch.tensor([scores[action] for action in actions])
    probabilities = logits.softmax(0)
    before = buffer_distance(current, target)
    advantages = []
    for action in actions:
        outcome = copy_buffer(current)
        _apply(outcome, action)
        advantages.append(before - buffer_distance(outcome, target))
    advantage = torch.tensor(advantages, dtype=probabilities.dtype)
    positive_mass = float(probabilities[advantage > 0].sum())
    expected = float((probabilities * advantage).sum())
    entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    return positive_mass, expected, entropy


def run_episode(model, vocab, item: dict, device: str, **search) -> dict:
    current = copy_buffer(item["buffers"][0])
    target = copy_buffer(item["buffers"][-1])
    initial = buffer_distance(current, target)
    max_steps = int(search.pop("max_steps"))
    # Zero means a deployment-visible budget large enough to reveal every
    # token once and repair every token once.  It depends only on the current
    # buffer length, never on the hidden clean target or its distance.
    if max_steps <= 0:
        max_steps = 2 * sum(len(sentence) for sentence in current)
    stop_threshold = float(search.pop("stop_threshold"))
    selected_advantages, positive_mass, expected, entropy = [], [], [], []
    excluded = {
        vocab.pad_id, vocab.token_to_id[vocab.UNK],
        vocab.token_to_id[MASK_TOKEN],
    }
    stopped_by_value = False
    for _ in range(max_steps):
        action, root_scores, selected_gar = search_first_action(
            model, item["prompt"], current, vocab.pad_id, device,
            excluded_tokens=excluded, **search,
        )
        if selected_gar <= stop_threshold:
            stopped_by_value = True
            break
        mass, exp_adv, ent = posterior_metrics(root_scores, current, target)
        before = buffer_distance(current, target)
        _apply(current, action)
        selected_advantages.append(before - buffer_distance(current, target))
        positive_mass.append(mass)
        expected.append(exp_adv)
        entropy.append(ent)
    final = buffer_distance(current, target)
    mean = lambda values: sum(values) / max(len(values), 1)
    return {
        "initial_distance": initial,
        "final_distance": final,
        "normalized_improvement": (initial - final) / max(initial, 1),
        "exact_recovery": current == target,
        "steps": len(selected_advantages),
        "stopped_by_value": stopped_by_value,
        "selected_advantage": mean(selected_advantages),
        "posterior_positive_mass": mean(positive_mass),
        "posterior_expected_true_advantage": mean(expected),
        "posterior_entropy": mean(entropy),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--top-positions", type=int, default=4)
    parser.add_argument("--top-tokens", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument(
        "--max-steps", type=int, default=0,
        help="replacement budget; <=0 uses twice the observable token count",
    )
    parser.add_argument(
        "--stop-threshold", type=float, default=0.0,
        help="stop before execution when selected one-step GAR is not larger",
    )
    parser.add_argument("--prior-weight", type=float, default=0.05)
    parser.add_argument("--gar-weight", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model, vocab, cfg = load_run(args.ckpt, args.device)
    if not getattr(model, "refinement_prior_enabled", False):
        raise ValueError("checkpoint has no learned refinement prior")
    dataset = build_dataset(cfg, vocab, "test", size=args.examples)
    settings = vars(args).copy()
    settings.pop("ckpt"); settings.pop("out"); settings.pop("examples")
    settings.pop("device")
    episodes = [
        run_episode(model, vocab, dataset[index], args.device, **settings)
        for index in range(args.examples)
    ]
    mean = lambda key: sum(item[key] for item in episodes) / len(episodes)
    payload = {
        "information_regime": "deployment_feasible_target_free_prior_jepa_gar",
        "control": "receding_horizon_execute_first_action_only",
        "target_usage": "post-decision metrics only",
        "settings": settings,
        "normalized_edit_distance_improvement": mean("normalized_improvement"),
        "exact_recovery_rate": mean("exact_recovery"),
        "value_stop_rate": mean("stopped_by_value"),
        "mean_selected_true_advantage": mean("selected_advantage"),
        "mean_posterior_positive_mass": mean("posterior_positive_mass"),
        "mean_posterior_expected_true_advantage": mean(
            "posterior_expected_true_advantage"
        ),
        "mean_posterior_entropy": mean("posterior_entropy"),
        "episodes": episodes,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
