#!/usr/bin/env python3
"""Generate complete iGSM solution text with receding-horizon edit MPC."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from textjepa.data.faithful_token_edits import FaithfulTokenEditDataset, MASK_TOKEN
from textjepa.data.token_edit_distance import boundary_token_edit_distance
from textjepa.planning.multiscale_edit_mpc import MultiscaleEditMPC, copy_buffer, flatten
from textjepa.utils.checkpoint import load_run


REGIMES = {
    # Train-support complexity, new problems.
    "id": {"op_range": (8, 21), "max_op": 21, "max_edge": 28, "seed": 7301},
    # Strictly longer official-generator problems than any training example.
    "ood_medium": {"op_range": (22, 24), "max_op": 28, "max_edge": 36,
                   "seed": 7302},
    "ood_long": {"op_range": (25, 28), "max_op": 28, "max_edge": 36,
                 "seed": 7303},
}


def softmax_entropy(scores):
    if not scores:
        return 0.0
    value = torch.tensor(list(scores.values()), dtype=torch.float)
    p = value.softmax(0)
    return float(-(p * p.clamp_min(1e-12).log()).sum())


def run_episode(planner, vocab, item, horizon, refinement_rounds,
                max_actions=0):
    current = copy_buffer(item["buffers"][0])
    target = copy_buffer(item["buffers"][-1])
    mask_id = vocab.token_to_id[MASK_TOKEN]
    initial_masks = sum(token == mask_id for token in flatten(current))
    token_count = len(flatten(current))
    budget = initial_masks + int(refinement_rounds) * token_count
    if int(max_actions) > 0:
        budget = min(budget, int(max_actions))
    selected_q, entropies = [], []
    stopped = "budget"
    for _ in range(budget):
        has_masks = mask_id in flatten(current)
        action, posterior, root_q = planner.first_action(
            item["prompt"], current, horizon,
            allow_refinement=not has_masks and refinement_rounds > 0,
        )
        if action is None:
            stopped = "no_candidates"
            break
        if not has_masks and root_q <= 0:
            stopped = "nonpositive_action_value"
            break
        selected_q.append(root_q)
        entropies.append(softmax_entropy(posterior))
        from textjepa.data.faithful_token_edits import _apply
        _apply(current, action)
    distance = boundary_token_edit_distance(current, target)
    exact_tokens = sum(
        int(a == b) for a, b in zip(flatten(current), flatten(target))
    )
    answer_text = vocab.encode(f"the answer is {item['answer']} .")
    return {
        "exact_sequence": current == target,
        "exact_answer_sentence": bool(current and current[-1] == target[-1]),
        "contains_rendered_answer": any(
            flatten(current)[i:i + len(answer_text)] == answer_text
            for i in range(max(len(flatten(current)) - len(answer_text) + 1, 0))
        ),
        "token_accuracy": exact_tokens / max(len(flatten(target)), 1),
        "token_edit_distance": distance,
        "normalized_edit_distance": distance / max(len(flatten(target)), 1),
        "generated_tokens": len(flatten(current)),
        "target_tokens": len(flatten(target)),
        "steps": len(selected_q),
        "mean_selected_action_value": sum(selected_q) / max(len(selected_q), 1),
        "mean_root_posterior_entropy": sum(entropies) / max(len(entropies), 1),
        "stop_reason": stopped,
        "generated_text": [vocab.decode(sentence) for sentence in current],
        "target_text": [vocab.decode(sentence) for sentence in target],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--regime", choices=REGIMES, default="id")
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--horizon", type=int, choices=(1, 2, 4, 8, 16), required=True)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--top-positions", type=int, default=4)
    parser.add_argument("--top-tokens", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--refinement-rounds", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=0)
    parser.add_argument("--prior-weight", type=float, default=0.05)
    parser.add_argument("--action-value-weight", type=float, default=1.0)
    parser.add_argument("--state-value-weight", type=float, default=0.25)
    parser.add_argument("--macro-prior-weight", type=float, default=0.05)
    parser.add_argument("--macro-value-weight", type=float, default=0.25)
    parser.add_argument("--disable-base-prior", action="store_true")
    args = parser.parse_args()

    model, vocab, cfg = load_run(args.ckpt, args.device)
    regime = REGIMES[args.regime]
    dataset = FaithfulTokenEditDataset(
        vocab, size=args.examples, seed=regime["seed"],
        max_op=regime["max_op"], max_edge=regime["max_edge"],
        op_range=regime["op_range"], corruption_mode="iterative_refinement",
        trajectory_variants=1, refinement_probability=0.0,
        min_edits=1, max_edits=1,
    )
    planner = MultiscaleEditMPC(
        model, vocab, args.device, beam_width=args.beam_width,
        top_positions=args.top_positions, top_tokens=args.top_tokens,
        max_candidates=args.max_candidates, prior_weight=args.prior_weight,
        action_value_weight=args.action_value_weight,
        state_value_weight=args.state_value_weight,
        macro_prior_weight=args.macro_prior_weight,
        macro_value_weight=args.macro_value_weight,
        use_base_prior=not args.disable_base_prior,
    )
    episodes = []
    operation_counts = []
    for index in range(args.examples):
        item = dataset[index]
        episode = run_episode(
            planner, vocab, item, args.horizon, args.refinement_rounds,
            args.max_actions,
        )
        fp, _ = dataset.source.problem(index)
        operation_counts.append(int(fp.p.n_op))
        # Full text is useful for qualitative audit but would dominate JSON.
        if index >= 4:
            episode.pop("generated_text")
            episode.pop("target_text")
        episodes.append(episode)

    mean = lambda key: sum(float(x[key]) for x in episodes) / len(episodes)
    payload = {
        "information_regime": (
            "target_free_planning; terminal target used only by evaluator; "
            "candidate_privileged=false; structure_known=true"
        ),
        "generation": "receding_horizon_execute_first_token_replacement",
        "planner": "base_prior_plus_distilled_action_value_plus_jepa_state_value",
        "hierarchy": (
            "executable_primitive_beams_macro-prior-and-subgoal-reranked"
            if model.use_macro else "none"
        ),
        "variant": model.variant,
        "regime": args.regime,
        "requested_op_range": list(regime["op_range"]),
        "observed_op_range": [min(operation_counts), max(operation_counts)],
        "horizon": args.horizon,
        "settings": {
            key: value for key, value in vars(args).items()
            if key not in {"ckpt", "out", "device"}
        },
        "exact_sequence_rate": mean("exact_sequence"),
        "exact_answer_sentence_rate": mean("exact_answer_sentence"),
        "contains_rendered_answer_rate": mean("contains_rendered_answer"),
        "token_accuracy": mean("token_accuracy"),
        "normalized_token_edit_distance": mean("normalized_edit_distance"),
        "mean_steps": mean("steps"),
        "mean_selected_action_value": mean("mean_selected_action_value"),
        "mean_root_posterior_entropy": mean("mean_root_posterior_entropy"),
        "episodes": episodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
