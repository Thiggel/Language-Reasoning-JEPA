#!/usr/bin/env python3
"""Matched generation evaluation on the original iGSM-medium length cells.

The paper trains on ``op <= 15`` and evaluates mixed ID, the ``op = 15``
boundary, and exact OOD lengths 20--23.  This evaluator also requests the
official test template-hash bins 16--22.  Current July-22 edit checkpoints
were trained by an older adapter on all hash bins, so their output explicitly
marks template-holdout contamination while preserving valid length OOD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from textjepa.data.faithful_token_edits import (
    FaithfulTokenEditDataset,
    STEP_BOUNDARY_TOKEN,
    faithful_replacement_vocab,
)
from textjepa.models.masked_diffusion_lm import MaskedDiffusionLM
from textjepa.planning.multiscale_edit_mpc import (
    MultiscaleEditMPC,
    _chunks_tensor,
    flatten,
)
from textjepa.utils.checkpoint import load_run


PAPER_MEDIUM_CELLS = {
    "id_mixed_op_le_15": (15, (None, 15)),
    "id_boundary_op_15": (15, (15, 15)),
    "ood_op_20": (20, (20, 20)),
    "ood_op_21": (21, (21, 21)),
    "ood_op_22": (22, (22, 22)),
    "ood_op_23": (23, (23, 23)),
}
OFFICIAL_TEST_HASH_BINS = tuple(range(16, 23))


def mdlm_model_kwargs(saved, payload, vocab):
    """Recover every shape-bearing model argument from a training checkpoint."""
    return {
        "vocab_size": payload["vocab_size"],
        "pad_id": payload["pad_id"],
        "mask_id": payload["mask_id"],
        "d_model": saved.d_model,
        "n_layers": saved.layers,
        "n_heads": saved.heads,
        "ff_mult": getattr(saved, "ff_mult", 4.0),
        "max_sequence_len": saved.max_sequence_len,
        "boundary_id": vocab.token_to_id[STEP_BOUNDARY_TOKEN],
        "attention_backend": getattr(saved, "attention_backend", "auto"),
        "sequence_packing": getattr(saved, "sequence_packing", False),
    }


def load_mdlm(path: str, device: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = argparse.Namespace(**payload["args"])
    vocab = faithful_replacement_vocab()
    model = MaskedDiffusionLM(**mdlm_model_kwargs(saved, payload, vocab))
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), vocab


def make_jepa_planner(
    model, vocab, device: str, scoring: str, candidate_budget: int,
    beam_width: int,
):
    """Construct a target-blind planner with an information-matched proposal set."""
    if candidate_budget < 1:
        raise ValueError("candidate_budget must be positive")
    if beam_width < 1:
        raise ValueError("beam_width must be positive")
    if scoring == "prior":
        prior_weight, action_value_weight, state_value_weight = 1.0, 0.0, 0.0
    elif scoring == "gar":
        # GAR is the deployment-time V(s,a) signal.  The learned state-distance
        # head is retained as a smaller tie-breaking/control-variate term.
        prior_weight, action_value_weight, state_value_weight = 0.05, 1.0, 0.25
    else:
        raise ValueError(f"unknown planner scoring: {scoring}")
    return MultiscaleEditMPC(
        model, vocab, device=device, beam_width=beam_width,
        top_positions=candidate_budget, top_tokens=candidate_budget,
        max_candidates=candidate_budget, prior_weight=prior_weight,
        action_value_weight=action_value_weight,
        state_value_weight=state_value_weight, macro_prior_weight=0.0,
        macro_value_weight=0.0, use_base_prior=True,
    )


def make_dataset(vocab, cell: str, examples: int, seed: int):
    max_op, op_range = PAPER_MEDIUM_CELLS[cell]
    return FaithfulTokenEditDataset(
        vocab, size=examples, seed=seed, max_op=max_op, max_edge=20,
        op_range=op_range, corruption_mode="iterative_refinement",
        trajectory_variants=1, refinement_probability=0.0,
        min_edits=1, max_edits=1, hash_bins=OFFICIAL_TEST_HASH_BINS,
    )


def means(episodes: list[dict], keys: tuple[str, ...]):
    return {
        key: sum(float(row[key]) for row in episodes) / max(len(episodes), 1)
        for key in keys
    }


@torch.no_grad()
def evaluate_mdlm(model, vocab, dataset, device: str):
    episodes = []
    for index in range(len(dataset)):
        item = dataset[index]
        prompt = _chunks_tensor(item["prompt"], vocab.pad_id, torch.device(device))
        initial = _chunks_tensor(
            item["buffers"][0], vocab.pad_id, torch.device(device)
        )
        sampled, _, response = model.sample(
            prompt, initial, schedule="confidence"
        )
        target = _chunks_tensor(
            item["buffers"][-1], vocab.pad_id, torch.device(device)
        )
        clean, _, _ = model.pack_clean(prompt, target)
        correct = sampled.eq(clean)
        response_correct = correct[response]
        response_positions = response[0].nonzero(as_tuple=False).flatten()
        last = int(response_positions[-1])
        episode = {
            "exact_sequence": bool(correct[response].all()),
            "answer_final_token_correct": bool(correct[0, last]),
            "token_accuracy": float(response_correct.float().mean()),
            "normalized_token_error": float(1.0 - response_correct.float().mean()),
            "target_tokens": int(response.sum()),
            "network_evaluations": int(response.sum()),
        }
        if index < 2:
            episode["generated_text"] = vocab.decode(
                sampled[0, response[0]].tolist()
            )
            episode["target_text"] = vocab.decode(clean[0, response[0]].tolist())
        episodes.append(episode)
    summary = means(
        episodes,
        ("exact_sequence", "answer_final_token_correct", "token_accuracy",
         "normalized_token_error", "target_tokens", "network_evaluations"),
    )
    return summary, episodes


@torch.no_grad()
def initial_candidate_diagnostics(planner, prompt, current, target):
    """Measure proposal coverage and GAR ranking without changing generation.

    ``proposal_correct_action_recall`` uses the learned deployment proposal
    set.  ``oracle_candidate_*`` deliberately inserts one gold action among
    otherwise incorrect learned proposals and is candidate-privileged.  The
    inserted action is only scored and is never returned to the planner.
    """
    state = planner.encode(prompt, current)
    proposed = planner.candidates(state, prompt, current, allow_refinement=False)
    gold = flatten(target)

    def correct(action):
        return (
            action[0] == "replace"
            and 0 <= action[1] < len(gold)
            and action[2] == gold[action[1]]
        )

    result = {
        "proposal_count": len(proposed),
        "proposal_correct_action_recall": float(
            any(correct(action) for action, _ in proposed)
        ),
        "oracle_candidate_available": 0.0,
    }
    current_flat = flatten(current)
    unresolved = [
        index for index, token in enumerate(current_flat)
        if token == planner.mask_id
    ]
    wrong = [action for action, _ in proposed if not correct(action)]
    if not unresolved or not wrong:
        return result
    position = unresolved[0]
    oracle = ("replace", position, gold[position])
    actions = [oracle, *wrong]
    codes = planner._action_codes(state, actions)
    q_codes = planner._value_action_codes(state, actions, codes)
    q = planner.model.action_value(
        state.global_state.expand(len(actions), -1), q_codes
    )
    result.update({
        "oracle_candidate_available": 1.0,
        "oracle_candidate_top1_accuracy": float(q[0] >= q[1:].max()),
        "oracle_candidate_pairwise_accuracy": float(
            (q[0] > q[1:]).float().mean()
        ),
        "oracle_candidate_margin": float(q[0] - q[1:].max()),
    })
    return result


@torch.no_grad()
def evaluate_jepa(
    model, vocab, dataset, device: str, *, scoring: str = "prior",
    horizon: int = 1, candidate_budget: int = 1, beam_width: int = 1,
):
    planner = make_jepa_planner(
        model, vocab, device, scoring, candidate_budget, beam_width
    )
    mask_id = vocab.token_to_id["<mask>"]
    episodes = []
    for index in range(len(dataset)):
        item = dataset[index]
        current = [list(sentence) for sentence in item["buffers"][0]]
        target = [list(sentence) for sentence in item["buffers"][-1]]
        diagnostic = initial_candidate_diagnostics(
            planner, item["prompt"], current, target
        )
        budget = sum(token == mask_id for token in flatten(current))
        stopped = "budget"
        for _ in range(budget):
            action, _, _ = planner.first_action(
                item["prompt"], current, horizon=horizon
            )
            if action is None:
                stopped = "no_candidates"
                break
            from textjepa.data.faithful_token_edits import _apply
            _apply(current, action)
        generated = flatten(current)
        gold = flatten(target)
        correct = sum(int(a == b) for a, b in zip(generated, gold))
        episode = {
            "exact_sequence": current == target,
            "exact_final_sentence": bool(current and current[-1] == target[-1]),
            "answer_final_token_correct": bool(
                current and target and current[-1] and target[-1]
                and current[-1][-1] == target[-1][-1]
            ),
            "token_accuracy": correct / max(len(gold), 1),
            "normalized_token_error": 1.0 - correct / max(len(gold), 1),
            "target_tokens": len(gold),
            "committed_actions": budget,
            "stop_reason": stopped,
            "initial_candidate_diagnostics": diagnostic,
        }
        if index < 2:
            episode["generated_text"] = [vocab.decode(x) for x in current]
            episode["target_text"] = [vocab.decode(x) for x in target]
        episodes.append(episode)
    summary = means(
        episodes,
        ("exact_sequence", "exact_final_sentence",
         "answer_final_token_correct", "token_accuracy",
         "normalized_token_error", "target_tokens", "committed_actions"),
    )
    summary["no_candidate_rate"] = sum(
        row["stop_reason"] == "no_candidates" for row in episodes
    ) / max(len(episodes), 1)
    diagnostics = [
        row["initial_candidate_diagnostics"] for row in episodes
    ]
    summary["initial_proposal_count"] = sum(
        row["proposal_count"] for row in diagnostics
    ) / max(len(diagnostics), 1)
    summary["initial_proposal_correct_action_recall"] = sum(
        row["proposal_correct_action_recall"] for row in diagnostics
    ) / max(len(diagnostics), 1)
    oracle = [
        row for row in diagnostics if row["oracle_candidate_available"]
    ]
    summary["oracle_candidate_diagnostic_count"] = len(oracle)
    for key in (
        "oracle_candidate_top1_accuracy",
        "oracle_candidate_pairwise_accuracy",
        "oracle_candidate_margin",
    ):
        summary[key] = (
            sum(row[key] for row in oracle) / len(oracle) if oracle else None
        )
    return summary, episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("mdlm", "jepa"), required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples-per-cell", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8600)
    parser.add_argument(
        "--planner-scoring", choices=("prior", "gar"), default="prior"
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--candidate-budget", type=int, default=1)
    parser.add_argument("--beam-width", type=int, default=1)
    args = parser.parse_args()

    if args.method == "mdlm":
        model, vocab = load_mdlm(args.ckpt, args.device)
        variant = "mdlm"
        evaluator = evaluate_mdlm
    else:
        model, vocab, _ = load_run(args.ckpt, args.device)
        variant = model.variant
        evaluator = lambda model, vocab, dataset, device: evaluate_jepa(
            model, vocab, dataset, device,
            scoring=args.planner_scoring, horizon=args.horizon,
            candidate_budget=args.candidate_budget,
            beam_width=args.beam_width,
        )

    cells = {}
    for offset, name in enumerate(PAPER_MEDIUM_CELLS):
        dataset = make_dataset(
            vocab, name, args.examples_per_cell, args.seed + 101 * offset
        )
        summary, episodes = evaluator(model, vocab, dataset, args.device)
        operation_counts = [
            int(dataset.source.problem(index)[0].p.n_op)
            for index in range(len(dataset))
        ]
        cells[name] = {
            "requested_op_range": list(PAPER_MEDIUM_CELLS[name][1]),
            "observed_op_range": [
                min(operation_counts), max(operation_counts)
            ],
            "summary": summary,
            "episodes": episodes,
        }
        print(json.dumps({"cell": name, **summary}, sort_keys=True))

    payload = {
        "method": args.method,
        "variant": variant,
        "benchmark": "original_iGSM_medium_length_generalization",
        "paper_protocol": {
            "training_support": "op <= 15",
            "id_cells": ["op <= 15", "op = 15"],
            "length_ood_cells": ["op = 20", "op = 21", "op = 22", "op = 23"],
            "official_test_hash_bins": list(OFFICIAL_TEST_HASH_BINS),
            "paper_examples_per_reported_cell": 4096,
            "this_pilot_examples_per_cell": args.examples_per_cell,
        },
        "validity": {
            "length_ood": True,
            "fresh_problems": True,
            "template_hash_holdout": False,
            "template_hash_holdout_reason": (
                "the evaluated checkpoints were trained by the prior adapter "
                "with all 23 official hash bins"
            ),
            "structure_known": True,
            "clean_target_visible_to_generator": False,
            "jepa_scoring": (
                {
                    "mode": args.planner_scoring,
                    "receding_horizon": args.horizon,
                    "candidate_budget": args.candidate_budget,
                    "beam_width": args.beam_width,
                    "proposal_source": "learned base content prior",
                    "position_source": (
                        "structurally enumerated unresolved token slots"
                    ),
                    "oracle_candidate_privileged_diagnostic": True,
                    "oracle_candidate_used_for_generation": False,
                } if args.method == "jepa" else None
            ),
        },
        "cells": cells,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
