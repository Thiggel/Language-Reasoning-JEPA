"""Decompose token-policy failure under genuinely generated prefixes.

No symbolic feasibility filter or auxiliary language model is used.  Every
policy is restricted only to the state-conditioned token prior's top-k support.
The audit compares re-encoding the generated prefix with recursively rolling
the JEPA predictor, and compares prior-only, learned geometry/value, and an
explicitly labelled oracle-outcome ceiling.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.lm import LMDataset, collate_lm
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA


def normalized_l1(rows: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    return (
        F.layer_norm(rows, rows.shape[-1:])
        - F.layer_norm(goal, goal.shape[-1:])
    ).abs().mean(-1)


def standardize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def summarize(values: list[float]) -> dict[str, float]:
    finite = torch.tensor(values, dtype=torch.float)
    finite = finite[torch.isfinite(finite)]
    if not len(finite):
        return {"mean": float("nan"), "p90": float("nan"), "n": 0}
    return {
        "mean": float(finite.mean()),
        "p90": float(torch.quantile(finite, 0.9)),
        "n": int(len(finite)),
    }


def load_model(checkpoint: str, device: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    if model.token_prior is None:
        raise ValueError("closed-loop decomposition requires a token prior")
    return model, vocab, cfg


def candidate_predictions(model, current, states, actions, candidates):
    count = len(candidates)
    return model.low_predictor.rollout(
        current.expand(count, -1), model.token_action(candidates)[:, None],
        state_history=states.expand(count, -1, -1),
        action_history=actions.expand(count, -1, -1),
    )[:, 0]


def encoded_outcomes(encoder, prefix, candidates):
    rows = torch.cat([
        prefix.expand(len(candidates), -1), candidates[:, None]
    ], dim=1)
    return encoder(rows)[:, -1]


def policy_name(score: str, prior_weight: float) -> str:
    return score if score == "prior" else f"{score}_prior{prior_weight:g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--prior-weights", type=float, nargs="+", default=[0.1, 0.3, 1, 3, 10]
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model, vocab, cfg = load_model(args.ckpt, args.device)
    dataset = LMDataset(
        vocab, size=args.examples, seed=cfg.data.val_seed + 196613,
        modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range), leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    loader = DataLoader(
        dataset, batch_size=1,
        collate_fn=lambda rows: collate_lm(rows, pad_id=vocab.pad_id),
    )
    forbidden = {vocab.pad_id, vocab.token_to_id[vocab.UNK]}
    policies = [("prior", 1.0)]
    for score in ("predicted_geometry", "learned_value", "oracle_geometry"):
        policies.extend((score, weight) for weight in args.prior_weights)
    collected: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(args.device)
            prompt_len = batch["prompt_len"].to(args.device)
            reference = model(tokens, prompt_len)
            length = min(int(reference["valid"][0].sum()), args.max_steps)
            prompt_width = int(prompt_len[0])
            prompt = tokens[:, :prompt_width]
            goal = reference["final_target"]

            for state_mode in ("reencoded", "predictor"):
                for score_name, prior_weight in policies:
                    label = f"{state_mode}/{policy_name(score_name, prior_weight)}"
                    generated: list[int] = []
                    state_path = [reference["prompt_state"]]
                    exact_prefix = True
                    first_error = length + 1
                    for position in range(length):
                        prefix = torch.cat([
                            prompt,
                            torch.tensor(
                                generated, device=tokens.device, dtype=tokens.dtype
                            )[None],
                        ], dim=1)
                        reencoded = model.encoder(prefix)[:, -1]
                        current = reencoded if state_mode == "reencoded" else state_path[-1]
                        history = torch.stack(state_path, dim=1)
                        action_history = model.token_action(torch.tensor(
                            generated, device=tokens.device, dtype=tokens.dtype
                        ))[None]
                        logits = model.token_prior(current)[0]
                        logits[list(forbidden)] = -torch.inf
                        top = logits.topk(min(args.topk, len(logits) - len(forbidden)))
                        candidates = top.indices
                        prior_cost = -top.values.log_softmax(0)
                        true_id = int(reference["action_ids"][0, position])
                        in_support = bool(candidates.eq(true_id).any())
                        collected[f"{label}/reference_in_support"].append(float(in_support))

                        predictions = candidate_predictions(
                            model, current, history, action_history, candidates
                        )
                        if score_name == "prior":
                            cost = prior_cost
                        else:
                            if score_name == "predicted_geometry":
                                consequence = normalized_l1(
                                    predictions, goal.expand_as(predictions)
                                )
                            elif score_name == "learned_value":
                                consequence = model.low_goal_value(
                                    predictions, goal.expand_as(predictions)
                                )
                            else:
                                outcomes = encoded_outcomes(
                                    model.teacher, prefix, candidates
                                )
                                consequence = normalized_l1(
                                    outcomes, goal.expand_as(outcomes)
                                )
                            cost = standardize(consequence) + prior_weight * standardize(prior_cost)
                        selected_index = int(cost.argmin())
                        selected = int(candidates[selected_index])
                        correct = selected == true_id
                        collected[f"{label}/position_accuracy"].append(float(correct))
                        collected[f"{label}/exact_prefix_before"].append(float(exact_prefix))
                        exact_prefix = exact_prefix and correct
                        if not correct and first_error == length + 1:
                            first_error = position + 1

                        generated.append(selected)
                        next_prefix = torch.cat([
                            prefix,
                            torch.tensor([[selected]], device=tokens.device, dtype=tokens.dtype),
                        ], dim=1)
                        actual_next = model.encoder(next_prefix)[:, -1]
                        predicted_next = predictions[selected_index:selected_index + 1]
                        collected[f"{label}/selected_transition_drift"].append(float(
                            normalized_l1(predicted_next, actual_next)[0]
                        ))
                        state_path.append(
                            actual_next if state_mode == "reencoded" else predicted_next
                        )
                    collected[f"{label}/exact_trace"].append(float(exact_prefix))
                    collected[f"{label}/first_error_step"].append(float(first_error))

    result = {key: summarize(values) for key, values in sorted(collected.items())}
    result["metadata"] = {
        "examples": args.examples,
        "topk": args.topk,
        "uses_symbolic_feasibility": False,
        "uses_auxiliary_lm": False,
        "oracle_geometry": "candidate-privileged diagnostic ceiling",
        "reencoded": "online encoder applied to the actually generated prefix",
        "predictor": "causal JEPA recursively rolled under generated tokens",
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
