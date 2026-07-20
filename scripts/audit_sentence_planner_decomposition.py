"""First-sentence oracle decomposition for the two-level planner interface."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from eval_sentence_hierarchy_planning import (
    build_codebook, categorical_cem, encode_generated, encode_macro_sentences,
    execute_low_level, normalized_distance,
)
from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.semantic_lm import SemanticBoundaryLMDataset
from textjepa.models.sentence_hierarchy import SentenceHierarchyJEPA


def token_accuracy(left, right):
    width = max(len(left), len(right), 1)
    return sum(a == b for a, b in zip(left, right)) / width


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=12)
    parser.add_argument("--codebook-examples", type=int, default=256)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    torch.manual_seed(int(cfg.seed) + 9187)
    vocab = build_vocab(cfg.data.modulus)
    model = SentenceHierarchyJEPA(len(vocab), vocab.pad_id, **cfg.model).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    dataset = SemanticBoundaryLMDataset(
        vocab, size=max(args.examples, args.codebook_examples),
        seed=cfg.data.val_seed + 104729, boundary_mode="semantic",
        modulus=cfg.data.modulus, n_vars_range=tuple(cfg.data.n_vars_range),
        leaf_prob=cfg.data.leaf_prob, steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    codebook, sentences = build_codebook(
        model, dataset, args.codebook_examples, vocab.pad_id
    )
    period = vocab.token_to_id["."]
    values = {key: [] for key in (
        "goal_prediction_mse", "oracle_macro_dynamics_mse",
        "oracle_subgoal_low_exact", "oracle_subgoal_low_token_accuracy",
        "predicted_oracle_macro_low_exact", "predicted_oracle_macro_low_token_accuracy",
        "learned_goal_macro_exact", "oracle_goal_macro_exact",
        "learned_goal_macro_token_accuracy", "oracle_goal_macro_token_accuracy",
        "learned_goal_subgoal_mse", "oracle_goal_subgoal_mse",
        "h1_h2_same_code", "h2_h4_same_code", "horizon_first_token_agreement",
        "horizon_first_token_agreement_no_prior", "macro_outcome_variance",
        "reference_sentence_in_codebook", "predicted_subgoal_retrieval_exact",
        "predicted_subgoal_retrieval_token_accuracy",
        "oracle_subgoal_low_exact_prior0", "oracle_subgoal_low_accuracy_prior0",
        "oracle_subgoal_low_exact_prior01", "oracle_subgoal_low_accuracy_prior01",
        "oracle_subgoal_low_exact_full_vocab", "oracle_subgoal_low_accuracy_full_vocab",
    )}
    rows = []
    for index in range(args.examples):
        item = dataset[index]
        prompt_len = int(item["prompt_len"])
        prompt = item["tokens"][:prompt_len]
        first_end = int(item["sentence_ends"][0])
        reference = item["tokens"][prompt_len:prompt_len + first_end]
        full = torch.tensor(item["tokens"], device=args.device).unsqueeze(0)
        high_targets = model.high_teacher(full)
        oracle_goal = high_targets[:, len(item["tokens"]) - 1]
        oracle_subgoal = high_targets[:, prompt_len + first_end - 1]
        context = encode_generated(model, prompt, prompt_len, period)
        values["goal_prediction_mse"].append(float(normalized_distance(
            context["high_goal"], oracle_goal
        )))
        reference_code = encode_macro_sentences(
            model, [reference], vocab.pad_id
        )
        predicted_reference_subgoal = model.high_predictor.rollout(
            context["high_state"], reference_code[:, None],
            state_history=context["high_history"],
            action_history=context["high_action_history"],
        )[:, 0]
        values["oracle_macro_dynamics_mse"].append(float(normalized_distance(
            predicted_reference_subgoal, oracle_subgoal
        )))
        common = dict(
            candidates=256, updates=10, elite=32, pool_size=64,
            smoothing=0.25, goal_weight=1.0, value_weight=1.0,
            prior_weight=0.1, support_weight=1.0, reachability_weight=1.0,
            pool_filter="prior",
        )
        proposals = {
            horizon: categorical_cem(
                model, context, codebook, horizon=horizon, **common
            ) for horizon in (1, 2, 4)
        }
        oracle_context = dict(context)
        oracle_context["high_goal"] = oracle_goal
        oracle_proposal = categorical_cem(
            model, oracle_context, codebook, horizon=2, **common
        )
        learned_index = int(proposals[2]["selected_codebook_index"])
        oracle_index = int(oracle_proposal["selected_codebook_index"])
        learned_sentence, oracle_sentence = sentences[learned_index], sentences[oracle_index]
        values["learned_goal_macro_exact"].append(learned_sentence == reference)
        values["oracle_goal_macro_exact"].append(oracle_sentence == reference)
        values["learned_goal_macro_token_accuracy"].append(token_accuracy(
            learned_sentence, reference
        ))
        values["oracle_goal_macro_token_accuracy"].append(token_accuracy(
            oracle_sentence, reference
        ))
        values["learned_goal_subgoal_mse"].append(float(normalized_distance(
            proposals[2]["subgoal"], oracle_subgoal
        )))
        values["oracle_goal_subgoal_mse"].append(float(normalized_distance(
            oracle_proposal["subgoal"], oracle_subgoal
        )))
        chosen = {
            horizon: int(proposal["selected_codebook_index"])
            for horizon, proposal in proposals.items()
        }
        values["h1_h2_same_code"].append(chosen[1] == chosen[2])
        values["h2_h4_same_code"].append(chosen[2] == chosen[4])
        generated_by_horizon = {}
        generated_by_horizon_no_prior = {}
        for horizon, proposal in proposals.items():
            generated_by_horizon[horizon], _, _ = execute_low_level(
                model, prompt, prompt_len, period, proposal["subgoal"],
                token_topk=20, max_sentence_tokens=32, low_prior_weight=1.0,
            )
            generated_by_horizon[horizon] = generated_by_horizon[horizon][prompt_len:]
            generated_by_horizon_no_prior[horizon], _, _ = execute_low_level(
                model, prompt, prompt_len, period, proposal["subgoal"],
                token_topk=20, max_sentence_tokens=32, low_prior_weight=0.0,
            )
            generated_by_horizon_no_prior[horizon] = (
                generated_by_horizon_no_prior[horizon][prompt_len:]
            )
        firsts = [tokens[0] if tokens else -1 for tokens in generated_by_horizon.values()]
        values["horizon_first_token_agreement"].append(len(set(firsts)) == 1)
        firsts_no_prior = [
            tokens[0] if tokens else -1
            for tokens in generated_by_horizon_no_prior.values()
        ]
        values["horizon_first_token_agreement_no_prior"].append(
            len(set(firsts_no_prior)) == 1
        )
        oracle_generated, _, _ = execute_low_level(
            model, prompt, prompt_len, period, oracle_subgoal,
            token_topk=20, max_sentence_tokens=32, low_prior_weight=1.0,
        )
        predicted_generated, _, _ = execute_low_level(
            model, prompt, prompt_len, period, predicted_reference_subgoal,
            token_topk=20, max_sentence_tokens=32, low_prior_weight=1.0,
        )
        oracle_generated = oracle_generated[prompt_len:]
        predicted_generated = predicted_generated[prompt_len:]
        values["oracle_subgoal_low_exact"].append(oracle_generated == reference)
        values["oracle_subgoal_low_token_accuracy"].append(token_accuracy(
            oracle_generated, reference
        ))
        for suffix, prior_weight, topk in (
            ("prior0", 0.0, 20), ("prior01", 0.1, 20),
            ("full_vocab", 0.0, len(vocab) - 1),
        ):
            alternative, _, _ = execute_low_level(
                model, prompt, prompt_len, period, oracle_subgoal,
                token_topk=topk, max_sentence_tokens=32,
                low_prior_weight=prior_weight,
            )
            alternative = alternative[prompt_len:]
            values[f"oracle_subgoal_low_exact_{suffix}"].append(
                alternative == reference
            )
            values[f"oracle_subgoal_low_accuracy_{suffix}"].append(
                token_accuracy(alternative, reference)
            )
        values["predicted_oracle_macro_low_exact"].append(
            predicted_generated == reference
        )
        values["predicted_oracle_macro_low_token_accuracy"].append(token_accuracy(
            predicted_generated, reference
        ))
        sample_codes = codebook[:min(64, len(codebook))]
        predicted = model.high_predictor.rollout(
            context["high_state"].expand(len(sample_codes), -1),
            sample_codes[:, None],
            state_history=context["high_history"].expand(len(sample_codes), -1, -1),
            action_history=context["high_action_history"].expand(len(sample_codes), -1, -1),
        )[:, 0]
        values["macro_outcome_variance"].append(float(predicted.var(0).mean()))
        try:
            reference_index = sentences.index(reference)
        except ValueError:
            reference_index = -1
        values["reference_sentence_in_codebook"].append(reference_index >= 0)
        # This is the clean high-dynamics retrieval test: every observed
        # sentence code is available and candidates are ranked only by the
        # predicted distance to the exact reachable next-sentence target.
        predicted_chunks = []
        for start in range(0, len(codebook), 256):
            codes = codebook[start:start + 256]
            predicted_chunks.append(model.high_predictor.rollout(
                context["high_state"].expand(len(codes), -1), codes[:, None],
                state_history=context["high_history"].expand(len(codes), -1, -1),
                action_history=context["high_action_history"].expand(len(codes), -1, -1),
            )[:, 0])
        all_predicted = torch.cat(predicted_chunks)
        retrieved_index = int(normalized_distance(
            all_predicted, oracle_subgoal.expand(len(all_predicted), -1)
        ).argmin())
        retrieved = sentences[retrieved_index]
        values["predicted_subgoal_retrieval_exact"].append(retrieved == reference)
        values["predicted_subgoal_retrieval_token_accuracy"].append(
            token_accuracy(retrieved, reference)
        )
        rows.append({
            "episode": index, "reference_tokens": len(reference),
            "selected_code_h1": chosen[1], "selected_code_h2": chosen[2],
            "selected_code_h4": chosen[4],
            "learned_goal_code": learned_index, "oracle_goal_code": oracle_index,
            "reference_code": reference_index,
            "predicted_subgoal_retrieval_code": retrieved_index,
        })
    result = {
        key: float(np.mean(samples)) for key, samples in values.items()
    }
    result.update(
        examples=args.examples, codebook_size=len(codebook), episodes=rows,
        uses_symbolic_feasibility=False, uses_auxiliary_lm=False,
        oracle_fields=[
            "oracle_goal_macro_*", "oracle_subgoal_*", "oracle_macro_dynamics_mse"
        ],
    )
    destination = Path(args.output) if args.output else (
        Path(os.environ["RUN_DIR"]) / "sentence_planner_decomposition.json"
        if os.environ.get("RUN_DIR")
        else Path(args.ckpt).parent / "sentence_planner_decomposition.json"
    )
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
