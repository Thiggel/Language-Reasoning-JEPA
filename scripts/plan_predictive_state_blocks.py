#!/usr/bin/env python3
"""Stage 3 exact-state gate and recurrent latent block beam search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch
from transformers import AutoTokenizer

from textjepa.models.action_transition import (
    ResidualCapture,
    parameter_free_rms_norm,
)
from textjepa.planning.predictive_state import (
    DirectValueModel,
    GoalDistanceModel,
    candidate_score,
)
from textjepa.training.predictive_state import (
    UpperStackRunner,
    load_stage1_checkpoint,
    load_stage2_checkpoint,
    teacher_forward,
    transition_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--transition-checkpoint", type=Path, required=True)
    parser.add_argument("--goal-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-mode", choices=("exact", "jump"), required=True)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--candidates-per-beam", type=int, default=4)
    parser.add_argument("--chunk-length", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--max-problems", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def load_transition(path, device, dtype):
    header = torch.load(path, map_location="cpu", weights_only=True)
    if header.get("kind") == "action_conditioned_cross_layer_stage2":
        return load_stage2_checkpoint(path, device=device, dtype=dtype)
    return load_stage1_checkpoint(path, device=device, dtype=dtype)


def sample_top_p(logits, temperature, top_p):
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_prob, sorted_index = probabilities.sort(descending=True, dim=-1)
    cumulative = sorted_prob.cumsum(-1)
    sorted_prob = sorted_prob.masked_fill(cumulative - sorted_prob >= top_p, 0)
    sorted_prob = sorted_prob / sorted_prob.sum(-1, keepdim=True)
    sample = torch.multinomial(sorted_prob, 1)
    token = sorted_index.gather(-1, sample)
    log_probability = probabilities.gather(-1, token).clamp_min(1e-30).log()
    return token, log_probability.squeeze(-1)


@torch.no_grad()
def generate_candidates(model, prefix, *, count, length, temperature, top_p):
    sequences = prefix.repeat(count, 1)
    log_sum = torch.zeros(count, device=prefix.device)
    chunks = []
    for _ in range(length):
        logits = model(sequences, use_cache=False, return_dict=True).logits[:, -1]
        token, log_probability = sample_top_p(logits, temperature, top_p)
        sequences = torch.cat([sequences, token], dim=1)
        chunks.append(token[:, 0])
        log_sum += log_probability
    return torch.stack(chunks, dim=1), log_sum


def fused_source(states, source_layers, position=-1):
    return torch.cat([
        parameter_free_rms_norm(states[layer][:, position])
        for layer in source_layers
    ], dim=-1)


@torch.no_grad()
def exact_leaf(model, capture, tokens, sources):
    output = teacher_forward(
        model, tokens, capture=capture, attention_mask=None, use_cache=False
    )
    return fused_source(output.states, sources)


@torch.no_grad()
def jump_leaf(model, predictor, capture, prompt, suffix):
    exact = teacher_forward(
        model, prompt, capture=capture, attention_mask=None, use_cache=True
    )
    sources = {
        layer: exact.states[layer][:, -1:]
        for layer in predictor.config.used_source_layers
    }
    runner = UpperStackRunner(
        model, target_layer=predictor.config.target_layer,
        source_layers=predictor.config.source_layers,
    )
    for offset in range(suffix.shape[1]):
        action = suffix[:, offset:offset + 1]
        predicted = transition_prediction(predictor, model, sources, action)
        sources, _ = runner.step(
            predicted, past_key_values=exact.past_key_values,
            position_index=prompt.shape[1] + offset,
        )
    return fused_source(sources, predictor.config.source_layers, position=-1)


def extract_number(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\s*\{\s*([^{}]+)\s*\}", text)
    if boxed:
        return boxed[-1].strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if min(args.beam_width, args.candidates_per_beam,
           args.chunk_length, args.depth) < 1:
        raise ValueError("beam settings must be positive")
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    model, predictor, transition_payload = load_transition(
        args.transition_checkpoint, args.device, dtype
    )
    if predictor is None:
        raise ValueError("planning needs a transition predictor")
    model.eval()
    predictor.eval()
    model_id = transition_payload.get(
        "model_id", transition_payload.get("stage1_payload", {}).get("model_id")
    )
    revision = transition_payload.get(
        "model_revision",
        transition_payload.get("stage1_payload", {}).get("model_revision"),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    goal_payload = torch.load(args.goal_checkpoint, map_location="cpu", weights_only=True)
    goal_model = GoalDistanceModel(
        int(goal_payload["state_size"]), int(goal_payload["geometry_size"])
    ).to(args.device)
    value_model = DirectValueModel(int(goal_payload["state_size"])).to(args.device)
    goal_model.load_state_dict(goal_payload["distance_model"])
    value_model.load_state_dict(goal_payload["value_model"])
    goal_model.eval()
    value_model.eval()
    source_layers = predictor.config.source_layers
    if goal_payload["state_size"] != len(source_layers) * predictor.config.hidden_size:
        raise ValueError("goal scorer and transition state interfaces differ")
    capture = ResidualCapture(
        model, set(source_layers) | {predictor.config.target_layer}
    )
    problems = [json.loads(line) for line in args.input.read_text(
        encoding="utf-8"
    ).splitlines()[:args.max_problems]]
    results = []
    for problem_index, problem in enumerate(problems):
        prompt_text = str(problem["prompt"])
        prompt = tokenizer(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(args.device)
        prompt_output = teacher_forward(
            model, prompt, capture=capture, attention_mask=None, use_cache=False
        )
        prompt_state = fused_source(prompt_output.states, source_layers)
        beams = [{
            "tokens": prompt,
            "suffix": prompt.new_empty((1, 0)),
            "log_sum": 0.0,
            "score": 0.0,
        }]
        depth_trace = []
        for depth_index in range(args.depth):
            candidates = []
            for beam in beams:
                chunks, log_prob = generate_candidates(
                    model, beam["tokens"], count=args.candidates_per_beam,
                    length=args.chunk_length, temperature=args.temperature,
                    top_p=args.top_p,
                )
                for candidate_index in range(args.candidates_per_beam):
                    chunk = chunks[candidate_index:candidate_index + 1]
                    tokens = torch.cat([beam["tokens"], chunk], dim=1)
                    suffix = torch.cat([beam["suffix"], chunk], dim=1)
                    if args.state_mode == "exact":
                        leaf = exact_leaf(model, capture, tokens, source_layers)
                    else:
                        leaf = jump_leaf(model, predictor, capture, prompt, suffix)
                    distance = goal_model(leaf, prompt_state)
                    remaining_budget = torch.tensor(
                        [(args.depth - depth_index - 1) / max(args.depth, 1)],
                        device=args.device,
                    )
                    value = value_model(leaf, prompt_state, remaining_budget)
                    cumulative = beam["log_sum"] + float(log_prob[candidate_index])
                    score = candidate_score(
                        log_probability_sum=torch.tensor([cumulative], device=args.device),
                        token_count=torch.tensor([suffix.shape[1]], device=args.device),
                        distance=distance, value_logit=value,
                        alpha=args.alpha, beta=args.beta, eta=args.eta,
                    )
                    candidates.append({
                        "tokens": tokens, "suffix": suffix,
                        "log_sum": cumulative, "score": float(score),
                        "distance": float(distance), "value_logit": float(value),
                    })
            candidates.sort(key=lambda row: row["score"], reverse=True)
            beams = candidates[:args.beam_width]
            depth_trace.append([{key: row[key] for key in (
                "score", "distance", "value_logit", "log_sum"
            )} for row in beams])
        samples = []
        for beam in beams:
            text = tokenizer.decode(beam["suffix"][0], skip_special_tokens=True)
            predicted = extract_number(text)
            target = None if problem.get("answer") is None else str(problem["answer"])
            samples.append({
                "text": text, "predicted_answer": predicted,
                "target_answer": target,
                "correct": None if target is None else predicted == target,
                "score": beam["score"], "distance": beam["distance"],
                "value_logit": beam["value_logit"],
            })
        results.append({
            "problem_id": str(problem.get("problem_id", problem_index)),
            "state_mode": args.state_mode,
            "depth_trace": depth_trace,
            "beams": samples,
            "top1_correct": samples[0]["correct"],
        })
    known = [row["top1_correct"] for row in results if row["top1_correct"] is not None]
    report = {
        "schema_version": 1,
        "kind": "predictive_state_block_beam",
        "state_mode": args.state_mode,
        "exact_state_gate": args.state_mode == "exact",
        "problems": len(results),
        "verified_top1_accuracy": (
            sum(known) / len(known) if known else None
        ),
        "settings": vars(args) | {
            "input": str(args.input),
            "transition_checkpoint": str(args.transition_checkpoint),
            "goal_checkpoint": str(args.goal_checkpoint),
            "output": str(args.output),
        },
        "results": results,
        "oracle_terminal_state_at_inference": False,
        "scorer_training_used_oracle_terminal_states": True,
        "scorer_training_used_candidate_privileged_outcomes": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "state_mode", "problems", "verified_top1_accuracy"
    )}), flush=True)
    capture.__exit__(None, None, None)


if __name__ == "__main__":
    main()
