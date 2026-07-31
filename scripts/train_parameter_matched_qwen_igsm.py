#!/usr/bin/env python3
"""Train exact-parameter-matched Qwen controls on collected iGSM tokens."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from time import perf_counter

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from textjepa.data.language_planning import (
    GENERATION_EOS_TOKEN_IDS,
    MODEL_ID,
    MODEL_REVISION,
    PAD_TOKEN_ID,
)
from textjepa.data.provenance import sha256_file
from textjepa.models.qwen_parameter_matched import (
    ADDED_NATIVE_LAYER_TYPES,
    TOKEN_JEPA_PARAMETER_BUDGET,
    append_parameter_matched_capacity,
    expose_exact_pretrained_budget,
    optimizer_groups,
    trainable_state_dict,
)
from textjepa.training.optim import cosine_warmup


CHECKPOINT_STEPS = (1_500, 3_000, 4_500, 6_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split-jsonl", type=Path, required=True)
    parser.add_argument("--eval-features", type=Path, action="append", default=[])
    parser.add_argument("--eval-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("added_capacity", "unfrozen"), required=True
    )
    parser.add_argument("--steps", type=int, default=6_000)
    parser.add_argument("--checkpoint-step", type=int, action="append")
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-eval-examples", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    return parser.parse_args()


def _load_features(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "input_ids", "attention_mask", "prompt_len", "solution_end",
        "problem_id", "dataset_fingerprint", "model_id", "model_revision",
    }
    if not required <= data.keys():
        raise ValueError("baseline requires canonical collected features")
    if data["model_id"] != MODEL_ID or data["model_revision"] != MODEL_REVISION:
        raise ValueError("feature backbone provenance is incompatible")
    return data


def _build_model(variant: str, dtype: torch.dtype, device: str):
    config = AutoConfig.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    original_layers = int(config.text_config.num_hidden_layers)
    if variant == "added_capacity":
        config.text_config.num_hidden_layers += len(ADDED_NATIVE_LAYER_TYPES)
        config.text_config.layer_types = (
            list(config.text_config.layer_types)
            + list(ADDED_NATIVE_LAYER_TYPES)
        )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        config=config,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    if variant == "added_capacity":
        masks = []
        accounting = append_parameter_matched_capacity(
            model, original_layer_count=original_layers
        )
    else:
        masks, accounting = expose_exact_pretrained_budget(model)
    if accounting["active_trainable_parameters"] != (
        TOKEN_JEPA_PARAMETER_BUDGET
    ):
        raise AssertionError("baseline parameter match failed")
    return model, masks, accounting


def _batch(data: dict, indices: torch.Tensor, device: str) -> dict:
    lengths = data["attention_mask"][indices].sum(-1)
    width = int(lengths.max())
    return {
        "input_ids": data["input_ids"][indices, :width].to(device),
        "attention_mask": data["attention_mask"][indices, :width].to(device),
        "prompt_len": data["prompt_len"][indices].to(device),
        "solution_end": data["solution_end"][indices].to(device),
    }


def _solution_loss(model, batch: dict) -> tuple[torch.Tensor, int]:
    output = model.model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    predictors, targets = [], []
    for row in range(len(batch["input_ids"])):
        start = int(batch["prompt_len"][row])
        end = int(batch["solution_end"][row])
        predictors.append(output.last_hidden_state[row, start - 1:end - 1])
        targets.append(batch["input_ids"][row, start:end])
    hidden = torch.cat(predictors)
    target = torch.cat(targets)
    logits = model.lm_head(hidden)
    return F.cross_entropy(logits.float(), target, reduction="mean"), len(target)


def _save_checkpoint(
    path: Path,
    model,
    *,
    variant: str,
    step: int,
    accounting: dict,
    features: dict,
    training: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": 1,
        "kind": "parameter_matched_qwen_igsm",
        "variant": variant,
        "step": step,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "parameter_accounting": accounting,
        "dataset_fingerprint": features["dataset_fingerprint"],
        "training": training,
        "trainable_state": trainable_state_dict(model),
    }, path)


def _load_trainable(model, state: dict[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    if set(state) - set(parameters):
        raise ValueError("checkpoint contains unknown trainable tensors")
    with torch.no_grad():
        for name, value in state.items():
            if parameters[name].shape != value.shape:
                raise ValueError(f"checkpoint tensor shape mismatch: {name}")
            parameters[name].copy_(
                value.to(parameters[name].device, parameters[name].dtype)
            )


def _answers(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        result[str(record["problem_id"])] = str(record["answer"])
    return result


def _extract_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\s*\{\s*(-?\d+)\s*\}", text)
    if boxed:
        return boxed[-1]
    numbers = re.findall(r"-?\d+", text)
    return numbers[-1] if numbers else None


@torch.no_grad()
def _evaluate(
    model,
    tokenizer,
    features: dict,
    answers: dict[str, str],
    *,
    maximum: int,
    max_new_tokens: int,
    device: str,
) -> dict:
    count = min(len(features["input_ids"]), maximum)
    correct = 0
    generated_tokens = 0
    examples = []
    model.eval()
    for row in range(count):
        prompt_len = int(features["prompt_len"][row])
        prompt = features["input_ids"][row, :prompt_len].to(device)[None]
        output = model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt, dtype=torch.bool),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=list(GENERATION_EOS_TOKEN_IDS),
            pad_token_id=PAD_TOKEN_ID,
            use_cache=True,
        )
        suffix = output[0, prompt_len:]
        text = tokenizer.decode(suffix, skip_special_tokens=True)
        predicted = _extract_answer(text)
        problem_id = str(features["problem_id"][row])
        target = answers[problem_id]
        hit = predicted == target
        correct += int(hit)
        generated_tokens += len(suffix)
        if len(examples) < 8:
            examples.append({
                "problem_id": problem_id,
                "predicted": predicted,
                "target": target,
                "correct": hit,
                "text": text,
            })
    return {
        "examples": count,
        "accuracy": correct / max(count, 1),
        "correct": correct,
        "generated_tokens": generated_tokens,
        "samples": examples,
    }


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.microbatch_size < 1 or (
        args.gradient_accumulation < 1
    ):
        raise ValueError("training steps and batch sizes must be positive")
    if len(args.eval_features) != len(args.eval_jsonl):
        raise ValueError("--eval-features and --eval-jsonl must align")
    dtype = {
        "bfloat16": torch.bfloat16, "float16": torch.float16
    }[args.dtype]
    torch.manual_seed(args.seed)
    features = _load_features(args.features)
    model, masks, accounting = _build_model(
        args.variant, dtype, args.device
    )
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 1e-3 if args.variant == "added_capacity" else 1e-5
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, masks, weight_decay=args.weight_decay),
        lr=learning_rate,
        betas=(0.9, 0.95),
    )
    warmup = max(1, round(args.steps * args.warmup_fraction))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup(step, args.steps, warmup),
    )
    checkpoint_steps = sorted(set(
        args.checkpoint_step or [
            step for step in CHECKPOINT_STEPS if step <= args.steps
        ] + [args.steps]
    ))
    sample_generator = torch.Generator().manual_seed(args.seed + 913)
    sample_count = len(features["input_ids"])
    history, token_count = [], 0
    started = perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        step_tokens = 0
        for _ in range(args.gradient_accumulation):
            indices = torch.randint(
                sample_count, (args.microbatch_size,),
                generator=sample_generator,
            )
            loss, tokens = _solution_loss(
                model, _batch(features, indices, args.device)
            )
            (loss / args.gradient_accumulation).backward()
            loss_sum += float(loss.detach())
            step_tokens += tokens
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()
        token_count += step_tokens
        row = {
            "step": step,
            "loss": loss_sum / args.gradient_accumulation,
            "lr": scheduler.get_last_lr()[0],
            "solution_tokens": step_tokens,
            "elapsed_seconds": perf_counter() - started,
        }
        if step == 1 or step % 100 == 0:
            history.append(row)
            print(json.dumps(row), flush=True)
        if step in checkpoint_steps:
            training = {
                "steps": args.steps,
                "learning_rate": learning_rate,
                "warmup_steps": warmup,
                "weight_decay": args.weight_decay,
                "microbatch_size": args.microbatch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "global_batch_size": (
                    args.microbatch_size * args.gradient_accumulation
                ),
                "seed": args.seed,
                "solution_tokens_seen": token_count,
            }
            _save_checkpoint(
                args.output / "checkpoints" / f"step_{step:06d}.pt",
                model, variant=args.variant, step=step,
                accounting=accounting, features=features, training=training,
            )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    evaluations = {}
    checkpoints = [
        args.output / "checkpoints" / f"step_{step:06d}.pt"
        for step in checkpoint_steps
    ]
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        _load_trainable(model, payload["trainable_state"])
        step_result = {}
        for feature_path, jsonl_path in zip(
            args.eval_features, args.eval_jsonl
        ):
            split_features = _load_features(feature_path)
            split_name = feature_path.stem
            step_result[split_name] = _evaluate(
                model, tokenizer, split_features, _answers(jsonl_path),
                maximum=args.max_eval_examples,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )
        evaluations[str(payload["step"])] = step_result
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "variant": args.variant,
        "parameter_accounting": accounting,
        "training": {
            "steps": args.steps,
            "checkpoint_steps": checkpoint_steps,
            "learning_rate": learning_rate,
            "warmup_steps": warmup,
            "global_batch_size": (
                args.microbatch_size * args.gradient_accumulation
            ),
            "solution_tokens_seen": token_count,
            "wall_seconds": perf_counter() - started,
        },
        "history": history,
        "evaluations": evaluations,
        "feature_sha256": sha256_file(args.features),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
