#!/usr/bin/env python3
"""Is text produced by the jump path actually any good?

Token agreement with greedy decoding is a harsh and misleading measure: one
flipped token cascades, so a low score is compatible with fluent output. This
scores the generated text itself.

Each decoding mode continues the same prompts, and every continuation is judged
by the *original unmodified* checkpoint, which never took part in training or
decoding. Repetition is reported too, since collapse into a loop is the
characteristic failure of a degraded decoder and perplexity alone rewards it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from textjepa.data.predictive_state import load_token_blocks
from textjepa.models.action_transition import ResidualCapture
from textjepa.training.predictive_state import (
    load_stage1_checkpoint,
    load_stage2_checkpoint,
    require_transformers_runtime,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-blocks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--prompt-length", type=int, default=256)
    parser.add_argument("--generate", type=int, default=128)
    parser.add_argument("--refresh-interval", type=int, action="append")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"),
                        default="bfloat16")
    return parser.parse_args()


def distinct_ngrams(tokens: torch.Tensor, order: int) -> float:
    """Share of n-grams that are unique. Low means the decoder is looping."""
    rows = []
    for row in tokens:
        grams = {
            tuple(row[i:i + order].tolist())
            for i in range(len(row) - order + 1)
        }
        rows.append(len(grams) / max(len(row) - order + 1, 1))
    return sum(rows) / max(len(rows), 1)


@torch.no_grad()
def judge_nll(judge, prompts: torch.Tensor, continuations: torch.Tensor) -> float:
    """Mean NLL of the continuation under an independent judge model."""
    sequence = torch.cat([prompts, continuations], dim=1)
    logits = judge(sequence, use_cache=False, return_dict=True).logits.float()
    begin = prompts.shape[1]
    log_probs = torch.log_softmax(logits[:, begin - 1:-1], dim=-1)
    picked = log_probs.gather(-1, continuations[:, :, None]).squeeze(-1)
    return float(-picked.mean())


def main() -> None:
    args = parse_args()
    require_transformers_runtime()
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    header = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    loader = (
        load_stage2_checkpoint
        if header.get("kind") == "action_conditioned_cross_layer_stage2"
        else load_stage1_checkpoint
    )
    model, predictor, payload = loader(
        args.checkpoint, device=args.device, dtype=dtype
    )
    model.eval()
    stage1 = payload.get("stage1_payload", payload)
    model_id, revision = stage1["model_id"], stage1["model_revision"]

    judge = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=dtype, low_cpu_mem_usage=True,
    ).to(args.device)
    judge.requires_grad_(False)
    judge.eval()

    dataset, _ = load_token_blocks(args.token_blocks, "validation")
    generator = torch.Generator().manual_seed(args.seed)
    rows = torch.randperm(len(dataset), generator=generator)[:args.prompts]
    prompts = torch.stack([
        dataset[int(row)]["input_ids"][:args.prompt_length] for row in rows
    ]).to(args.device)

    from evaluate_action_transition_rollout import (  # noqa: E402
        full_greedy_decode, jump_greedy_decode,
    )

    intervals = sorted(set(args.refresh_interval or [4, 16, 64]))
    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "prompts": int(prompts.shape[0]),
        "prompt_length": int(prompts.shape[1]),
        "generated_tokens": args.generate,
        "judge": f"{model_id}@{revision} (original, unmodified)",
        "modes": {},
    }

    with ResidualCapture(model, set(predictor.config.used_source_layers)
                         | {predictor.config.target_layer}) as capture:
        modes = [("full", "full")] + [("jump_only", None)] + [
            (f"refresh_{n}", n) for n in intervals
        ]
        for label, setting in modes:
            outputs = []
            for index in range(prompts.shape[0]):
                one = prompts[index:index + 1]
                if setting == "full":
                    outputs.append(full_greedy_decode(model, one, args.generate))
                else:
                    generated, _ = jump_greedy_decode(
                        model, predictor, capture, one, args.generate,
                        refresh_interval=setting,
                    )
                    outputs.append(generated)
            continuations = torch.cat(outputs, dim=0)
            report["modes"][label] = {
                "judge_nll": judge_nll(judge, prompts, continuations),
                "distinct_3": distinct_ngrams(continuations.cpu(), 3),
                "distinct_5": distinct_ngrams(continuations.cpu(), 5),
            }
            print(json.dumps({label: report["modes"][label]}), flush=True)

    baseline = report["modes"]["full"]["judge_nll"]
    for label, values in report["modes"].items():
        values["excess_judge_nll_vs_full"] = values["judge_nll"] - baseline
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
