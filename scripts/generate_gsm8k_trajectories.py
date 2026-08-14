#!/usr/bin/env python3
"""Sample GSM8K solutions and label each one with an external verifier.

Stage 3A needs trajectories whose correctness is known before any latent state
is looked at. Correctness here comes from exact numeric agreement with the
dataset's own gold answer, never from anything the model or a probe produces,
and the resulting labels are candidate-privileged information: they may be used
to build and evaluate a geometry, never as an inference-time signal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


INSTRUCT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FINAL_ANSWER = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
LAST_NUMBER = re.compile(r"(-?[\d,]+(?:\.\d+)?)")
INSTRUCTION = (
    "Solve the problem step by step. Put each step on its own line. "
    "End with a final line of the form '#### <answer>'."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=INSTRUCT_MODEL_ID)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-problems", type=int, default=300)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"),
                        default="bfloat16")
    return parser.parse_args()


def normalize(value: str) -> str | None:
    try:
        return f"{float(value.replace(',', '')):.6f}"
    except ValueError:
        return None


def gold_answer(answer: str) -> str | None:
    match = FINAL_ANSWER.search(answer)
    return normalize(match.group(1)) if match else None


def predicted_answer(text: str) -> str | None:
    """Prefer the requested marker; fall back to the last number produced."""
    match = FINAL_ANSWER.search(text)
    if match:
        return normalize(match.group(1))
    numbers = LAST_NUMBER.findall(text)
    return normalize(numbers[-1]) if numbers else None


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    problems = []
    with args.problems.open(encoding="utf-8") as handle:
        for line in handle:
            problems.append(json.loads(line))
            if len(problems) >= args.max_problems:
                break

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.model_revision, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, revision=args.model_revision, dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    requests = []
    for problem in problems:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user",
              "content": f"{INSTRUCTION}\n\n{problem['question'].strip()}"}],
            tokenize=False, add_generation_prompt=True,
        )
        for sample in range(args.samples):
            requests.append((problem, prompt, sample))

    written = correct_count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for start in range(0, len(requests), args.batch_size):
            batch = requests[start:start + args.batch_size]
            encoded = tokenizer(
                [prompt for _, prompt, _ in batch], return_tensors="pt",
                padding=True, add_special_tokens=False,
            ).to(args.device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded, do_sample=True, temperature=args.temperature,
                    top_p=args.top_p, max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completions = tokenizer.batch_decode(
                generated[:, encoded["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            for (problem, prompt, sample), completion in zip(batch, completions):
                gold = gold_answer(problem["answer"])
                predicted = predicted_answer(completion)
                is_correct = gold is not None and predicted == gold
                correct_count += int(is_correct)
                out.write(json.dumps({
                    "problem_id": problem["problem_id"],
                    "sample": sample,
                    "prompt": prompt,
                    "trajectory": completion.strip(),
                    "correct": bool(is_correct),
                    "gold_answer": gold,
                    "predicted_answer": predicted,
                    "verifier": "gsm8k_gold_exact_numeric",
                }) + "\n")
                written += 1
            print(json.dumps({
                "generated": written, "of": len(requests),
                "running_accuracy": correct_count / max(written, 1),
            }), flush=True)
    print(json.dumps({
        "status": "completed", "records": written,
        "accuracy": correct_count / max(written, 1),
        "problems": len(problems), "samples_per_problem": args.samples,
        "model_id": args.model_id, "model_revision": args.model_revision,
        "verifier": "gsm8k_gold_exact_numeric",
        "candidate_privileged_outcomes": True,
    }), flush=True)


if __name__ == "__main__":
    main()
