#!/usr/bin/env python3
"""Sample counterfactual continuations at each step boundary of a gold solution.

The energy head is trained to rank the continuation that actually occurred above
continuations that did not. "Which continuation occurred" is in the dataset, so
this is self-supervised: no verifier judges the model's output, and no step
counter is used anywhere.

For every prefix of a gold solution, this writes the gold next step together
with several sampled alternatives drawn at a spread of temperatures. Whether a
sampled alternative happens to be correct is never labelled or used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


INSTRUCT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
INSTRUCTION = (
    "Solve the problem step by step. Put each step on its own line. "
    "End with a final line of the form '#### <answer>'."
)
# Observed plus greedy plus a temperature spread, following the sampling policy
# already used elsewhere in the repository.
SAMPLING_POLICY = (0.0, 0.5, 0.5, 0.8, 0.8, 1.0, 1.2)
# GSM8K gold steps carry `<<48/2=24>>` calculator annotations that model
# output never contains. Left in, an energy head could separate observed
# from sampled continuations on that formatting alone and learn nothing
# about reasoning.
CALCULATOR_ANNOTATION = re.compile(r"<<[^>]*>>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=INSTRUCT_MODEL_ID)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-problems", type=int, default=400)
    parser.add_argument("--max-steps-per-problem", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"),
                        default="bfloat16")
    return parser.parse_args()


def gold_steps(answer: str) -> list[str]:
    """Split a gold solution into its reasoning steps, final answer last."""
    lines = [
        CALCULATOR_ANNOTATION.sub("", line).strip()
        for line in answer.splitlines() if line.strip()
    ]
    return [line for line in lines if line]


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

    # One request per (prefix, temperature). The gold step needs no generation.
    requests = []
    for problem in problems:
        steps = gold_steps(problem["answer"])[:args.max_steps_per_problem + 1]
        if len(steps) < 2:
            continue
        header = tokenizer.apply_chat_template(
            [{"role": "user",
              "content": f"{INSTRUCTION}\n\n{problem['question'].strip()}"}],
            tokenize=False, add_generation_prompt=True,
        )
        for index in range(len(steps) - 1):
            prefix = header + "\n".join(steps[:index]) + ("\n" if index else "")
            for temperature in SAMPLING_POLICY:
                requests.append({
                    "problem_id": problem["problem_id"],
                    "step_index": index,
                    "prefix": prefix,
                    "observed": steps[index],
                    "temperature": temperature,
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("w", encoding="utf-8") as out:
        for start in range(0, len(requests), args.batch_size):
            batch = requests[start:start + args.batch_size]
            encoded = tokenizer(
                [item["prefix"] for item in batch], return_tensors="pt",
                padding=True, add_special_tokens=False,
            ).to(args.device)
            temperature = batch[0]["temperature"]
            greedy = temperature == 0.0
            with torch.no_grad():
                generated = model.generate(
                    **encoded, do_sample=not greedy,
                    temperature=None if greedy else temperature,
                    top_p=None if greedy else args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completions = tokenizer.batch_decode(
                generated[:, encoded["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            for item, completion in zip(batch, completions):
                # Keep only the next step, so the candidate is comparable in
                # granularity to the observed one.
                alternative = completion.strip().splitlines()
                alternative = alternative[0].strip() if alternative else ""
                out.write(json.dumps({
                    "problem_id": item["problem_id"],
                    "step_index": item["step_index"],
                    "prefix": item["prefix"],
                    "observed": item["observed"],
                    "alternative": alternative,
                    "temperature": item["temperature"],
                    "is_observed": alternative == item["observed"],
                }) + "\n")
                written += 1
            if start % (args.batch_size * 20) == 0:
                print(json.dumps({"written": written, "of": len(requests)}),
                      flush=True)
    print(json.dumps({
        "status": "completed", "records": written,
        "problems": len(problems), "sampling_policy": list(SAMPLING_POLICY),
        "model_id": args.model_id, "model_revision": args.model_revision,
        "supervision": "observed_vs_sampled_continuation",
        "verifier_used": False,
        "candidate_privileged_outcomes": False,
    }), flush=True)


if __name__ == "__main__":
    main()
