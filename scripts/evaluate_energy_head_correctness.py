#!/usr/bin/env python3
"""Does an energy trained without correctness transfer to correctness?

The head was trained only to rank the continuation that occurred above sampled
alternatives. Here it scores separately generated trajectories whose
correctness an external verifier already established. Those labels are used for
evaluation only; nothing here is trained.

The comparison that matters is against the language model's own likelihood on
the same trajectories, because an energy that only reproduces likelihood adds
nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from textjepa.models.action_transition import ResidualCapture
from textjepa.training.predictive_state import (
    require_transformers_runtime,
    teacher_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-problems", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"),
                        default="bfloat16")
    return parser.parse_args()


class EnergyHead(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty(len(scores), dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    total = ranks[labels == 1].sum()
    return float((total - positive * (positive + 1) / 2) / (positive * negative))


def main() -> None:
    args = parse_args()
    require_transformers_runtime()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    payload = torch.load(args.head, map_location="cpu", weights_only=False)
    layers = tuple(payload["source_layers"])
    center, scale = payload["center"], payload["scale"]
    dtype = getattr(torch, args.dtype)

    model = AutoModelForCausalLM.from_pretrained(
        payload["model_id"], revision=payload["revision"], dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.requires_grad_(False)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        payload["model_id"], revision=payload["revision"], padding_side="right"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    head = EnergyHead(center.numel(), payload["head"]["net.3.weight"].shape[0])
    head.load_state_dict(payload["head"])
    head = head.to(args.device).eval()

    seen, records = set(), []
    with args.trajectories.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row["trajectory"].strip():
                continue
            seen.add(row["problem_id"])
            if len(seen) > args.max_problems:
                break
            records.append(row)

    energies, likelihoods, labels, problems = [], [], [], []
    with ResidualCapture(model, layers) as capture:
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            texts = [r["prompt"] + r["trajectory"] for r in batch]
            prefix_lengths = [
                len(tokenizer(r["prompt"], add_special_tokens=False)["input_ids"])
                for r in batch
            ]
            encoded = tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_length, add_special_tokens=False,
            ).to(args.device)
            with torch.no_grad():
                output = teacher_forward(
                    model, encoded["input_ids"], capture=capture,
                    attention_mask=encoded["attention_mask"], use_cache=False,
                )
                hidden = torch.cat(
                    [output.states[layer] for layer in layers], dim=-1
                ).float()
                log_probs = torch.log_softmax(output.logits.float(), dim=-1)
            for row, item in enumerate(batch):
                length = int(encoded["attention_mask"][row].sum())
                last = length - 1
                begin = min(prefix_lengths[row], last)
                if last <= begin:
                    continue
                state = ((hidden[row, last].cpu() - center) / scale)
                with torch.no_grad():
                    energies.append(float(head(state[None].to(args.device))))
                targets = encoded["input_ids"][row, begin + 1:length]
                picked = log_probs[row, begin:length - 1].gather(
                    -1, targets[:, None]
                ).squeeze(-1)
                likelihoods.append(float(picked.mean()))
                labels.append(int(bool(item["correct"])))
                problems.append(item["problem_id"])

    energy = torch.tensor(energies)
    likelihood = torch.tensor(likelihoods)
    label = torch.tensor(labels)

    # Lower energy should mean better, so negate for an AUC on "is correct".
    report = {
        "schema_version": 1,
        "kind": "energy_head_correctness_transfer",
        "head": str(args.head),
        "trajectories": len(labels),
        "problems": len(set(problems)),
        "correct_fraction": float(label.float().mean()),
        "energy_auc_for_correct": roc_auc(-energy, label),
        "likelihood_auc_for_correct": roc_auc(likelihood, label),
        "verifier_used_in_training": False,
        "verifier_used_for_evaluation_only": True,
    }

    # Matched-problem comparison: within a problem, does the energy prefer a
    # correct trajectory over an incorrect one? This removes any between-problem
    # difficulty effect.
    wins = ties = total = 0
    likelihood_wins = 0
    by_problem: dict[str, list[int]] = {}
    for index, name in enumerate(problems):
        by_problem.setdefault(name, []).append(index)
    for indices in by_problem.values():
        positives = [i for i in indices if label[i] == 1]
        negatives = [i for i in indices if label[i] == 0]
        for positive in positives:
            for negative in negatives:
                total += 1
                if float(energy[positive]) < float(energy[negative]):
                    wins += 1
                elif float(energy[positive]) == float(energy[negative]):
                    ties += 1
                if float(likelihood[positive]) > float(likelihood[negative]):
                    likelihood_wins += 1
    report["matched_problem_pairs"] = total
    report["energy_matched_problem_accuracy"] = wins / max(total, 1)
    report["likelihood_matched_problem_accuracy"] = likelihood_wins / max(total, 1)
    report["ties"] = ties

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
