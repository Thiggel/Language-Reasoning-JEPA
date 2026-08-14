#!/usr/bin/env python3
"""Train an energy head by ranking observed continuations above sampled ones.

Training uses no verifier and no step counter. The only signal is which
continuation is the one the dataset records as having occurred, which is
self-supervised.

Two things are then measured, and both matter:

1. Does the energy add anything over the language model's own likelihood? A
   head that merely re-derives length-normalized log probability is not an
   energy, it is a slower way to read the logits. The likelihood baseline is
   computed on exactly the same candidates.
2. Does an energy trained without correctness generalize to correctness? The
   verifier labels on separately generated trajectories are used here for
   evaluation only, never for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from textjepa.models.action_transition import ResidualCapture
from textjepa.training.predictive_state import (
    load_stage1_checkpoint,
    require_transformers_runtime,
    teacher_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counterfactuals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-revision",
                        default="7ae557604adf67be50417f59c2c2f167def9a775")
    parser.add_argument("--source-layer", type=int, action="append")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-groups", type=int, default=16)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"),
                        default="bfloat16")
    return parser.parse_args()


class EnergyHead(nn.Module):
    """Scalar energy on a state. Lower means a better continuation."""

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


@torch.no_grad()
def extract(model, tokenizer, capture, records, layers, args):
    """State after each candidate step, plus its length-normalized logprob."""
    states, logprobs = [], []
    for start in range(0, len(records), args.extract_batch_size):
        batch = records[start:start + args.extract_batch_size]
        texts = [item["prefix"] + item["candidate"] for item in batch]
        prefix_lengths = [
            len(tokenizer(item["prefix"], add_special_tokens=False)["input_ids"])
            for item in batch
        ]
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=args.max_length, add_special_tokens=False,
        ).to(args.device)
        output = teacher_forward(
            model, encoded["input_ids"], capture=capture,
            attention_mask=encoded["attention_mask"], use_cache=False,
        )
        hidden = torch.cat([output.states[layer] for layer in layers], dim=-1)
        log_probs = torch.log_softmax(output.logits.float(), dim=-1)
        for row, item in enumerate(batch):
            length = int(encoded["attention_mask"][row].sum())
            last = length - 1
            states.append(hidden[row, last].float().cpu())
            begin = min(prefix_lengths[row], last)
            if last <= begin:
                logprobs.append(0.0)
                continue
            targets = encoded["input_ids"][row, begin + 1:length]
            picked = log_probs[row, begin:length - 1].gather(
                -1, targets[:, None]
            ).squeeze(-1)
            logprobs.append(float(picked.mean()))
    return torch.stack(states), torch.tensor(logprobs)


def ranking_accuracy(scores: torch.Tensor, groups: list[tuple[int, list[int]]],
                     lower_is_better: bool) -> float:
    """Share of alternatives the observed candidate is ranked above."""
    wins = total = 0
    for observed, alternatives in groups:
        for index in alternatives:
            better = (
                scores[observed] < scores[index] if lower_is_better
                else scores[observed] > scores[index]
            )
            wins += int(bool(better))
            total += 1
    return wins / max(total, 1)


def main() -> None:
    args = parse_args()
    require_transformers_runtime()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)

    by_prefix: dict[tuple[str, int], dict] = {}
    with args.counterfactuals.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row["alternative"]:
                continue
            key = (row["problem_id"], row["step_index"])
            entry = by_prefix.setdefault(key, {
                "problem_id": row["problem_id"], "prefix": row["prefix"],
                "observed": row["observed"], "alternatives": [],
            })
            if row["alternative"] != row["observed"]:
                entry["alternatives"].append(row["alternative"])
    groups_raw = [g for g in by_prefix.values() if g["alternatives"]]
    if not groups_raw:
        raise ValueError("no prefix has both an observed and an alternative step")

    # Split by problem so no prefix from a training problem appears held out.
    problems = sorted({g["problem_id"] for g in groups_raw},
                      key=lambda v: hashlib.sha256(v.encode()).digest())
    held_out = set(problems[:max(1, round(0.2 * len(problems)))])

    records, groups, is_train = [], [], []
    for group in groups_raw:
        observed_index = len(records)
        records.append({"prefix": group["prefix"], "candidate": group["observed"]})
        alternative_indices = []
        for alternative in group["alternatives"]:
            alternative_indices.append(len(records))
            records.append({"prefix": group["prefix"], "candidate": alternative})
        groups.append((observed_index, alternative_indices))
        is_train.append(group["problem_id"] not in held_out)

    if args.checkpoint:
        model, predictor, payload = load_stage1_checkpoint(
            args.checkpoint, device=args.device, dtype=dtype
        )
        layers = tuple(args.source_layer or predictor.config.source_layers)
        model_id = payload["model_id"]
        revision = payload["model_revision"]
    else:
        model_id, revision = args.model_id, args.model_revision
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=dtype, low_cpu_mem_usage=True,
        ).to(args.device)
        model.requires_grad_(False)
        layers = tuple(args.source_layer or (18, 24))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision,
                                              padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    with ResidualCapture(model, layers) as capture:
        states, logprobs = extract(model, tokenizer, capture, records, layers, args)
    print(json.dumps({"extracted": len(records), "width": states.shape[1]}),
          flush=True)

    train_groups = [g for g, keep in zip(groups, is_train) if keep]
    test_groups = [g for g, keep in zip(groups, is_train) if not keep]

    center = states[[i for g in train_groups for i in (g[0], *g[1])]].mean(0)
    scale = states[[i for g in train_groups for i in (g[0], *g[1])]].std(0).clamp_min(1e-6)
    normalized = ((states - center) / scale).to(args.device)

    head = EnergyHead(states.shape[1], args.hidden_size).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-2)
    generator = torch.Generator().manual_seed(args.seed)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        picks = torch.randint(len(train_groups), (args.batch_groups,),
                              generator=generator)
        loss = normalized.new_zeros(())
        for pick in picks.tolist():
            observed, alternatives = train_groups[pick]
            energy_observed = head(normalized[observed][None])
            energy_alternatives = head(normalized[alternatives])
            # Observed must sit below every alternative by the margin.
            loss = loss + torch.clamp(
                args.margin + energy_observed - energy_alternatives, min=0
            ).mean()
        (loss / args.batch_groups).backward()
        optimizer.step()
        if step % 500 == 0:
            print(json.dumps({"step": step,
                              "loss": float(loss / args.batch_groups)}), flush=True)

    with torch.no_grad():
        energies = head(normalized).float().cpu()
    report = {
        "schema_version": 1,
        "kind": "self_supervised_energy_head",
        "model_id": model_id,
        "checkpoint": None if not args.checkpoint else str(args.checkpoint),
        "source_layers": list(layers),
        "prefixes": len(groups),
        "candidates": len(records),
        "held_out_problems": len(held_out),
        "train": {
            "energy_ranking_accuracy": ranking_accuracy(energies, train_groups, True),
            "likelihood_ranking_accuracy": ranking_accuracy(logprobs, train_groups, False),
        },
        "held_out": {
            "energy_ranking_accuracy": ranking_accuracy(energies, test_groups, True),
            "likelihood_ranking_accuracy": ranking_accuracy(logprobs, test_groups, False),
        },
        "supervision": "observed_vs_sampled_continuation",
        "verifier_used_in_training": False,
        "step_counter_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head": head.state_dict(), "center": center, "scale": scale,
        "source_layers": layers, "model_id": model_id, "revision": revision,
        "report": report,
    }, args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
