"""Closed-loop greedy token-prior control for the sentence hierarchy model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.semantic_lm import SemanticBoundaryLMDataset
from textjepa.models.sentence_hierarchy import SentenceHierarchyJEPA


@torch.no_grad()
def generate(model, prompt, max_tokens):
    generated = list(prompt)
    device = next(model.parameters()).device
    for _ in range(max_tokens):
        tokens = torch.tensor(generated, device=device).unsqueeze(0)
        state = model.encoder(tokens)[:, -1]
        logits = model.token_prior(state)
        logits[:, model.pad_id] = -torch.inf
        generated.append(int(logits.argmax(-1)))
    return generated[len(prompt):]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = SentenceHierarchyJEPA(
        len(vocab), vocab.pad_id, **cfg.model
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    dataset = SemanticBoundaryLMDataset(
        vocab, size=args.examples, seed=cfg.data.val_seed + 104729,
        boundary_mode="semantic", modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range), leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    exact = correct = total = sentences = 0
    for index in range(args.examples):
        item = dataset[index]
        prompt = item["tokens"][:item["prompt_len"]]
        reference = item["tokens"][item["prompt_len"]:]
        generated = generate(model, prompt, args.max_tokens)
        exact += int(generated == reference)
        overlap = min(len(reference), len(generated))
        correct += sum(generated[i] == reference[i] for i in range(overlap))
        total += max(len(reference), len(generated))
        sentences += sum(vocab.id_to_token[token] == "." for token in generated)
    result = {
        "exact_trace_success": exact / args.examples,
        "token_accuracy": correct / max(total, 1),
        "completed_sentence_count": sentences,
        "examples": args.examples, "max_tokens": args.max_tokens,
        "uses_symbolic_feasibility": False, "uses_auxiliary_lm": False,
    }
    destination = Path(args.ckpt).parent / "sentence_prior_closed_loop.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
