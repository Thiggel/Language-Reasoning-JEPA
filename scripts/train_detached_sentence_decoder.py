"""Train a post-hoc decoder without allowing gradients into sentence states."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.semantic_lm import (
    SemanticBoundaryLMDataset, collate_semantic_lm,
)
from textjepa.models.pooled_sentence_jepa import (
    PooledSentenceJEPA, PrefixAutoregressiveDecoder,
)


@torch.no_grad()
def extract(model, vocab, cfg, seed, examples, batch_size, device):
    dataset = SemanticBoundaryLMDataset(
        vocab, size=examples, seed=seed, boundary_mode="semantic",
        modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range),
        leaf_prob=cfg.data.leaf_prob, steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=partial(collate_semantic_lm, pad_id=vocab.pad_id),
    )
    states, sentences = [], []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        encoded = model.state_encoder(tokens)
        for row in range(len(tokens)):
            prompt = int(batch["prompt_len"][row])
            previous_end = 0
            ends = batch["sentence_ends"][row]
            ends = ends[ends.ge(0)].tolist()
            for end in ends:
                position = prompt + int(end) - 1
                states.append(encoded[row, position].cpu())
                sentences.append(tokens[
                    row, prompt + previous_end:prompt + int(end)
                ].cpu())
                previous_end = int(end)
    width = max(map(len, sentences))
    ids = torch.full((len(sentences), width), vocab.pad_id, dtype=torch.long)
    valid = torch.zeros_like(ids, dtype=torch.bool)
    for index, sentence in enumerate(sentences):
        ids[index, :len(sentence)] = sentence
        valid[index, :len(sentence)] = True
    return torch.stack(states), ids, valid


def sequence_loss(decoder, state, ids, valid):
    logits = decoder(state, ids, valid)
    loss = F.cross_entropy(logits.transpose(1, 2), ids, reduction="none")
    return (loss * valid).sum() / valid.sum().clamp_min(1)


@torch.no_grad()
def evaluate(decoder, loader, pad_id, device):
    totals = {
        "tokens": 0, "teacher_correct": 0, "free_correct": 0,
        "sentences": 0, "free_exact": 0, "teacher_loss": 0.0,
        "shuffled_loss": 0.0, "zero_loss": 0.0,
    }
    for state, ids, valid in loader:
        state, ids, valid = state.to(device), ids.to(device), valid.to(device)
        logits = decoder(state, ids, valid)
        permutation = torch.roll(torch.arange(len(state), device=device), 1)
        shuffled = decoder(state[permutation], ids, valid)
        zero = decoder(torch.zeros_like(state), ids, valid)
        count = int(valid.sum())
        totals["teacher_loss"] += float(F.cross_entropy(
            logits[valid], ids[valid], reduction="sum"
        ))
        totals["shuffled_loss"] += float(F.cross_entropy(
            shuffled[valid], ids[valid], reduction="sum"
        ))
        totals["zero_loss"] += float(F.cross_entropy(
            zero[valid], ids[valid], reduction="sum"
        ))
        totals["teacher_correct"] += int(logits.argmax(-1)[valid].eq(ids[valid]).sum())
        generated = torch.full_like(ids, pad_id)
        generation_valid = torch.ones_like(valid)
        for step in range(ids.shape[1]):
            step_logits = decoder(state, generated, generation_valid)
            generated[:, step] = step_logits[:, step].argmax(-1)
        totals["free_correct"] += int(generated[valid].eq(ids[valid]).sum())
        totals["free_exact"] += int(((generated == ids) | ~valid).all(1).sum())
        totals["tokens"] += count
        totals["sentences"] += len(ids)
    return {
        "teacher_forced_ce": totals["teacher_loss"] / totals["tokens"],
        "shuffled_state_ce": totals["shuffled_loss"] / totals["tokens"],
        "zero_state_ce": totals["zero_loss"] / totals["tokens"],
        "teacher_forced_token_accuracy": totals["teacher_correct"] / totals["tokens"],
        "free_running_token_accuracy": totals["free_correct"] / totals["tokens"],
        "free_running_exact_sentence": totals["free_exact"] / totals["sentences"],
        "sentences": totals["sentences"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-examples", type=int, default=1024)
    parser.add_argument("--test-examples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = PooledSentenceJEPA(
        len(vocab), vocab.pad_id, period_id=vocab.token_to_id["."],
        question_id=vocab.token_to_id["?"], **cfg.model,
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval().requires_grad_(False)
    train = extract(
        model, vocab, cfg, 510007, args.train_examples, 16, args.device
    )
    test = extract(
        model, vocab, cfg, 610031, args.test_examples, 16, args.device
    )
    train_loader = DataLoader(
        TensorDataset(*train), batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(*test), batch_size=args.batch_size, shuffle=False
    )
    decoder = PrefixAutoregressiveDecoder(
        len(vocab), vocab.pad_id, model.d_state, d_model=256, n_layers=2,
        n_heads=8, ff_mult=4, max_len=train[1].shape[1],
    ).to(args.device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.01)
    for _ in range(args.epochs):
        decoder.train()
        for state, ids, valid in train_loader:
            state, ids, valid = state.to(args.device), ids.to(args.device), valid.to(args.device)
            loss = sequence_loss(decoder, state, ids, valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
    decoder.eval()
    result = evaluate(decoder, test_loader, vocab.pad_id, args.device)
    result.update({
        "encoder_frozen": True, "problem_disjoint_train_test": True,
        "decoder_layers": 2, "decoder_width": 256,
        "decoder_memory_tokens": 1,
    })
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
