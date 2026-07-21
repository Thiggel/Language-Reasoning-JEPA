"""Unconstrained generation for token- and sentence-LM iGSM baselines.

No feasible-action list or environment feedback is exposed during generation.
The symbolic environment is used only after generation to score validity and
problem completion, matching the pooled-sentence JEPA generation diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

try:
    from eval_pooled_sentence_planning import (
        summarize_examples,
        summarize_generation_validity,
        validate_generated_trace,
    )
except ModuleNotFoundError:
    from scripts.eval_pooled_sentence_planning import (
        summarize_examples,
        summarize_generation_validity,
        validate_generated_trace,
    )
from textjepa.data.igsm.dataset import build_vocab
from textjepa.models.lm_baseline import DecoderLM
from textjepa.models.sent_lm import SentenceLM
from textjepa.utils.checkpoint import build_dataset


@torch.no_grad()
def token_beam_generate(model, prompt, length, width):
    """Ordinary left-to-right beam search over vocabulary tokens."""
    device = next(model.parameters()).device
    beams = [(list(prompt), 0.0)]
    for _ in range(int(length)):
        lengths = torch.tensor([len(tokens) for tokens, _ in beams], device=device)
        padded = torch.full(
            (len(beams), int(lengths.max())), model.pad_id,
            dtype=torch.long, device=device,
        )
        for row, (tokens, _) in enumerate(beams):
            padded[row, :len(tokens)] = torch.tensor(tokens, device=device)
        logits = model(padded)[torch.arange(len(beams), device=device), lengths - 1]
        logits[:, model.pad_id] = -torch.inf
        logp = logits.log_softmax(-1)
        branch = min(int(width), logp.shape[-1] - 1)
        values, ids = logp.topk(branch, -1)
        candidates = []
        for row, (tokens, score) in enumerate(beams):
            for column in range(branch):
                candidates.append((
                    tokens + [int(ids[row, column])],
                    score + float(values[row, column]),
                ))
        beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:width]
    return beams[0][0][len(prompt):]


def _chunk_tensor(chunks, pad_id, device):
    width = max(map(len, chunks))
    out = torch.full((1, len(chunks), width), pad_id, dtype=torch.long, device=device)
    for index, chunk in enumerate(chunks):
        out[0, index, :len(chunk)] = torch.tensor(chunk, device=device)
    return out


@torch.no_grad()
def sentence_context(model, prompt_chunks, generated_sentences):
    """Context used to decode the next sentence, without a latent target."""
    device = next(model.parameters()).device
    prompt = _chunk_tensor(prompt_chunks, model.pad_id, device)
    if generated_sentences:
        steps = _chunk_tensor(generated_sentences, model.pad_id, device)
        step_mask = torch.ones(1, len(generated_sentences), dtype=torch.bool, device=device)
    else:
        steps = torch.full((1, 1, 1), model.pad_id, dtype=torch.long, device=device)
        step_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
    prompt_emb = model.encode_chunks(prompt)
    step_emb = model.encode_chunks(steps)
    initial, states = model.state_model(
        prompt_emb,
        torch.ones(1, len(prompt_chunks), dtype=torch.bool, device=device),
        step_emb,
        step_mask,
    )
    return states[:, len(generated_sentences) - 1] if generated_sentences else initial


@torch.no_grad()
def sentence_next_logits(model, context, prefixes):
    """Next-token logits from the sentence decoder for variable prefixes."""
    device = context.device
    lengths = torch.tensor([len(prefix) + 1 for prefix in prefixes], device=device)
    width = int(lengths.max())
    inputs = torch.full(
        (len(prefixes), width), model.pad_id, dtype=torch.long, device=device
    )
    for row, prefix in enumerate(prefixes):
        if prefix:
            inputs[row, 1:len(prefix) + 1] = torch.tensor(prefix, device=device)
    hidden = model.dec_tok(inputs) + model.dec_pos[:, :width]
    causal = torch.nn.Transformer.generate_square_subsequent_mask(width, device=device)
    decoded = model.decoder(hidden, context.expand(len(prefixes), -1).unsqueeze(1), tgt_mask=causal)
    return model.dec_head(decoded)[torch.arange(len(prefixes), device=device), lengths - 1]


@torch.no_grad()
def decode_sentence(model, context, period_id, width, max_tokens):
    """Beam-decode one sentence from a single discourse context."""
    beams = [([], 0.0, False)]
    for _ in range(int(max_tokens)):
        active = [item for item in beams if not item[2]]
        completed = [item for item in beams if item[2]]
        if not active:
            break
        logits = sentence_next_logits(model, context, [item[0] for item in active])
        logits[:, model.pad_id] = -torch.inf
        logp = logits.log_softmax(-1)
        branch = min(int(width), logp.shape[-1] - 1)
        values, ids = logp.topk(branch, -1)
        candidates = list(completed)
        for row, (prefix, score, _) in enumerate(active):
            for column in range(branch):
                token = int(ids[row, column])
                candidates.append((
                    prefix + [token], score + float(values[row, column]),
                    token == period_id,
                ))
        beams = sorted(candidates, key=lambda item: item[1], reverse=True)[:width]
    return beams[0][0]


@torch.no_grad()
def sentence_generate(model, prompt_chunks, length, width, period_id):
    generated, sentences = [], []
    while len(generated) < int(length):
        context = sentence_context(model, prompt_chunks, sentences)
        sentence = decode_sentence(
            model, context, period_id, width,
            min(model.dec_pos.shape[1], int(length) - len(generated)),
        )
        if not sentence:
            break
        generated.extend(sentence)
        sentences.append(sentence)
    return generated[:length]


def load_model(kind, checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    if kind == "token":
        model = DecoderLM(
            len(vocab), vocab.pad_id, cfg.model.d_model, cfg.model.n_layers,
            cfg.model.n_heads, cfg.model.ff_mult, cfg.model.max_len,
        )
    else:
        model = SentenceLM(len(vocab), vocab.pad_id, **cfg.model)
        if model.latent_target:
            raise ValueError("the CE-only sentence baseline requires latent_target=false")
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), vocab, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("token", "sentence"), required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--width", type=int, choices=(1, 8), default=1)
    parser.add_argument("--eval-seed", type=int, default=200003)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    model, vocab, cfg = load_model(args.kind, args.ckpt, args.device)
    eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    eval_cfg.data.test_seed = args.eval_seed
    dataset = build_dataset(eval_cfg, vocab, split="test", size=args.examples)
    fixed, matched_prefixes, references, validity = [], [], [], []
    for index in range(args.examples):
        item = dataset[index]
        prompt_chunks = item["prompt"]
        prompt = [token for chunk in prompt_chunks for token in chunk]
        reference = [token for chunk in item["steps"] for token in chunk][:args.max_tokens]
        if args.kind == "token":
            generate = lambda count: token_beam_generate(model, prompt, count, args.width)
        else:
            generate = lambda count: sentence_generate(
                model, prompt_chunks, count, args.width, vocab.token_to_id["."]
            )
        fixed_trace = generate(args.max_tokens)
        problem, _ = dataset.problem(index)
        fixed.append(fixed_trace)
        matched_prefixes.append(fixed_trace[:len(reference)])
        references.append(reference)
        validity.append(validate_generated_trace(fixed_trace, problem, vocab))
    result = {
        "model_kind": args.kind,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "objective": "next_token_ce" if args.kind == "token" else "next_sentence_decoder_ce_only",
        "beam_width": args.width,
        "examples": args.examples,
        "eval_seed": args.eval_seed,
        "max_tokens": args.max_tokens,
        "reference_length_prefix_metrics": summarize_examples(
            matched_prefixes, references,
            {vocab.token_to_id["."], vocab.token_to_id["?"]}, args.eval_seed,
        ),
        "fixed_budget_generation": summarize_generation_validity(validity),
        "uses_oracle_length": False,
        "uses_symbolic_feedback": False,
        "uses_feasible_action_candidates": False,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
