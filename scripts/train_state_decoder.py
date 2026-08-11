"""Train a state -> sentence read-out on a FROZEN JEPA backbone.

What this does, in plain terms: take a trained checkpoint, freeze it
completely, run it over iGSM problems to get its latent reasoning states, and
train a small 2-layer decoder to write out, in words, the reasoning sentence
that each state stands for ("so the number of shiny apples is 3 plus 4 = 7 ."),
plus a linear classifier on the final state for the numeric answer (a residue
class mod ``data.modulus``).

It is a *renderer*, not part of the JEPA objective: the backbone is in eval
mode with every parameter ``requires_grad=False`` and states are computed under
``torch.no_grad()``, so no gradient can shape the representation.  The main
trainer and objectives are untouched.

Usage::

    .venv/bin/python scripts/train_state_decoder.py \
        --ckpt runs/.../model/best.pt --out runs/.../decoder-s0-v1 \
        --epochs 8 --train-size 8000 --device cuda:1
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from textjepa.models.state_decoder import FrozenStateSentenceDecoder
from textjepa.probing.state_decoder_io import (
    load_frozen,
    make_loader,
    true_states,
)
from textjepa.training.trainer import to_device
from textjepa.utils import seed_everything

DEFAULT_CKPT = (
    "runs/autonomy/intent_phrase/2026-08-08-intent-stabilizer-sweep-v1/"
    "stab-ldad-ema-s0-v1/model/best.pt"
)


def build_decoder(cfg, vocab, max_len: int) -> FrozenStateSentenceDecoder:
    return FrozenStateSentenceDecoder(
        d_state=int(cfg.model.d_model),
        vocab_size=len(vocab),
        max_len=max_len,
        n_layers=2,
        n_heads=4,
        n_answers=int(cfg.data.modulus),
    )


def batch_loss(model, decoder, batch, pad_id: int):
    """Teacher-forced CE on step sentences + CE on the final-state answer."""
    states = true_states(model, batch).detach()
    tokens = batch["step_tokens"][..., : decoder.max_len]
    logits = decoder(states, tokens)
    valid = batch["step_mask"].unsqueeze(-1) & tokens.ne(pad_id)
    # PAD is the end marker: keep the first PAD of each real sentence as a
    # target so generation learns to stop.
    end = batch["step_mask"].unsqueeze(-1) & _first_pad(tokens, pad_id)
    keep = valid | end
    target = tokens.masked_fill(~keep, -100)
    sentence_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
        ignore_index=-100,
    )
    last = batch["step_mask"].sum(1).clamp(min=1) - 1
    final = states[torch.arange(states.shape[0], device=states.device), last]
    answer_loss = F.cross_entropy(
        decoder.answer_logits(final), batch["answer"]
    )
    with torch.no_grad():
        correct = (logits.argmax(-1) == tokens) & keep
        token_acc = correct.sum() / keep.sum().clamp(min=1)
        answer_acc = (
            decoder.answer_logits(final).argmax(-1) == batch["answer"]
        ).float().mean()
    return sentence_loss, answer_loss, float(token_acc), float(answer_acc)


def _first_pad(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Bool mask selecting only the first PAD position of each sentence."""
    is_pad = tokens.eq(pad_id)
    return is_pad & (is_pad.cumsum(-1) == 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--train-size", type=int, default=8000)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device

    model, vocab, cfg = load_frozen(args.ckpt, device)
    decoder = build_decoder(cfg, vocab, args.max_len).to(device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"decoder parameters: {n_params}")

    train = make_loader(
        cfg, vocab, "train", args.train_size, args.batch_size,
        shuffle=True, workers=args.workers,
    )
    val = make_loader(
        cfg, vocab, "val", args.val_size, args.batch_size,
        workers=args.workers,
    )
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.01)

    rows: list[dict] = []
    best = float("inf")
    for epoch in range(args.epochs):
        decoder.train()
        started = time.time()
        totals = [0.0, 0.0, 0.0, 0.0]
        n = 0
        for batch in train:
            batch = to_device(batch, device)
            s_loss, a_loss, t_acc, a_acc = batch_loss(
                model, decoder, batch, vocab.pad_id
            )
            loss = s_loss + a_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            assert not any(
                p.grad is not None for p in model.parameters()
            ), "gradient reached the frozen backbone"
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()
            totals = [
                totals[0] + float(s_loss), totals[1] + float(a_loss),
                totals[2] + t_acc, totals[3] + a_acc,
            ]
            n += 1
        decoder.eval()
        v_totals = [0.0, 0.0, 0.0, 0.0]
        v_n = 0
        with torch.no_grad():
            for batch in val:
                batch = to_device(batch, device)
                s_loss, a_loss, t_acc, a_acc = batch_loss(
                    model, decoder, batch, vocab.pad_id
                )
                v_totals = [
                    v_totals[0] + float(s_loss), v_totals[1] + float(a_loss),
                    v_totals[2] + t_acc, v_totals[3] + a_acc,
                ]
                v_n += 1
        row = {
            "epoch": epoch,
            "train_sentence_ce": totals[0] / max(n, 1),
            "train_answer_ce": totals[1] / max(n, 1),
            "train_token_acc": totals[2] / max(n, 1),
            "train_answer_acc": totals[3] / max(n, 1),
            "val_sentence_ce": v_totals[0] / max(v_n, 1),
            "val_answer_ce": v_totals[1] / max(v_n, 1),
            "val_token_acc": v_totals[2] / max(v_n, 1),
            "val_answer_acc": v_totals[3] / max(v_n, 1),
            "seconds": time.time() - started,
        }
        rows.append(row)
        print(json.dumps(row))
        with (out / "metrics.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        score = row["val_sentence_ce"] + row["val_answer_ce"]
        if score < best:
            best = score
            torch.save(
                {
                    "decoder": decoder.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "max_len": args.max_len,
                    "d_state": int(cfg.model.d_model),
                    "n_answers": int(cfg.data.modulus),
                },
                out / "decoder.pt",
            )
    print(f"best val score: {best:.4f}; wrote {out}")


if __name__ == "__main__":
    main()
