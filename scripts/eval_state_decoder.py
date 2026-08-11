"""Evaluate the frozen-state sentence read-out.

Two questions, both on held-out problems:

(a) TRUE states — feed the decoder the states the encoder actually computes
    after reading each real reasoning sentence, and ask it to write that
    sentence back out.  This measures how much of the sentence (structure and
    numbers) survives in the latent state at all.

(b) IMAGINED states — start from ``s_0`` (prompt only, no reasoning read yet)
    and roll the predictor forward along the ground-truth action sequence,
    decoding each imagined state.  This measures whether the model's *own*
    latent simulation of the reasoning still renders as correct sentences, as
    a function of rollout depth 1..8.

Reported metrics per condition: sentence exact-match, token accuracy, and
value correctness (the "= v ." result number parsed out of the decoded vs the
reference sentence).  Answer-head accuracy is reported for the final state.

Usage::

    .venv/bin/python scripts/eval_state_decoder.py \
        --ckpt .../best.pt --decoder .../decoder.pt --out .../eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from textjepa.models.state_decoder import FrozenStateSentenceDecoder
from textjepa.probing.state_decoder_io import (
    imagined_states,
    load_frozen,
    make_loader,
    true_states,
)
from textjepa.training.trainer import to_device
from textjepa.utils import seed_everything


def parse_value(text: str) -> str | None:
    """Result number of a rendered step sentence, or ``None``.

    Step sentences are rendered by ``textjepa.data.igsm.render.step_sentence``
    as either ``so the number of X is <c> .`` (leaf) or
    ``so the number of X is <a> <op> <b> = <v> .``  The result is therefore the
    token after ``=`` when present, else the token before the final ``.``.
    """
    words = text.split()
    while words and words[-1] == ".":
        words = words[:-1]
    if not words:
        return None
    if "=" in words:
        at = words.index("=")
        return words[at + 1] if at + 1 < len(words) else None
    return words[-1]


class Accumulator:
    def __init__(self) -> None:
        self.n = 0
        self.exact = 0
        self.tokens = 0
        self.token_hits = 0
        self.value_n = 0
        self.value_hits = 0

    def add(self, hypothesis: list[int], reference: list[int], vocab) -> None:
        self.n += 1
        self.exact += int(hypothesis == reference)
        for i, ref in enumerate(reference):
            self.tokens += 1
            self.token_hits += int(i < len(hypothesis) and hypothesis[i] == ref)
        want = parse_value(vocab.decode(reference))
        got = parse_value(vocab.decode(hypothesis))
        if want is not None:
            self.value_n += 1
            self.value_hits += int(got == want)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "exact_match": self.exact / max(self.n, 1),
            "token_acc": self.token_hits / max(self.tokens, 1),
            "value_acc": self.value_hits / max(self.value_n, 1),
        }


def load_decoder(path: str, vocab, device: str):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    decoder = FrozenStateSentenceDecoder(
        d_state=blob["d_state"],
        vocab_size=len(vocab),
        max_len=blob["max_len"],
        n_layers=2,
        n_heads=4,
        n_answers=blob["n_answers"],
    )
    decoder.load_state_dict(blob["decoder"])
    return decoder.to(device).eval()


def strip(tokens: torch.Tensor, pad_id: int) -> list[int]:
    ids: list[int] = []
    for token in tokens.tolist():
        if token == pad_id:
            break
        ids.append(int(token))
    return ids


def markdown(title: str, rows: list[tuple[str, dict]]) -> str:
    lines = [f"### {title}", "", "| slice | n | exact | token acc | value acc |",
             "| --- | --- | --- | --- | --- |"]
    for name, m in rows:
        lines.append(
            f"| {name} | {m['n']} | {m['exact_match']:.3f} | "
            f"{m['token_acc']:.3f} | {m['value_acc']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--size", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, vocab, cfg = load_frozen(args.ckpt, args.device)
    decoder = load_decoder(args.decoder, vocab, args.device)
    loader = make_loader(
        cfg, vocab, args.split, args.size, args.batch_size, workers=2
    )

    true_all = Accumulator()
    true_by_depth = {d: Accumulator() for d in range(1, args.max_depth + 1)}
    imagined_by_depth = {d: Accumulator() for d in range(1, args.max_depth + 1)}
    answer_hits = answer_n = 0
    samples: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, args.device)
            mask = batch["step_mask"]
            B, T = mask.shape
            states = true_states(model, batch)
            imagined = imagined_states(model, batch, args.max_depth)

            flat = decoder.generate(states.reshape(B * T, -1))
            imag_flat = decoder.generate(
                imagined.reshape(B * imagined.shape[1], -1)
            )
            for b in range(B):
                for t in range(T):
                    if not bool(mask[b, t]):
                        continue
                    reference = strip(batch["step_tokens"][b, t], vocab.pad_id)
                    hypothesis = flat[b * T + t]
                    true_all.add(hypothesis, reference, vocab)
                    if t + 1 <= args.max_depth:
                        true_by_depth[t + 1].add(hypothesis, reference, vocab)
                    if len(samples) < 20 and b == 0:
                        samples.append({
                            "step": t,
                            "reference": vocab.decode(reference),
                            "true_state": vocab.decode(hypothesis),
                        })
                for d in range(1, imagined.shape[1] + 1):
                    t = d - 1
                    if not bool(mask[b, t]):
                        continue
                    reference = strip(batch["step_tokens"][b, t], vocab.pad_id)
                    imagined_by_depth[d].add(
                        imag_flat[b * imagined.shape[1] + t], reference, vocab
                    )

            last = mask.sum(1).clamp(min=1) - 1
            final = states[torch.arange(B, device=states.device), last]
            answer_hits += int(
                (decoder.answer_logits(final).argmax(-1) == batch["answer"])
                .sum()
            )
            answer_n += B

    result = {
        "ckpt": args.ckpt,
        "decoder": args.decoder,
        "split": args.split,
        "size": args.size,
        "true_states": true_all.as_dict(),
        "true_states_by_step": {
            str(d): a.as_dict() for d, a in true_by_depth.items() if a.n
        },
        "imagined_by_depth": {
            str(d): a.as_dict() for d, a in imagined_by_depth.items() if a.n
        },
        "answer_head_acc": answer_hits / max(answer_n, 1),
        "answer_head_n": answer_n,
        "samples": samples,
    }
    (out / "state_decoder_eval.json").write_text(json.dumps(result, indent=2))
    text = (
        markdown("True encoded states", [("all", result["true_states"])]
                 + [(f"step {d}", m)
                    for d, m in result["true_states_by_step"].items()])
        + "\n"
        + markdown("Imagined states (predictor rollout from s0)",
                   [(f"depth {d}", m)
                    for d, m in result["imagined_by_depth"].items()])
        + f"\nAnswer-head accuracy (final true state): "
          f"{result['answer_head_acc']:.3f} over {answer_n} problems\n"
    )
    (out / "state_decoder_eval.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
