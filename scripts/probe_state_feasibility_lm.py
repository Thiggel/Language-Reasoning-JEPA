"""State-readout probes on LM baselines, for comparison with the JEPA probes.

This is the LM counterpart of CONTROL 1 in ``scripts/probe_state_feasibility.py``
(2026-08-11 state-readout controls).  Same problems, same probe heads, same
label construction and the same shuffled-state control -- only the source of
the frozen state changes:

  * ``token_lm``    (``DecoderLM``): the causal token state at the last token
    before the next intent phrase, i.e. exactly the boundary at which the LM
    policy must choose its next action.  Obtained the same way as
    ``scripts/export_intent_representations.py``, so no future text is seen.
  * ``sentence_lm`` (``SentenceLM``): the context latent that must predict the
    next action chunk (``contexts()`` at ``target_mask``), the sentence-level
    analogue of the same causal boundary.

Both are sampled at the SAME causal boundary as the JEPA pooled state
``s_t`` (state before step ``t``), as required by the representation-analysis
contract in ``projects/intent_phrase/PAPER_EXPERIMENTS.md``.

Two candidate-action embeddings, run as separate variants:

  * ``u_lm_own``   -- the LM's OWN representation of the action phrase: for the
    token LM, the phrase is encoded on its own and the last-token state is
    taken; for the sentence LM, the phrase is encoded by ``chunk_encoder``
    (literally the vector the model uses for an intent chunk).  This is the
    honest "what the LM has" variant.
  * ``u_jepa_ref`` -- the frozen JEPA action encoder's 16-d ``u(c)``, used as a
    FIXED REFERENCE so that the state is the only thing that differs from the
    JEPA probe.  This is a diagnostic device, not something the LM possesses.

NOTE (evidence labelling): resolvedness and feasibility labels come from the
symbolic oracle, so every number here is CANDIDATE-PRIVILEGED diagnostic
evidence, not a planning result.

CONTROL 2 of the JEPA script (catalogue-softmax heads replicating the failed
contrastive action prior) has no LM analogue and is deliberately not run.

Usage:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/probe_state_feasibility_lm.py \
        --ckpt runs/lm_intent/best.pt --kind token_lm \
        --jepa-ckpt <path/best.pt> --n-train 2000 --n-val 800 --out out.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from textjepa.data.igsm.dataset import collate
from textjepa.data.igsm.render import catalogue_phrases
from textjepa.data.lm import (
    IntentSentencePolicyDataset,
    collate_intent_sentence_policy,
)
from textjepa.models.lm_baseline import DecoderLM
from textjepa.models.sent_lm import SentenceLM
from textjepa.utils.checkpoint import (
    build_dataset,
    build_vocab_for_config,
    load_run,
)

from probe_state_feasibility import (  # noqa: E402  (same directory)
    run_control1,
    var_depths,
)

torch.set_num_threads(min(16, torch.get_num_threads()))


def load_lm(ckpt_path: str, kind: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    vocab = build_vocab_for_config(cfg)
    cls = DecoderLM if kind == "token_lm" else SentenceLM
    model = cls(vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model)
    model.load_state_dict(ckpt["model"])
    return model.eval(), vocab, cfg


def _pad_token_chunks(seqs: list[list[int]], pad: int) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    return torch.tensor(
        [s + [pad] * (width - len(s)) for s in seqs], dtype=torch.long
    )


@torch.no_grad()
def token_lm_features(model: DecoderLM, items: list[dict], vocab) -> list[torch.Tensor]:
    """Causal token state at the last token before each intent phrase.

    Batched over problems; end padding cannot leak into earlier positions
    because ``DecoderLM.hidden`` masks pads under a causal mask.
    """
    streams, endpoints = [], []
    for item in items:
        stream = [t for sentence in item["prompt"] for t in sentence]
        ends = []
        for action, outcome in zip(item["actions"], item["steps"]):
            ends.append(len(stream) - 1)
            stream.extend(action)
            stream.extend(outcome)
        streams.append(stream)
        endpoints.append(ends)
    tokens = _pad_token_chunks(streams, vocab.pad_id)
    if tokens.shape[1] > model.pos.shape[1]:
        raise ValueError(
            f"trace of {tokens.shape[1]} tokens exceeds the LM context "
            f"({model.pos.shape[1]})"
        )
    hidden = model.hidden(tokens)
    return [hidden[b, torch.tensor(ends)] for b, ends in enumerate(endpoints)]


@torch.no_grad()
def token_lm_phrase_embeddings(
    model: DecoderLM, phrase_groups: list[list[str]], vocab
) -> list[torch.Tensor]:
    """Last-token causal state of each action phrase encoded on its own."""
    seqs = [vocab.encode(p) for group in phrase_groups for p in group]
    hidden = model.hidden(_pad_token_chunks(seqs, vocab.pad_id))
    last = torch.tensor([len(s) - 1 for s in seqs])
    flat = hidden[torch.arange(len(seqs)), last]
    return _ungroup(flat, phrase_groups)


@torch.no_grad()
def sentence_lm_features(
    model: SentenceLM, items: list[dict], vocab
) -> list[torch.Tensor]:
    """Context latent that must predict the next intent chunk."""
    wrapped = IntentSentencePolicyDataset(items)
    batch = collate_intent_sentence_policy(
        [wrapped[i] for i in range(len(items))], vocab.pad_id
    )
    contexts = model.contexts(batch)
    return [contexts[b][batch["target_mask"][b]] for b in range(len(items))]


@torch.no_grad()
def sentence_lm_phrase_embeddings(
    model: SentenceLM, phrase_groups: list[list[str]], vocab
) -> list[torch.Tensor]:
    seqs = [vocab.encode(p) for group in phrase_groups for p in group]
    tokens = _pad_token_chunks(seqs, vocab.pad_id)
    return _ungroup(model.encode_chunks(tokens.unsqueeze(0))[0], phrase_groups)


@torch.no_grad()
def jepa_u(jepa_model, phrase_groups: list[list[str]], vocab) -> list[torch.Tensor]:
    seqs = [vocab.encode(p) for group in phrase_groups for p in group]
    tokens = _pad_token_chunks(seqs, vocab.pad_id)
    return _ungroup(jepa_model.encode_actions(tokens.unsqueeze(0))[0], phrase_groups)


def _ungroup(flat: torch.Tensor, groups: list[list]) -> list[torch.Tensor]:
    out, offset = [], 0
    for group in groups:
        out.append(flat[offset:offset + len(group)])
        offset += len(group)
    return out


@torch.no_grad()
def extract(
    model, kind, dataset, vocab, n: int, jepa_model=None, batch_size: int = 32
) -> list[dict]:
    """Per-problem frozen LM features at the causal boundary + oracle structure."""
    feat = token_lm_features if kind == "token_lm" else sentence_lm_features
    phrase_emb = (
        token_lm_phrase_embeddings if kind == "token_lm"
        else sentence_lm_phrase_embeddings
    )
    records: list[dict] = []
    for start in range(0, n, batch_size):
        idxs = list(range(start, min(start + batch_size, n)))
        items = [dataset[i] for i in idxs]
        problems = [dataset.problem(i)[0] for i in idxs]
        groups = [catalogue_phrases(p) for p in problems]
        states = feat(model, items, vocab)
        u_lm = phrase_emb(model, groups, vocab)
        u_jepa = jepa_u(jepa_model, groups, vocab) if jepa_model else None
        for b, item in enumerate(items):
            trace = list(item["var_idx"])
            if len(states[b]) != len(trace):
                raise RuntimeError("state and step counts differ")
            record = dict(
                s=states[b].clone(),
                u_lm=u_lm[b].clone(),
                trace=trace,
                parents=[tuple(v.parents) for v in problems[b].vars],
                depth=var_depths(problems[b]),
                n_vars=len(problems[b].vars),
            )
            if u_jepa is not None:
                record["u_jepa"] = u_jepa[b].clone()
            records.append(record)
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", choices=("token_lm", "sentence_lm"), required=True)
    ap.add_argument(
        "--jepa-ckpt",
        help="JEPA checkpoint supplying the fixed-reference u(c) variant",
    )
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="probe_state_feasibility_lm.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, vocab, cfg = load_lm(args.ckpt, args.kind)
    jepa_model = None
    if args.jepa_ckpt:
        jepa_model, jepa_vocab, _ = load_run(args.jepa_ckpt, device="cpu")
        if len(jepa_vocab) != len(vocab):
            raise ValueError("JEPA and LM vocabularies differ; u(c) not comparable")
    train_ds = build_dataset(cfg, vocab, split="train", size=args.n_train)
    val_ds = build_dataset(cfg, vocab, split="val", size=args.n_val)

    t0 = time.time()
    print("extracting frozen features ...", flush=True)
    train_rec = extract(model, args.kind, train_ds, vocab, args.n_train, jepa_model)
    val_rec = extract(model, args.kind, val_ds, vocab, args.n_val, jepa_model)
    print(f"  done in {time.time() - t0:.0f}s", flush=True)

    results = {
        "ckpt": args.ckpt, "kind": args.kind, "jepa_ckpt": args.jepa_ckpt,
        "n_train": args.n_train, "n_val": args.n_val, "seed": args.seed,
        "d_state": int(train_rec[0]["s"].shape[1]),
        "d_action_lm": int(train_rec[0]["u_lm"].shape[1]),
        "evidence": "oracle-labelled, candidate-privileged diagnostic",
    }
    # variant A: the LM's own action-phrase representation (+ onehot control)
    specs = [
        ("u_lm_own", dict(u_key="u_lm", onehot=False)),
        ("onehot_var", dict(u_key="u_lm", onehot=True)),
    ]
    if jepa_model is not None:
        # variant B: frozen JEPA u(c) as a fixed reference embedding
        specs.append(("u_jepa_ref", dict(u_key="u_jepa", onehot=False)))
    print("CONTROL 1: read-out probes ...", flush=True)
    results["control1_probes"] = run_control1(
        train_rec, val_rec, args.seed, feature_specs=specs
    )
    for key, block in results["control1_probes"].items():
        print(f"  {key}: " + ", ".join(
            f"{k}={v['auc']:.3f}/{v['acc']:.3f}"
            for k, v in block.items() if isinstance(v, dict)
        ), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
