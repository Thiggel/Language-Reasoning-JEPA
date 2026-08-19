"""Uniform frozen-representation interface over the four model families.

Every adapter exposes the SAME four operations, so a geometry metric can be
computed on a DiscourseJEPA, a FlatIntentJEPA, a token LM and a sentence LM
without any per-family branching in the analysis code:

``encode_states(prompt, steps)``
    causal state BEFORE each step, ``[T + 1, D]`` (row ``t`` is the state
    after ``steps[:t]``).
``encode_actions_ctx(prompt, steps, phrases)``
    the model's representation of each candidate intent phrase *in the
    context* of that history, ``[n, D]``.  This is the position-matched
    comparison: for the flat JEPA and the token LM it is the hidden state at
    the phrase's last token; for the chunked families it is the chunk vector.
``encode_actions_free(phrases)``
    the context-free action code (``d_action`` for DiscourseJEPA, ``d_model``
    for the sentence LM), or ``None`` where the family has none.
``predict(states, actions)``
    the JEPA predictor's imagined next state, or ``None`` for LM baselines
    (they have no action-conditioned latent dynamics -- that absence is a
    finding, not an omission).

Nothing here trains or mutates a checkpoint; every call is ``torch.no_grad``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from pathlib import Path

from omegaconf import OmegaConf


# --------------------------------------------------------------------------- #
def pad_stack(seqs: list[list[int]], pad: int) -> torch.Tensor:
    width = max((len(s) for s in seqs), default=1)
    return torch.tensor(
        [s + [pad] * (width - len(s)) for s in seqs], dtype=torch.long
    )


class Adapter:
    """Interface; see the module docstring."""

    name: str = "adapter"
    family: str = "?"
    d_state: int = 0
    d_action: int = 0
    trained: bool = True

    def encode_states(self, prompt, steps):
        raise NotImplementedError

    def encode_actions_ctx(self, prompt, steps, phrases):
        raise NotImplementedError

    def encode_actions_free(self, phrases):
        return None

    def predict(self, states, actions):
        return None

    def info(self) -> dict:
        return {
            "name": self.name, "family": self.family,
            "d_state": self.d_state, "d_action": self.d_action,
            "trained": self.trained,
            "has_predictor": self.predict(
                torch.zeros(1, self.d_state), torch.zeros(1, self.d_action)
            ) is not None if self.d_state else False,
        }


# --------------------------------------------------------------------------- #
# chunked families (sentence-per-chunk): DiscourseJEPA, SentenceLM
# --------------------------------------------------------------------------- #
class _Chunked(Adapter):
    def __init__(self, model, vocab):
        self.model, self.vocab = model, vocab

    def _chunks(self, texts: list[str]) -> torch.Tensor:
        """[1, C, L]; an empty list becomes one all-pad chunk."""
        ids = [self.vocab.encode(t) for t in texts] or [[self.vocab.pad_id]]
        return pad_stack(ids, self.vocab.pad_id).unsqueeze(0)


class DiscourseJEPAAdapter(_Chunked):
    family = "jepa_discourse"

    def __init__(self, model, vocab, name: str, trained: bool = True):
        super().__init__(model, vocab)
        self.name, self.trained = name, trained
        self.d_state = int(model.state_model.d_model) if hasattr(
            model.state_model, "d_model"
        ) else int(next(model.state_model.parameters()).shape[-1])
        self.d_action = int(model.core.d_action)

    @torch.no_grad()
    def encode_states(self, prompt, steps):
        pt = self._chunks(prompt)
        pm = torch.ones(1, pt.shape[1], dtype=torch.bool)
        if steps:
            st = self._chunks(steps)
            sm = torch.ones(1, st.shape[1], dtype=torch.bool)
        else:  # one dummy padded step, masked out
            st = torch.full((1, 1, 1), self.vocab.pad_id, dtype=torch.long)
            sm = torch.zeros(1, 1, dtype=torch.bool)
        s0, states = self.model.encode_states(pt, pm, st, sm)
        if not steps:
            return s0  # [1, D]
        return torch.cat([s0.unsqueeze(1), states], dim=1)[0]  # [T+1, D]

    @torch.no_grad()
    def encode_actions_ctx(self, prompt, steps, phrases):
        # DiscourseJEPA's chunk encoder is context-free by construction: the
        # candidate embedding u(c) does not see the history.  Returning the
        # wide chunk vector keeps the width comparable to the state.
        return self.model.encode_chunks(self._chunks(phrases))[0]

    @torch.no_grad()
    def encode_actions_free(self, phrases):
        return self.model.encode_actions(self._chunks(phrases))[0]

    @torch.no_grad()
    def predict(self, states, actions):
        if actions.shape[-1] != self.d_action:
            return None
        return self.model.predictor(states, actions)


class SentenceLMAdapter(_Chunked):
    family = "lm_sentence"

    def __init__(self, model, vocab, name: str, trained: bool = True):
        super().__init__(model, vocab)
        self.name, self.trained = name, trained
        self.d_state = int(model.latent_head[-1].out_features) if hasattr(
            model.latent_head, "__getitem__"
        ) else int(model.dec_pos.shape[-1])
        self.d_action = self.d_state

    @torch.no_grad()
    def encode_states(self, prompt, steps):
        pe = self.model.encode_chunks(self._chunks(prompt))
        pm = torch.ones(1, pe.shape[1], dtype=torch.bool)
        if steps:
            se = self.model.encode_chunks(self._chunks(steps))
            sm = torch.ones(1, se.shape[1], dtype=torch.bool)
        else:
            se = torch.zeros(1, 1, pe.shape[-1])
            sm = torch.zeros(1, 1, dtype=torch.bool)
        s0, states = self.model.state_model(pe, pm, se, sm)
        if not steps:
            return s0
        return torch.cat([s0.unsqueeze(1), states], dim=1)[0]

    @torch.no_grad()
    def encode_actions_ctx(self, prompt, steps, phrases):
        return self.model.encode_chunks(self._chunks(phrases))[0]

    @torch.no_grad()
    def encode_actions_free(self, phrases):
        """The sentence LM's chunk encoder is context-free by construction."""
        return self.model.encode_chunks(self._chunks(phrases))[0]


# --------------------------------------------------------------------------- #
# flat token families: DecoderLM, FlatIntentJEPA
# --------------------------------------------------------------------------- #
class _Flat(Adapter):
    def __init__(self, model, vocab, max_len: int):
        self.model, self.vocab, self.max_len = model, vocab, max_len

    def _stream(self, texts: list[str]) -> list[int]:
        out: list[int] = []
        for t in texts:
            out.extend(self.vocab.encode(t))
        return out


class TokenLMAdapter(_Flat):
    family = "lm_token"

    def __init__(self, model, vocab, name: str, trained: bool = True):
        super().__init__(model, vocab, int(model.pos.shape[1]))
        self.name, self.trained = name, trained
        self.d_state = int(model.pos.shape[-1])
        self.d_action = self.d_state

    @torch.no_grad()
    def encode_states(self, prompt, steps):
        """State = causal token state at the LAST token of the history so far
        (the position at which the next intent phrase would begin)."""
        stream, ends = self._stream(prompt), []
        ends.append(len(stream) - 1)
        for s in steps:
            stream.extend(self.vocab.encode(s))
            ends.append(len(stream) - 1)
        stream = stream[-self.max_len:]
        shift = max(0, len(self._stream(prompt) + self._stream(steps)) - self.max_len)
        h = self.model.hidden(torch.tensor(stream).unsqueeze(0))[0]
        idx = torch.tensor([max(0, e - shift) for e in ends])
        return h[idx]

    @torch.no_grad()
    def encode_actions_ctx(self, prompt, steps, phrases, batch: int = 32):
        """Hidden state at each phrase's last token, appended to the history
        (position-matched to the flat JEPA).

        All candidates share one history, so they are encoded as a padded
        BATCH rather than one forward each.  The model is causal and pads are
        masked, so trailing padding cannot change a phrase's last-token state
        -- ``test_token_lm_batched_context_matches_single`` pins that.
        """
        hist = self._stream(prompt) + self._stream(steps)
        rows = []
        for start in range(0, len(phrases), batch):
            chunk = phrases[start:start + batch]
            seqs = [(hist + self.vocab.encode(p))[-self.max_len:] for p in chunk]
            toks = pad_stack(seqs, self.vocab.pad_id)
            h = self.model.hidden(toks)
            last = torch.tensor([len(s) - 1 for s in seqs])
            rows.append(h[torch.arange(len(seqs)), last])
        return torch.cat(rows)

    @torch.no_grad()
    def encode_actions_free(self, phrases):
        seqs = [self.vocab.encode(p) for p in phrases]
        h = self.model.hidden(pad_stack(seqs, self.vocab.pad_id))
        last = torch.tensor([len(s) - 1 for s in seqs])
        return h[torch.arange(len(seqs)), last]


class FlatJEPAAdapter(_Flat):
    family = "jepa_flat"

    def __init__(self, model, vocab, name: str, trained: bool = True):
        super().__init__(model, vocab, int(model.encoder.pos.shape[1]))
        self.name, self.trained = name, trained
        self.d_state = int(model.encoder.pos.shape[-1])
        self.d_action = self.d_state

    @torch.no_grad()
    def encode_states(self, prompt, steps):
        stream, ends = self._stream(prompt), []
        ends.append(len(stream) - 1)
        for s in steps:
            stream.extend(self.vocab.encode(s))
            ends.append(len(stream) - 1)
        shift = max(0, len(stream) - self.max_len)
        h = self.model.encode(torch.tensor(stream[-self.max_len:]).unsqueeze(0))[0]
        idx = torch.tensor([max(0, e - shift) for e in ends])
        return h[idx]

    @torch.no_grad()
    def encode_actions_ctx(self, prompt, steps, phrases):
        hist = (self._stream(prompt) + self._stream(steps))[-self.max_len:]
        return self.model.encode_candidates_in_context(
            hist, [self.vocab.encode(p) for p in phrases], torch.device("cpu")
        )

    @torch.no_grad()
    def encode_actions_free(self, phrases):
        """No-history variant of the in-context phrase pass."""
        return self.model.encode_candidates_in_context(
            [self.vocab.pad_id], [self.vocab.encode(p) for p in phrases],
            torch.device("cpu"),
        )

    @torch.no_grad()
    def predict(self, states, actions):
        return self.model.predict(states, actions)


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]


def rebase_data_paths(cfg):
    """Point compiled-corpus paths at THIS checkout.

    Runs trained on another cluster store absolute ``$HOME``-anchored paths in
    their config; the corpora themselves are versioned under ``data/`` here.
    Only the prefix up to ``data/`` is replaced, so the corpus identity (domain,
    split, filename) is untouched.
    """
    for key in ("train_path", "val_path", "test_path"):
        value = cfg.data.get(key) if hasattr(cfg.data, "get") else None
        if not value or Path(value).exists():
            continue
        parts = Path(value).parts
        if "data" in parts:
            rebased = REPO_ROOT.joinpath(*parts[parts.index("data"):])
            if rebased.exists():
                cfg.data[key] = str(rebased)
    return cfg


@dataclass
class LoadedRun:
    adapter: Adapter
    vocab: object
    cfg: object


def load_adapter(
    spec: str, path: str, name: str | None = None, random_init: bool = False,
    device: str = "cpu", flat_lm_init: str | None = None,
) -> LoadedRun:
    """``spec`` in {jepa, flat_jepa, token_lm, sent_lm}.

    ``random_init=True`` rebuilds the architecture from the checkpoint config
    but keeps freshly initialised weights -- the mandatory control for every
    probe (a randomly-wired encoder plus a trainable head can already look
    informative; see the "collusion effect" finding).
    """
    name = name or f"{spec}:{Path(path).parent.parent.name}"
    if spec == "jepa":
        from hydra.utils import instantiate
        from textjepa.utils.checkpoint import (
            _migrate_legacy_state_dict, build_vocab_for_config,
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = rebase_data_paths(OmegaConf.create(ckpt["cfg"]))
        vocab = build_vocab_for_config(cfg)
        model = instantiate(cfg.model, vocab_size=len(vocab), pad_id=vocab.pad_id)
        if not random_init:
            missing, unexpected = model.load_state_dict(
                _migrate_legacy_state_dict(ckpt["model"]), strict=False
            )
            if unexpected:
                raise RuntimeError(f"unexpected checkpoint keys: {unexpected}")
            if missing:
                print(f"note: modules added after this run: {missing}")
        return LoadedRun(
            DiscourseJEPAAdapter(model.to(device).eval(), vocab, name,
                                 not random_init), vocab, cfg
        )
    if spec == "flat_jepa":
        from textjepa.data.faithful import cached_faithful_vocab
        from textjepa.models.flat_intent_jepa import FlatIntentJEPA
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        vocab = cached_faithful_vocab(*ckpt["vocab_caps"])
        mcfg = dict(cfg["model"])
        mcfg["init_from_lm"] = flat_lm_init if random_init is False else None
        if flat_lm_init is not None:
            # "LM-init before JEPA training": build the encoder exactly as the
            # trained run did (weights copied from the token LM), then do NOT
            # load the trained state dict.  Isolates what JEPA training adds
            # to the SAME weights.
            mcfg["init_from_lm"] = flat_lm_init
            model = FlatIntentJEPA(
                vocab_size=len(vocab), pad_id=vocab.pad_id, **mcfg
            )
            return LoadedRun(
                FlatJEPAAdapter(model.eval(), vocab, name, trained=False), vocab, cfg
            )
        mcfg["init_from_lm"] = None
        model = FlatIntentJEPA(vocab_size=len(vocab), pad_id=vocab.pad_id, **mcfg)
        if not random_init:
            model.load_state_dict(ckpt["model"], strict=True)
        return LoadedRun(
            FlatJEPAAdapter(model.eval(), vocab, name, not random_init), vocab, cfg
        )
    if spec in ("token_lm", "sent_lm"):
        from textjepa.models.lm_baseline import DecoderLM
        from textjepa.models.sent_lm import SentenceLM
        from textjepa.utils.checkpoint import build_vocab_for_config
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = rebase_data_paths(OmegaConf.create(ckpt["cfg"]))
        vocab = build_vocab_for_config(cfg)
        cls = DecoderLM if spec == "token_lm" else SentenceLM
        model = cls(vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model)
        if not random_init:
            model.load_state_dict(ckpt["model"])
        model.eval()
        ad = (TokenLMAdapter if spec == "token_lm" else SentenceLMAdapter)(
            model, vocab, name, not random_init
        )
        return LoadedRun(ad, vocab, cfg)
    raise ValueError(f"unknown adapter spec {spec!r}")

