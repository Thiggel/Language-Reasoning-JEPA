"""Shared LDAD cycle machinery: imagine a displacement, decode/score a phrase.

Every open-ended interface in this repo (``ldad_cycle``, ``generator_cycle``,
``cem_cycle``) rests on the same one-step LDAD cycle: apply the predictor to
``(state, action code)``, take the displacement ``predictor(s, u) - s``, and
push it through the checkpoint's observed-action decoder, which emits one
token distribution per position of the action phrase.  The interfaces differ
only in what they do with those logits:

* candidate interfaces have a *known* phrase per candidate and want its
  teacher-forced mean token log-probability (:func:`phrase_log_probs`);
* the embedding-space CEM has no phrase and must *read one out* of the logits
  by greedy decoding, scoring its own emitted tokens (:func:`greedy_phrases`).

Both live here so there is exactly one predictor -> displacement -> decoder
path and one definition of "mean token log-probability" in the codebase.
"""

from __future__ import annotations

import torch

from textjepa.data.igsm.graph import Problem
from textjepa.data.igsm.render import catalogue_phrases

Tensor = torch.Tensor

# Score assigned to a displacement whose greedy decode is empty (immediate
# PAD): it names no phrase at all and must never win a population.
INVALID_SCORE = -1e4


def require_ldad_decoder(model, context: str):
    """Return the checkpoint's one-step observed-action decoder, or raise."""
    decoder = getattr(model, "observed_action_decoder", None)
    if decoder is None:
        # RuntimeError, not ValueError: the knob is valid, the checkpoint is
        # the wrong one. Matches the pre-existing ldad/generator contract.
        raise RuntimeError(
            f"{context} requires a checkpoint trained with "
            "model.observed_action_ldad=true"
        )
    if getattr(model, "observed_action_ldad_horizon", 1) != 1:
        raise ValueError(f"{context} requires a one-step LDAD decoder")
    return decoder


def delta_logits(
    model,
    state: Tensor,
    codes: Tensor,
    context: str = "LDAD cycle scoring",
) -> Tensor:
    """Decoder logits for the displacements imagined from ``state``.

    ``state`` is a single state ([D] or [1, D]); ``codes`` is [n, d_action].
    Returns [n, max_len, vocab].
    """
    decoder = require_ldad_decoder(model, context)
    flat = state.reshape(1, -1)
    expanded = flat.expand(codes.shape[0], -1)
    return decoder(model.predictor(expanded, codes) - expanded)


def phrase_log_probs(logits: Tensor, token_ids: list[list[int]]) -> Tensor:
    """Mean token log-prob of each given phrase under its row of ``logits``.

    Teacher-forced: row ``i`` is scored against ``token_ids[i]``, truncated to
    the decoder's positions.  This is the score used to rank *known* candidate
    phrases.
    """
    if len(token_ids) != logits.shape[0]:
        raise ValueError("one token sequence per logits row is required")
    log_probs = logits.log_softmax(-1)
    scores = []
    for row, ids in enumerate(token_ids):
        if not ids:
            scores.append(torch.full_like(log_probs[row, 0, 0], INVALID_SCORE))
            continue
        length = min(len(ids), log_probs.shape[1])
        target = torch.tensor(ids[:length], device=log_probs.device)
        scores.append(
            log_probs[row, :length].gather(-1, target.unsqueeze(-1)).mean()
        )
    return torch.stack(scores)


def greedy_phrases(logits: Tensor, vocab) -> tuple[list[str], Tensor]:
    """Greedy-decode a phrase per logits row and score its own tokens.

    The LDAD decoder emits each position from the displacement alone, so
    greedy decoding is a per-position argmax.  A phrase ends at its first PAD;
    its score is the mean log-probability of the emitted (pre-PAD) tokens, on
    the same scale as :func:`phrase_log_probs`.  Rows that decode to nothing
    get :data:`INVALID_SCORE`.
    """
    log_probs = logits.log_softmax(-1)
    chosen = log_probs.argmax(-1)  # [n, max_len]
    token_log_probs = log_probs.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
    # Keep only the prefix before the first PAD.
    prefix = chosen.ne(vocab.pad_id).to(torch.int32).cumprod(dim=-1).bool()
    lengths = prefix.sum(-1)
    scores = torch.where(
        lengths > 0,
        (token_log_probs * prefix).sum(-1) / lengths.clamp_min(1),
        torch.full_like(token_log_probs[:, 0], INVALID_SCORE),
    )
    phrases = [
        vocab.decode(row[:length].tolist())
        for row, length in zip(chosen.cpu(), lengths.cpu().tolist())
    ]
    return phrases, scores


def phrase_tokens(phrases: list[str], vocab, device) -> Tensor:
    """[n] phrases -> [n, 1, L] padded token ids for the action encoder."""
    ids = [vocab.encode(text) for text in phrases]
    # An empty decoded phrase still needs one PAD position to encode.
    L = max(max((len(i) for i in ids), default=1), 1)
    out = torch.full((len(ids), 1, L), vocab.pad_id, dtype=torch.long)
    for row, i in enumerate(ids):
        out[row, 0, : len(i)] = torch.tensor(i)
    return out.to(device)


@torch.no_grad()
def encode_phrases(model, vocab, device, phrases: list[str]) -> Tensor:
    """[n] phrases -> [n, d_action] action codes."""
    return model.encode_actions(phrase_tokens(phrases, vocab, device)).squeeze(1)


@torch.no_grad()
def training_action_codes(
    model, vocab, device, problems: list[Problem]
) -> Tensor:
    """Action embeddings of every action phrase of the given problems.

    The one collection path shared by every eval-time proposal distribution
    fitted without retraining (the ``cem_cycle`` Gaussian prior and the
    ``codebook_cycle`` k-means codebook).  Callers must pass TRAINING problems
    only: evaluation problems would leak their action catalogue into the
    proposal distribution.
    """
    phrases = [
        phrase for problem in problems for phrase in catalogue_phrases(problem)
    ]
    if not phrases:
        raise ValueError("no training action phrases to fit on")
    return encode_phrases(model, vocab, device, phrases)
