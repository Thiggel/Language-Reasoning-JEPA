"""Frozen-LM sentence proposals and exact grounding for hierarchical MPC."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from textjepa.data.language_planning import (
    GENERATION_EOS_TOKEN_IDS,
    PAD_TOKEN_ID,
    STEP_DELIMITER,
)


@dataclass(frozen=True)
class GroundedSentenceCandidates:
    tokens: Tensor
    mask: Tensor
    lengths: Tensor
    log_probabilities: Tensor
    endpoint_hidden: Tensor
    terminal: Tensor


def trim_reasoning_candidate(
    token_ids: list[int], newline_ids: list[int], *, maximum: int
) -> tuple[list[int], bool, bool]:
    """Trim at the first genuine delimiter/EOS, independent of min length."""

    for index, token in enumerate(token_ids[:maximum]):
        if token in GENERATION_EOS_TOKEN_IDS:
            return token_ids[:index], True, True
        end = index + 1
        if newline_ids and token_ids[max(0, end - len(newline_ids)):end] == (
            newline_ids
        ):
            return token_ids[:end], True, False
    return token_ids[:maximum], False, False


def _stopping_criteria(newline_ids: list[int], prompt_width: int):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class BoundaryStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            generated = input_ids[:, prompt_width:]
            stopped = torch.zeros(
                len(generated), dtype=torch.bool, device=input_ids.device
            )
            if newline_ids:
                width = len(newline_ids)
                delimiter = torch.tensor(
                    newline_ids, device=input_ids.device
                )
                enough = generated.shape[1] >= width
                if enough:
                    stopped |= (generated[:, -width:] == delimiter).all(-1)
            for token in GENERATION_EOS_TOKEN_IDS:
                stopped |= generated[:, -1].eq(token)
            return stopped

    return StoppingCriteriaList([BoundaryStop()])


@torch.no_grad()
def generate_complete_reasoning_candidates(
    frozen_model,
    tokenizer,
    prefix: Tensor,
    *,
    population: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    reference: Tensor | None = None,
    max_sampling_rounds: int = 8,
) -> list[tuple[Tensor, bool]]:
    """Generate complete newline/EOS actions, optionally including an oracle.

    Returned booleans mark EOS-terminated actions. Incomplete mid-sentence
    samples are rejected rather than entering sentence-level planning.
    """

    if population < 1 or max_tokens < 1 or max_sampling_rounds < 1:
        raise ValueError("candidate generation sizes must be positive")
    if temperature <= 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("candidate sampling parameters are invalid")
    if prefix.ndim != 1 or len(prefix) < 1:
        raise ValueError("candidate prefix must be a nonempty token vector")
    newline = tokenizer.encode(STEP_DELIMITER, add_special_tokens=False)
    candidates: list[tuple[Tensor, bool]] = []
    if reference is not None:
        values, complete, terminal = trim_reasoning_candidate(
            reference.tolist(), newline, maximum=max_tokens
        )
        if not complete or not values:
            raise ValueError("oracle reference is not a complete step")
        candidates.append((torch.tensor(values, dtype=torch.long), terminal))
    needed = population - len(candidates)
    device = prefix.device
    common = {
        "max_new_tokens": max_tokens,
        "pad_token_id": PAD_TOKEN_ID,
        "eos_token_id": list(GENERATION_EOS_TOKEN_IDS),
        "stopping_criteria": _stopping_criteria(newline, len(prefix)),
    }
    # One greedy candidate is part of both worker-gate and deployable pools.
    if needed:
        generated = frozen_model.generate(
            prefix[None], do_sample=False, **common
        )[0, len(prefix):].tolist()
        values, complete, terminal = trim_reasoning_candidate(
            generated, newline, maximum=max_tokens
        )
        if complete and values:
            candidates.append((torch.tensor(values), terminal))
    round_index = 0
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    while len(candidates) < population and round_index < max_sampling_rounds:
        count = max(population - len(candidates), 4)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed + round_index)
            generated = frozen_model.generate(
                prefix[None], do_sample=True,
                temperature=temperature, top_p=top_p, top_k=top_k,
                num_return_sequences=count, **common,
            )[:, len(prefix):]
        for row in generated.tolist():
            values, complete, terminal = trim_reasoning_candidate(
                row, newline, maximum=max_tokens
            )
            if complete and values:
                candidates.append((torch.tensor(values), terminal))
                if len(candidates) == population:
                    break
        round_index += 1
    if len(candidates) != population:
        raise RuntimeError(
            f"only {len(candidates)}/{population} complete candidate steps "
            f"were generated within K0={max_tokens}"
        )
    return candidates


@torch.no_grad()
def exact_ground_sentence_candidates(
    frozen_model,
    prefix: Tensor,
    candidates: list[tuple[Tensor, bool]],
) -> GroundedSentenceCandidates:
    """Compute exact LM endpoint states and suffix log probabilities."""

    if not candidates:
        raise ValueError("candidate population is empty")
    device = prefix.device
    lengths = torch.tensor(
        [len(tokens) for tokens, _ in candidates],
        dtype=torch.long, device=device,
    )
    if bool((lengths < 1).any()):
        raise ValueError("candidate actions must be nonempty")
    suffix_width = int(lengths.max())
    tokens = torch.full(
        (len(candidates), suffix_width), PAD_TOKEN_ID,
        dtype=torch.long, device=device,
    )
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    full_width = len(prefix) + suffix_width
    full = torch.full(
        (len(candidates), full_width), PAD_TOKEN_ID,
        dtype=torch.long, device=device,
    )
    attention = torch.zeros_like(full, dtype=torch.bool)
    for row, (candidate, _) in enumerate(candidates):
        candidate = candidate.to(device)
        length = len(candidate)
        tokens[row, :length] = candidate
        mask[row, :length] = True
        full[row, :len(prefix)] = prefix
        full[row, len(prefix):len(prefix) + length] = candidate
        attention[row, :len(prefix) + length] = True
    output = frozen_model(
        input_ids=full, attention_mask=attention,
        output_hidden_states=True, use_cache=False, return_dict=True,
    )
    hidden = output.hidden_states[-1]
    endpoint = hidden[
        torch.arange(len(candidates), device=device),
        len(prefix) + lengths - 1,
    ]
    # Token x_{n+r} is scored by the logit after prefix length n+r.
    log_probability = torch.zeros(
        len(candidates), device=device, dtype=torch.float32
    )
    for row, length in enumerate(lengths.tolist()):
        logits = output.logits[
            row, len(prefix) - 1:len(prefix) + length - 1
        ].float()
        log_probability[row] = torch.log_softmax(logits, -1).gather(
            -1, tokens[row, :length, None]
        ).sum()
    return GroundedSentenceCandidates(
        tokens=tokens, mask=mask, lengths=lengths,
        log_probabilities=log_probability, endpoint_hidden=endpoint,
        terminal=torch.tensor(
            [terminal for _, terminal in candidates],
            dtype=torch.bool, device=device,
        ),
    )
