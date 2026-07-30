"""Pinned data/indexing contract for hierarchical language planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence
from collections.abc import Mapping
import hashlib
import random
import re

import torch


# First experiment: newest sub-1B post-trained Qwen available when this
# contract was revised. The revision is immutable even if Hub main advances.
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
TRANSFORMERS_VERSION = "5.13.0"
THINKING_MODE = False
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
PAD_TOKEN_ID = 248044
PRIMARY_EOS_TOKEN_ID = 248046
GENERATION_EOS_TOKEN_IDS = (248046, 248044)
STEP_DELIMITER = "\n"
MIN_STEP_TOKENS = 4
MAX_STEP_TOKENS = 64


class IGSMSplit(str, Enum):
    TRAIN = "train"
    ID_VALIDATION = "id_validation"
    ID_TEST = "id_test"
    NEAR_LENGTH_OOD = "near_length_ood"
    FAR_LENGTH_OOD = "far_length_ood"
    STRUCTURAL_OOD = "structural_ood"
    PARAPHRASE_OOD = "paraphrase_ood"


IGSM_DEPTHS = {
    IGSMSplit.TRAIN: range(2, 7),
    IGSMSplit.ID_VALIDATION: range(2, 7),
    IGSMSplit.ID_TEST: range(2, 7),
    IGSMSplit.NEAR_LENGTH_OOD: range(7, 10),
    IGSMSplit.FAR_LENGTH_OOD: range(10, 13),
    IGSMSplit.STRUCTURAL_OOD: range(2, 7),
    IGSMSplit.PARAPHRASE_OOD: range(2, 7),
}

IGSM_REFERENCE_COUNTS = {
    IGSMSplit.TRAIN: 200_000,
    IGSMSplit.ID_VALIDATION: 25_000,
    IGSMSplit.ID_TEST: 25_000,
    IGSMSplit.NEAR_LENGTH_OOD: 15_000,
    IGSMSplit.FAR_LENGTH_OOD: 15_000,
    IGSMSplit.STRUCTURAL_OOD: 25_000,
    IGSMSplit.PARAPHRASE_OOD: 25_000,
}


@dataclass(frozen=True)
class CounterfactualSamplingCell:
    source: str
    count: int
    temperature: float | None
    top_p: float | None


COUNTERFACTUAL_MIXTURE = (
    CounterfactualSamplingCell("observed", 1, None, None),
    CounterfactualSamplingCell("greedy", 1, 0.0, None),
    CounterfactualSamplingCell("sample_t0.5", 2, 0.5, 0.95),
    CounterfactualSamplingCell("sample_t0.8", 2, 0.8, 0.95),
    CounterfactualSamplingCell("sample_t1.0", 1, 1.0, 0.95),
    CounterfactualSamplingCell("sample_t1.2", 1, 1.2, 0.95),
)


def prompt_token_ids(tokenizer: Any, problem_text: str) -> list[int]:
    """Apply the pinned chat prompt without adding a tokenizer BOS token."""
    ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem_text},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=THINKING_MODE,
    )
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    return list(ids)


def render_igsm_solution(
    reasoning_operations: Sequence[str], answer: str
) -> list[str]:
    if not reasoning_operations:
        raise ValueError("iGSM solution needs at least one reasoning operation")
    lines = [operation.rstrip("\n") + "\n" for operation in reasoning_operations]
    lines.append(f"Final answer: \\boxed{{{answer}}}\n")
    return lines


def normalize_natural_reasoning_steps(
    tokenizer: Any,
    trace: str,
    *,
    max_tokens: int = MAX_STEP_TOKENS,
    min_tokens: int = 4,
) -> list[str]:
    """Create boundary metadata without changing the trace's token content."""
    if not trace:
        raise ValueError("reasoning trace is empty")
    def token_length(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def bounded_chunks(text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            low, high = start + 1, len(text)
            best = None
            while low <= high:
                middle = (low + high) // 2
                if token_length(text[start:middle]) <= max_tokens:
                    best, low = middle, middle + 1
                else:
                    high = middle - 1
            if best is None:
                raise ValueError("one character exceeds the maximum span")
            if best < len(text):
                candidates = [
                    match.end() for match in re.finditer(
                        r"[.!?;:\n]\s*", text[start:best]
                    )
                ]
                if candidates:
                    best = start + candidates[-1]
            chunks.append(text[start:best])
            start = best
        return chunks

    pieces = []
    for line in trace.splitlines(keepends=True):
        pieces.extend(bounded_chunks(line))
    merged: list[str] = []
    index = 0
    while index < len(pieces):
        piece = pieces[index]
        if token_length(piece) < min_tokens:
            if merged and token_length(merged[-1] + piece) <= max_tokens:
                merged[-1] += piece
                index += 1
                continue
            if index + 1 < len(pieces) and token_length(
                piece + pieces[index + 1]
            ) <= max_tokens:
                pieces[index + 1] = piece + pieces[index + 1]
                index += 1
                continue
        merged.append(piece)
        index += 1
    for index, piece in enumerate(merged):
        if token_length(piece) >= min_tokens:
            continue
        if index > 0:
            while token_length(merged[index]) < min_tokens and len(
                merged[index - 1]
            ) > 1:
                moved = merged[index - 1][-1]
                merged[index - 1] = merged[index - 1][:-1]
                merged[index] = moved + merged[index]
        elif len(merged) > 1:
            while token_length(merged[0]) < min_tokens and len(
                merged[1]
            ) > 1:
                moved = merged[1][0]
                merged[0] += moved
                merged[1] = merged[1][1:]
    if not merged:
        raise ValueError("trace produced no reasoning boundaries")
    if "".join(merged) != trace:
        raise ValueError("natural-step normalization changed trace text")
    if any(token_length(piece) > max_tokens for piece in merged):
        raise ValueError("normalized reasoning span exceeds maximum length")
    if len(merged) > 1 and any(
        token_length(piece) < min_tokens for piece in merged
    ):
        raise ValueError("could not satisfy minimum reasoning span length")
    return merged


def _tokenize_full_text_with_boundaries(
    tokenizer: Any,
    step_text: Sequence[str],
) -> tuple[list[int], list[int]]:
    """Tokenize once and admit only character boundaries that are token boundaries."""
    full_text = "".join(step_text)
    full_ids = list(tokenizer.encode(full_text, add_special_tokens=False))
    if not full_ids:
        raise ValueError("solution tokenized to an empty sequence")
    local_boundaries = [0]
    prefix = ""
    for step in step_text:
        prefix += step
        prefix_ids = list(tokenizer.encode(prefix, add_special_tokens=False))
        if full_ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "reasoning-step character boundary is not a stable tokenizer "
                "prefix boundary"
            )
        local_boundaries.append(len(prefix_ids))
    if local_boundaries[-1] != len(full_ids):
        raise ValueError("step boundaries do not reach full solution encoding")
    if any(
        right <= left
        for left, right in zip(local_boundaries[:-1], local_boundaries[1:])
    ):
        raise ValueError("reasoning action tokenized to an empty span")
    return full_ids, local_boundaries


def tokenize_external_steps(
    tokenizer: Any,
    prompt_ids: Sequence[int],
    step_text: Sequence[str],
) -> tuple[list[int], list[int], int]:
    """Tokenize GSM8K/OpenThoughts boundary metadata without editing text."""
    solution_ids, local = _tokenize_full_text_with_boundaries(
        tokenizer, step_text
    )
    if any(
        right - left > MAX_STEP_TOKENS
        for left, right in zip(local[:-1], local[1:])
    ):
        raise ValueError("external reasoning span exceeds maximum token length")
    boundaries = [len(prompt_ids) + offset for offset in local]
    end = boundaries[-1]
    return (
        list(prompt_ids) + solution_ids + [PRIMARY_EOS_TOKEN_ID],
        boundaries,
        end,
    )


def tokenize_steps(
    tokenizer: Any,
    prompt_ids: Sequence[int],
    step_lines: Sequence[str],
) -> tuple[list[int], list[int], int]:
    """Tokenize newline-terminated actions and return global prefix boundaries."""
    if getattr(tokenizer, "add_bos_token", False):
        raise ValueError("pinned tokenizer contract requires add_bos_token=False")
    for line in step_lines:
        if not line.endswith(STEP_DELIMITER):
            raise ValueError("every reasoning action must terminate in newline")
    solution_ids, local = _tokenize_full_text_with_boundaries(
        tokenizer, step_lines
    )
    if any(
        right - left > MAX_STEP_TOKENS
        for left, right in zip(local[:-1], local[1:])
    ):
        raise ValueError("iGSM reasoning span exceeds maximum token length")
    boundaries = [len(prompt_ids) + offset for offset in local]
    solution_end = len(prompt_ids) + len(solution_ids)
    input_ids = list(prompt_ids) + solution_ids + [PRIMARY_EOS_TOKEN_ID]
    return input_ids, boundaries, solution_end


@dataclass
class LanguagePlanningExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_len: int
    solution_end: int
    step_boundaries: torch.Tensor
    reasoning_depth: int
    canonical_state_ids: torch.Tensor
    problem_id: str
    template_family: str
    graph_family: str
    symbolically_verified: bool = True

    def validate(self) -> None:
        if self.input_ids.ndim != 1 or self.input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a LongTensor vector")
        if self.attention_mask.shape != self.input_ids.shape or (
            self.attention_mask.dtype != torch.bool
        ):
            raise ValueError("attention_mask must be an aligned boolean vector")
        if not bool(self.attention_mask.all()):
            raise ValueError("individual examples must be unpadded")
        if not 1 <= self.prompt_len < self.solution_end < len(self.input_ids):
            raise ValueError("invalid prompt/solution endpoints")
        if int(self.input_ids[self.solution_end]) != PRIMARY_EOS_TOKEN_ID:
            raise ValueError("primary EOS must immediately follow solution_end")
        boundaries = self.step_boundaries
        if boundaries.ndim != 1 or boundaries.dtype != torch.long:
            raise ValueError("step_boundaries must be a LongTensor vector")
        if int(boundaries[0]) != self.prompt_len or int(
            boundaries[-1]
        ) != self.solution_end:
            raise ValueError("boundary endpoints violate prompt/solution contract")
        if bool((boundaries[1:] <= boundaries[:-1]).any()):
            raise ValueError("boundaries must be strictly increasing")
        reconstructed = torch.cat([
            self.input_ids[int(left):int(right)]
            for left, right in zip(boundaries[:-1], boundaries[1:])
        ])
        if not torch.equal(
            reconstructed, self.input_ids[self.prompt_len:self.solution_end]
        ):
            raise ValueError("boundary slices do not reconstruct the solution")
        if len(boundaries) != self.reasoning_depth + 2:
            raise ValueError(
                "reasoning_depth must exclude the final answer-emission line"
            )
        if self.canonical_state_ids.shape != boundaries.shape:
            raise ValueError("canonical states must label every boundary")
        if not self.problem_id or not self.template_family or not self.graph_family:
            raise ValueError("diagnostic family metadata is required")
        if not self.symbolically_verified:
            raise ValueError("language-planning examples must be verified")


def collate_language_planning_examples(
    examples: Sequence[LanguagePlanningExample],
) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot collate an empty example list")
    for example in examples:
        example.validate()
    batch = len(examples)
    token_width = max(len(example.input_ids) for example in examples)
    boundary_width = max(len(example.step_boundaries) for example in examples)
    input_ids = torch.full(
        (batch, token_width), PAD_TOKEN_ID, dtype=torch.long
    )
    attention_mask = torch.zeros(batch, token_width, dtype=torch.bool)
    boundaries = torch.full(
        (batch, boundary_width), -1, dtype=torch.long
    )
    boundary_mask = torch.zeros(batch, boundary_width, dtype=torch.bool)
    canonical = torch.full((batch, boundary_width), -1, dtype=torch.long)
    for row, example in enumerate(examples):
        length = len(example.input_ids)
        count = len(example.step_boundaries)
        input_ids[row, :length] = example.input_ids
        attention_mask[row, :length] = True
        boundaries[row, :count] = example.step_boundaries
        boundary_mask[row, :count] = True
        canonical[row, :count] = example.canonical_state_ids
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_len": torch.tensor(
            [example.prompt_len for example in examples], dtype=torch.long
        ),
        "solution_end": torch.tensor(
            [example.solution_end for example in examples], dtype=torch.long
        ),
        "boundaries": boundaries,
        "step_boundaries": boundaries,
        "step_boundary_mask": boundary_mask,
        "reasoning_depth": torch.tensor(
            [example.reasoning_depth for example in examples], dtype=torch.long
        ),
        "canonical_state_ids": canonical,
        "problem_id": [example.problem_id for example in examples],
        "template_family": [example.template_family for example in examples],
        "graph_family": [example.graph_family for example in examples],
        "symbolically_verified": [
            example.symbolically_verified for example in examples
        ],
    }


def validate_igsm_split(
    split: IGSMSplit,
    reasoning_depth: int,
    *,
    graph_held_out: bool = False,
    paraphrase_held_out: bool = False,
) -> None:
    if reasoning_depth not in IGSM_DEPTHS[split]:
        raise ValueError(f"depth {reasoning_depth} is invalid for {split.value}")
    if split is IGSMSplit.STRUCTURAL_OOD and not graph_held_out:
        raise ValueError("structural OOD requires a held-out graph family")
    if split is IGSMSplit.PARAPHRASE_OOD and not paraphrase_held_out:
        raise ValueError("paraphrase OOD requires a held-out template family")


def assign_igsm_split(
    record: dict[str, Any],
    *,
    held_out_graph_families: set[str],
    held_out_template_families: set[str],
) -> IGSMSplit:
    """Deterministically assign a verified record without using text length."""
    depth = int(record["reasoning_depth"])
    if not bool(record.get("symbolically_verified", False)):
        raise ValueError("iGSM split assignment requires symbolic verification")
    graph = str(record["graph_family"])
    template = str(record["template_family"])
    if depth in IGSM_DEPTHS[IGSMSplit.NEAR_LENGTH_OOD]:
        return IGSMSplit.NEAR_LENGTH_OOD
    if depth in IGSM_DEPTHS[IGSMSplit.FAR_LENGTH_OOD]:
        return IGSMSplit.FAR_LENGTH_OOD
    if graph in held_out_graph_families and template in held_out_template_families:
        raise ValueError(
            "structural and paraphrase OOD families must use orthogonal pools"
        )
    if graph in held_out_graph_families:
        validate_igsm_split(
            IGSMSplit.STRUCTURAL_OOD, depth, graph_held_out=True
        )
        return IGSMSplit.STRUCTURAL_OOD
    if template in held_out_template_families:
        validate_igsm_split(
            IGSMSplit.PARAPHRASE_OOD, depth, paraphrase_held_out=True
        )
        return IGSMSplit.PARAPHRASE_OOD
    if depth not in IGSM_DEPTHS[IGSMSplit.TRAIN]:
        raise ValueError("verified symbolic depth lies outside all buckets")
    digest = hashlib.sha256(str(record["problem_id"]).encode()).digest()
    partition = int.from_bytes(digest[:8], "big") % 10
    if partition == 0:
        return IGSMSplit.ID_VALIDATION
    if partition == 1:
        return IGSMSplit.ID_TEST
    return IGSMSplit.TRAIN


def balanced_igsm_manifest(
    records: Sequence[dict[str, Any]],
    split: IGSMSplit,
    count: int,
    *,
    held_out_graph_families: set[str],
    held_out_template_families: set[str],
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Select a fixed, depth-balanced manifest from a verified candidate pool."""
    if count < 1:
        raise ValueError("manifest count must be positive")
    problem_ids = [str(record["problem_id"]) for record in records]
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError("duplicate problem_id values are not allowed")
    eligible = [
        record for record in records
        if assign_igsm_split(
            record,
            held_out_graph_families=held_out_graph_families,
            held_out_template_families=held_out_template_families,
        ) is split
    ]
    depths = list(IGSM_DEPTHS[split])
    base, remainder = divmod(count, len(depths))
    rng = random.Random(seed)
    selected = []
    for offset, depth in enumerate(depths):
        bucket = [
            record for record in eligible
            if int(record["reasoning_depth"]) == depth
        ]
        rng.shuffle(bucket)
        required = base + (1 if offset < remainder else 0)
        if len(bucket) < required:
            raise ValueError(
                f"cannot fill {split.value} depth {depth}: "
                f"need {required}, found {len(bucket)}"
            )
        selected.extend(bucket[:required])
    rng.shuffle(selected)
    return selected


def collate_counterfactual_records(
    records: Sequence[dict[str, Any]],
) -> dict[str, torch.Tensor | list[str]]:
    """Pack exact generated branches for ordinary rectangular learner batches."""
    if not records:
        raise ValueError("cannot collate empty counterfactual records")
    width = max(len(record["token_ids"]) for record in records)
    hidden_width = records[0]["root_hidden"].shape[-1]
    count = len(records)
    tokens = torch.full((count, width), PAD_TOKEN_ID, dtype=torch.long)
    suffix_hidden = records[0]["suffix_hidden"].new_zeros(
        count, width, hidden_width
    )
    lengths = torch.zeros(count, dtype=torch.long)
    root_hidden = records[0]["root_hidden"].new_zeros(count, hidden_width)
    sentence_eligible = torch.zeros(count, dtype=torch.bool)
    terminal = torch.zeros(count, dtype=torch.bool)
    log_probability = torch.zeros(count, dtype=torch.float32)
    temperature = torch.zeros(count, dtype=torch.float32)
    temperature_defined = torch.zeros(count, dtype=torch.bool)
    allowed_sources = {cell.source for cell in COUNTERFACTUAL_MIXTURE}
    token_history_width = max(
        len(record.get("root_token_history_hidden", record["root_hidden"][None]))
        for record in records
    )
    token_history_hidden = records[0]["root_hidden"].new_zeros(
        count, token_history_width, hidden_width
    )
    token_history_actions = torch.full(
        (count, max(0, token_history_width - 1)),
        PAD_TOKEN_ID, dtype=torch.long,
    )
    token_history_lengths = torch.zeros(count, dtype=torch.long)

    sentence_history_width = max(
        len(record.get("root_sentence_history_hidden", record["root_hidden"][None]))
        for record in records
    )
    sentence_history_hidden = records[0]["root_hidden"].new_zeros(
        count, sentence_history_width, hidden_width
    )
    sentence_history_lengths = torch.zeros(count, dtype=torch.long)
    sentence_action_count = max(0, sentence_history_width - 1)
    sentence_history_span_ids = torch.full(
        (count, sentence_action_count, MAX_STEP_TOKENS),
        PAD_TOKEN_ID, dtype=torch.long,
    )
    sentence_history_span_mask = torch.zeros_like(
        sentence_history_span_ids, dtype=torch.bool
    )
    for row, record in enumerate(records):
        if str(record["source"]) not in allowed_sources:
            raise ValueError("unknown counterfactual candidate source")
        candidate_temperature = record.get("temperature")
        if candidate_temperature is not None:
            candidate_temperature = float(candidate_temperature)
            if not torch.isfinite(torch.tensor(candidate_temperature)) or (
                candidate_temperature < 0
            ):
                raise ValueError("counterfactual temperature is invalid")
            temperature[row] = candidate_temperature
            temperature_defined[row] = True
        length = len(record["token_ids"])
        if length < 1 or length > MAX_STEP_TOKENS:
            raise ValueError("counterfactual length violates suffix policy")
        if record["suffix_hidden"].shape != (length, hidden_width):
            raise ValueError("counterfactual exact hidden states do not align")
        if bool(record.get("sentence_eligible", False)) and not bool(
            record.get("completed_boundary", False)
        ):
            raise ValueError("mid-sentence truncation cannot enter sentence replay")
        tokens[row, :length] = record["token_ids"]
        suffix_hidden[row, :length] = record["suffix_hidden"]
        lengths[row] = length
        root_hidden[row] = record["root_hidden"]
        sentence_eligible[row] = bool(record.get("sentence_eligible", False))
        terminal[row] = bool(record.get("terminal_eos", False))
        candidate_logp = torch.as_tensor(
            record.get("lm_log_probability", 0.0), dtype=torch.float32
        )
        if candidate_logp.numel() != 1 or not torch.isfinite(candidate_logp):
            raise ValueError("counterfactual log probability must be finite")
        log_probability[row] = candidate_logp
        token_history = record.get(
            "root_token_history_hidden", record["root_hidden"][None]
        )
        token_history_ids = record.get(
            "root_token_history_action_ids",
            torch.empty(0, dtype=torch.long),
        )
        history_length = len(token_history)
        if token_history.shape != (history_length, hidden_width) or (
            len(token_history_ids) != history_length - 1
        ):
            raise ValueError("token predictor history is misaligned")
        token_history_hidden[row, :history_length] = token_history
        token_history_actions[row, :len(token_history_ids)] = token_history_ids
        token_history_lengths[row] = history_length

        sentence_history = record.get(
            "root_sentence_history_hidden", record["root_hidden"][None]
        )
        sentence_spans = record.get("root_sentence_history_spans", [])
        sentence_length = len(sentence_history)
        if sentence_history.shape != (sentence_length, hidden_width) or (
            len(sentence_spans) != sentence_length - 1
        ):
            raise ValueError("sentence predictor history is misaligned")
        sentence_history_hidden[row, :sentence_length] = sentence_history
        sentence_history_lengths[row] = sentence_length
        for index, span in enumerate(sentence_spans):
            if len(span) < 1 or len(span) > MAX_STEP_TOKENS:
                raise ValueError("sentence history span violates suffix policy")
            sentence_history_span_ids[row, index, :len(span)] = span
            sentence_history_span_mask[row, index, :len(span)] = True
    return {
        "token_ids": tokens,
        "root_hidden": root_hidden,
        "suffix_hidden": suffix_hidden,
        "lengths": lengths,
        "sentence_eligible": sentence_eligible,
        "terminal_eos": terminal,
        "source": [str(record["source"]) for record in records],
        "lm_log_probability": log_probability,
        "temperature": temperature,
        "temperature_defined": temperature_defined,
        "root_token_history_hidden": token_history_hidden,
        "root_token_history_action_ids": token_history_actions,
        "root_token_history_lengths": token_history_lengths,
        "root_sentence_history_hidden": sentence_history_hidden,
        "root_sentence_history_lengths": sentence_history_lengths,
        "root_sentence_history_span_ids": sentence_history_span_ids,
        "root_sentence_history_span_mask": sentence_history_span_mask,
    }


@dataclass(frozen=True)
class PlanningDepthKey:
    dataset_split: str
    symbolic_depth: int
    k0: int
    k1: int
    population_size: int
    cem_iterations: int
    endpoint_kind: str

    def validate(self) -> None:
        if not self.dataset_split:
            raise ValueError("dataset split is required")
        if self.symbolic_depth < 1:
            raise ValueError("symbolic depth must be positive")
        if self.k0 not in {8, 16, 32, 64}:
            raise ValueError("K0 must belong to the mandated planning grid")
        if self.k1 not in {1, 2, 4, 8}:
            raise ValueError("K1 must belong to the mandated planning grid")
        if self.population_size < 1 or self.cem_iterations < 1:
            raise ValueError("planning compute parameters must be positive")
        if self.endpoint_kind not in {"predicted", "exact", "achieved"}:
            raise ValueError("unknown endpoint evaluation kind")
