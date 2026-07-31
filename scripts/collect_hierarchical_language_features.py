#!/usr/bin/env python3
"""Collect exact frozen-Qwen features and optional next-step branches.

The collector pins the model revision and Transformers version.  Input JSONL
rows contain rendered iGSM fields: ``problem_text``, ``reasoning_operations``,
``answer``, ``canonical_state_ids``, ``problem_id``, ``template_family``, and
``graph_family``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from textjepa.analysis.compute import (
    ComputeLedger,
    inference_flops,
)
from textjepa.data.language_planning import (
    COUNTERFACTUAL_MIXTURE,
    GENERATION_EOS_TOKEN_IDS,
    MAX_STEP_TOKENS,
    MIN_STEP_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    PAD_TOKEN_ID,
    PRIMARY_EOS_TOKEN_ID,
    STEP_DELIMITER,
    TRANSFORMERS_VERSION,
    LanguagePlanningExample,
    collate_language_planning_examples,
    prompt_token_ids,
    render_igsm_solution,
    normalize_natural_reasoning_steps,
    tokenize_external_steps,
    tokenize_steps,
)
from textjepa.data.provenance import artifact_fingerprint, sha256_file
from textjepa.utils.language_planning_runtime import (
    backend_metadata as _backend_metadata,
    load_reference_model,
    text_parameter_count as _text_parameter_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counterfactual-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--example-limit", type=int, default=0,
        help="Deterministic post-shard feature limit; zero keeps the shard.",
    )
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--reencode-batch-size", type=int, default=32)
    parser.add_argument(
        "--counterfactual-example-limit", type=int, default=0,
        help=(
            "Deterministic rotating-subset collection bound; zero uses every "
            "example in the shard."
        ),
    )
    parser.add_argument(
        "--counterfactual-engine",
        choices=("shared-cache", "legacy"),
        default="shared-cache",
        help=(
            "shared-cache prefills each root once, dynamically removes "
            "finished branches, then independently exact-reencodes them"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_examples(path: Path, tokenizer) -> list[LanguagePlanningExample]:
    examples = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            prompt = prompt_token_ids(tokenizer, row["problem_text"])
            corpus = row.get("corpus", "igsm")
            if corpus == "igsm":
                if not bool(row.get("symbolically_verified", False)):
                    raise ValueError(
                        "iGSM collection requires symbolic verification"
                    )
                operations = row["reasoning_operations"]
                step_lines = render_igsm_solution(
                    operations, str(row["answer"])
                )
                ids, boundaries, solution_end = tokenize_steps(
                    tokenizer, prompt, step_lines
                )
                depth = len(operations)
                canonical = row["canonical_state_ids"]
            elif corpus in {"gsm8k", "openthoughts"}:
                if not bool(row.get("verified", False)):
                    raise ValueError(
                        f"{corpus} collection requires a verified trace"
                    )
                step_lines = normalize_natural_reasoning_steps(
                    tokenizer, row["solution_text"]
                )
                ids, boundaries, solution_end = tokenize_external_steps(
                    tokenizer, prompt, step_lines
                )
                depth = int(row.get("reasoning_depth", len(step_lines) - 1))
                canonical = [-1] * len(boundaries)
            else:
                raise ValueError(f"unknown corpus: {corpus}")
            example = LanguagePlanningExample(
                input_ids=torch.tensor(ids, dtype=torch.long),
                attention_mask=torch.ones(len(ids), dtype=torch.bool),
                prompt_len=len(prompt),
                solution_end=solution_end,
                step_boundaries=torch.tensor(boundaries, dtype=torch.long),
                reasoning_depth=depth,
                canonical_state_ids=torch.tensor(
                    canonical, dtype=torch.long
                ),
                problem_id=str(row["problem_id"]),
                template_family=str(row["template_family"]),
                graph_family=str(row["graph_family"]),
                symbolically_verified=bool(
                    row.get(
                        "symbolically_verified", row.get("verified", False)
                    )
                ),
            )
            example.validate()
            examples.append(example)
    if not examples:
        raise ValueError("input JSONL contains no examples")
    return examples


@torch.no_grad()
def encode_examples(
    examples, model, device: str, batch_size: int,
    ledger: ComputeLedger | None = None,
):
    ledger = ComputeLedger() if ledger is None else ledger
    model_parameters = _text_parameter_count(model)
    chunks = []
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start:start + batch_size]
        batch = collate_language_planning_examples(batch_examples)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with ledger.measure(
            "observed_trace_encoding",
            estimated_flops=inference_flops(
                model_parameters, int(mask.sum())
            ),
            items=len(batch_examples),
        ):
            output = model(
                input_ids=ids,
                attention_mask=mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
        hidden = output.hidden_states[-1].cpu()
        chunks.append((batch, hidden))
    tensor_names = (
        "input_ids", "attention_mask", "prompt_len", "solution_end",
        "boundaries", "step_boundary_mask", "reasoning_depth",
        "canonical_state_ids",
    )
    max_tokens = max(chunk[0]["input_ids"].shape[1] for chunk in chunks)
    max_boundaries = max(chunk[0]["boundaries"].shape[1] for chunk in chunks)
    payload: dict[str, object] = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
    }
    for name in tensor_names:
        values = []
        for batch, _ in chunks:
            value = batch[name]
            target = max_boundaries if "boundar" in name or name == "canonical_state_ids" else max_tokens
            if name in {"prompt_len", "solution_end", "reasoning_depth"}:
                values.append(value)
                continue
            pad = target - value.shape[1]
            fill = False if value.dtype == torch.bool else (
                PAD_TOKEN_ID if name == "input_ids" else -1
            )
            values.append(torch.nn.functional.pad(value, (0, pad), value=fill))
        payload[name] = torch.cat(values)
    hidden_values = []
    for _, hidden in chunks:
        hidden_values.append(torch.nn.functional.pad(
            hidden, (0, 0, 0, max_tokens - hidden.shape[1])
        ))
    payload["hidden_states"] = torch.cat(hidden_values)
    payload["step_boundaries"] = payload["boundaries"]
    for name in ("problem_id", "template_family", "graph_family"):
        payload[name] = [
            getattr(example, name) for example in examples
        ]
    payload["symbolically_verified"] = [
        example.symbolically_verified for example in examples
    ]
    return payload


def _trim_generated(tokenizer, token_ids: list[int]) -> tuple[list[int], bool, bool]:
    newline = tokenizer.encode(STEP_DELIMITER, add_special_tokens=False)
    for index, token in enumerate(token_ids):
        if token in GENERATION_EOS_TOKEN_IDS:
            return token_ids[:index], True, True
        end = index + 1
        if token_ids[max(0, end - len(newline)):end] == newline:
            return token_ids[:end], True, False
    return token_ids[:MAX_STEP_TOKENS], False, False


def _newline_stopping_criteria(tokenizer, prompt_width: int):
    from transformers import StoppingCriteria, StoppingCriteriaList

    newline = tokenizer.encode(
        STEP_DELIMITER, add_special_tokens=False
    )

    class NewlineAfterMinimum(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            generated = input_ids[:, prompt_width:]
            stopped = torch.zeros(
                len(generated), dtype=torch.bool, device=input_ids.device
            )
            for row, tokens in enumerate(generated.tolist()):
                if len(tokens) < MIN_STEP_TOKENS:
                    continue
                stopped[row] = tokens[-len(newline):] == newline
            return stopped

    return StoppingCriteriaList([NewlineAfterMinimum()])


def _materialize_candidates(tokenizer, candidates):
    materialized = []
    for candidate in candidates:
        suffix, complete, eos = _trim_generated(
            tokenizer, candidate["raw"]
        )
        if not suffix:
            continue
        stop_token = next(
            (
                token for token in candidate["raw"]
                if token in GENERATION_EOS_TOKEN_IDS
            ),
            None,
        ) if eos else None
        materialized.append({
            **candidate, "suffix": suffix,
            "scoring_suffix": (
                suffix + [stop_token] if stop_token is not None else suffix
            ),
            "complete": complete, "eos": eos,
        })
    return materialized


@torch.no_grad()
def _exact_reencode_candidates(
    materialized, model, device, reencode_batch_size, ledger,
    model_parameters,
):
    records = []
    for start in range(0, len(materialized), reencode_batch_size):
        chunk = materialized[start:start + reencode_batch_size]
        full_ids = [
            candidate["ids"][:candidate["root"]]
            + candidate["scoring_suffix"]
            for candidate in chunk
        ]
        width = max(map(len, full_ids))
        full = torch.full(
            (len(chunk), width), PAD_TOKEN_ID,
            dtype=torch.long, device=device,
        )
        attention = torch.zeros_like(full, dtype=torch.bool)
        offsets = []
        for row, ids in enumerate(full_ids):
            offset = width - len(ids)
            offsets.append(offset)
            full[row, offset:] = torch.tensor(ids, device=device)
            attention[row, offset:] = True
        kept_logits = min(width, MAX_STEP_TOKENS + 2)
        with ledger.measure(
            "counterfactual_exact_reencoding",
            estimated_flops=inference_flops(
                model_parameters, sum(map(len, full_ids))
            ),
            items=len(chunk),
        ):
            output = model(
                input_ids=full, attention_mask=attention,
                output_hidden_states=True, use_cache=False,
                return_dict=True, logits_to_keep=kept_logits,
            )
        for row, candidate in enumerate(chunk):
            root = candidate["root"]
            suffix = candidate["suffix"]
            scoring_suffix = candidate["scoring_suffix"]
            offset = offsets[row]
            hidden = output.hidden_states[-1][row]
            logit_start = kept_logits - len(scoring_suffix) - 1
            logit_end = kept_logits - 1
            suffix_log_probability = torch.log_softmax(
                output.logits[row, logit_start:logit_end].float(), -1,
            ).gather(
                -1,
                torch.tensor(
                    scoring_suffix, device=output.logits.device
                )[:, None],
            ).sum().cpu()
            example = candidate["example"]
            ids = candidate["ids"]
            boundaries = candidate["boundaries"]
            step = candidate["step"]
            records.append({
                "problem_id": example.problem_id,
                "root_prefix_len": root,
                "source": candidate["source"],
                "temperature": candidate["temperature"],
                "lm_log_probability": suffix_log_probability,
                "token_ids": torch.tensor(suffix, dtype=torch.long),
                "root_hidden": hidden[offset + root - 1].cpu(),
                "suffix_hidden": hidden[
                    offset + root:offset + root + len(suffix)
                ].cpu(),
                "completed_boundary": candidate["complete"],
                "terminal_eos": candidate["eos"],
                "sentence_eligible": candidate["complete"],
                # Exact bounded histories reconstruct xi=(z,kappa).
                "root_token_history_hidden": hidden[
                    offset + max(example.prompt_len, root - 63) - 1:
                    offset + root
                ].cpu(),
                "root_token_history_action_ids": torch.tensor(
                    ids[max(example.prompt_len, root - 63):root],
                    dtype=torch.long,
                ),
                "root_sentence_history_hidden": hidden[
                    torch.tensor(
                        boundaries[max(0, step - 31):step + 1],
                        device=hidden.device,
                    ) + offset - 1
                ].cpu(),
                "root_sentence_history_spans": [
                    torch.tensor(
                        ids[boundaries[index]:boundaries[index + 1]],
                        dtype=torch.long,
                    )
                    for index in range(max(0, step - 31), step)
                ],
            })
    return records


@torch.no_grad()
def _collect_counterfactuals_legacy(
    examples, tokenizer, model, device: str, seed: int,
    ledger: ComputeLedger | None = None,
    generation_batch_size: int = 32,
    reencode_batch_size: int = 32,
):
    """Collect flattened root/candidate batches with exact re-encoding."""
    if generation_batch_size < 1 or reencode_batch_size < 1:
        raise ValueError("counterfactual batch sizes must be positive")
    # Transformers 5.13 does not accept ``generator=`` in ``generate``.
    # Sampling uses the global device RNG, so isolate and seed it explicitly
    # for each flattened sampling cell/batch below.
    ledger = ComputeLedger() if ledger is None else ledger
    model_parameters = _text_parameter_count(model)
    roots = []
    for example in examples:
        ids = example.input_ids.tolist()
        boundaries = example.step_boundaries.tolist()
        for step, root in enumerate(boundaries[:-1]):
            roots.append({
                "example": example, "ids": ids, "boundaries": boundaries,
                "step": step, "root": root,
            })
    candidates = [{
        **root,
        "source": "observed",
        "temperature": None,
        "raw": root["ids"][
            root["root"]:root["boundaries"][root["step"] + 1]
        ],
    } for root in roots]
    sampling_call = 0
    for cell in COUNTERFACTUAL_MIXTURE[1:]:
        requests = [
            root for root in roots for _ in range(cell.count)
        ]
        for start in range(0, len(requests), generation_batch_size):
            chunk = requests[start:start + generation_batch_size]
            width = max(request["root"] for request in chunk)
            prefix = torch.full(
                (len(chunk), width), PAD_TOKEN_ID,
                dtype=torch.long, device=device,
            )
            attention = torch.zeros_like(prefix, dtype=torch.bool)
            for row, request in enumerate(chunk):
                root = request["root"]
                prefix[row, width - root:] = torch.tensor(
                    request["ids"][:root], device=device
                )
                attention[row, width - root:] = True
            kwargs = dict(
                max_new_tokens=MAX_STEP_TOKENS,
                min_new_tokens=MIN_STEP_TOKENS,
                eos_token_id=list(GENERATION_EOS_TOKEN_IDS),
                pad_token_id=PAD_TOKEN_ID,
                attention_mask=attention,
                stopping_criteria=_newline_stopping_criteria(
                    tokenizer, width
                ),
            )
            if cell.temperature == 0:
                kwargs["do_sample"] = False
            else:
                kwargs.update(
                    do_sample=True, temperature=cell.temperature,
                    # The documented mixture is nucleus-only.  Disable the
                    # Transformers library's inherited top-k cutoff.
                    top_p=cell.top_p, top_k=0,
                )
            with ledger.measure(
                "counterfactual_generation",
                estimated_flops=inference_flops(
                    model_parameters,
                    sum(request["root"] + MAX_STEP_TOKENS for request in chunk),
                ),
                items=len(chunk),
            ):
                if cell.temperature == 0:
                    generated = model.generate(prefix, **kwargs)
                else:
                    cuda_devices = []
                    if prefix.device.type == "cuda":
                        cuda_devices = [prefix.device.index or 0]
                    with torch.random.fork_rng(devices=cuda_devices):
                        torch.manual_seed(seed + sampling_call)
                        generated = model.generate(prefix, **kwargs)
                    sampling_call += 1
            for request, generated_row in zip(chunk, generated):
                candidates.append({
                    **request,
                    "source": cell.source,
                    "temperature": cell.temperature,
                    "raw": generated_row[width:].tolist(),
                })
    return _exact_reencode_candidates(
        _materialize_candidates(tokenizer, candidates),
        model, device, reencode_batch_size, ledger, model_parameters,
    )


def _sampling_specifications():
    specifications = []
    for cell in COUNTERFACTUAL_MIXTURE[1:]:
        specifications.extend([cell] * cell.count)
    return specifications


def _sample_supported_tokens(
    logits: torch.Tensor,
    specifications,
    generated_lengths: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply the documented greedy/nucleus policies to heterogeneous rows."""
    from transformers.generation.logits_process import TopPLogitsWarper

    if logits.ndim != 2 or len(logits) != len(specifications):
        raise ValueError("logits/specification batch mismatch")
    scores = logits.float().clone()
    too_short = generated_lengths < MIN_STEP_TOKENS
    if bool(too_short.any()):
        eos = torch.tensor(
            GENERATION_EOS_TOKEN_IDS, device=scores.device
        )
        rows = too_short.nonzero(as_tuple=False).squeeze(-1)
        scores[rows[:, None], eos[None, :]] = -torch.inf
    selected = torch.empty(
        len(scores), dtype=torch.long, device=scores.device
    )
    greedy = torch.tensor(
        [cell.temperature == 0 for cell in specifications],
        dtype=torch.bool, device=scores.device,
    )
    if bool(greedy.any()):
        selected[greedy] = scores[greedy].argmax(-1)
    temperatures = sorted({
        float(cell.temperature) for cell in specifications
        if cell.temperature != 0
    })
    for temperature in temperatures:
        group_indices = [
            index for index, cell in enumerate(specifications)
            if float(cell.temperature) == temperature
        ]
        rows = torch.tensor(
            group_indices,
            dtype=torch.long, device=scores.device,
        )
        group = scores.index_select(0, rows) / temperature
        top_p = float(specifications[group_indices[0]].top_p)
        group = TopPLogitsWarper(top_p)(
            torch.empty(
                len(group), 0, dtype=torch.long, device=scores.device
            ),
            group,
        )
        probabilities = torch.softmax(group, -1)
        selected[rows] = torch.multinomial(
            probabilities, 1, generator=generator
        ).squeeze(-1)
    return selected


def _cache_select(cache, indices: torch.Tensor) -> None:
    """Select hybrid attention/recurrent cache rows with one consistent API."""
    if hasattr(cache, "reorder_cache"):
        cache.reorder_cache(indices)
        # Transformers 5.13's LinearAttentionLayer reorders its tensors but
        # leaves this bookkeeping attribute stale.
        for layer in getattr(cache, "layers", ()):
            if hasattr(layer, "conv_states") or hasattr(
                layer, "recurrent_states"
            ):
                layer.batch_size = len(indices)
        return
    if hasattr(cache, "batch_select_indices"):
        cache.batch_select_indices(indices)
        return
    raise RuntimeError("cache does not support batch selection")


@torch.no_grad()
def _generate_with_shared_root_cache(
    roots, tokenizer, model, device, seed, generation_batch_size,
    ledger, model_parameters,
):
    """Prefill each root once and dynamically decode all seven branches."""
    specifications = _sampling_specifications()
    branches_per_root = len(specifications)
    if branches_per_root != 7:
        raise ValueError("the pinned mixture must have seven generated slots")
    if generation_batch_size < 1:
        raise ValueError("generation batch size must be positive")
    # This is a literal upper bound for both prefill and decoding tensors.
    roots_per_chunk = generation_batch_size
    newline = tokenizer.encode(
        STEP_DELIMITER, add_special_tokens=False
    )
    if not newline:
        raise ValueError("step delimiter tokenization cannot be empty")
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed)
    candidates = []
    processed_tokens = 0
    calls = 0
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
    start_time = perf_counter()
    for chunk_start in range(0, len(roots), roots_per_chunk):
        root_chunk = roots[chunk_start:chunk_start + roots_per_chunk]
        width = max(request["root"] for request in root_chunk)
        prefix = torch.full(
            (len(root_chunk), width), PAD_TOKEN_ID,
            dtype=torch.long, device=device,
        )
        attention = torch.zeros_like(prefix, dtype=torch.bool)
        for row, request in enumerate(root_chunk):
            root = request["root"]
            prefix[row, width - root:] = torch.tensor(
                request["ids"][:root], device=device
            )
            attention[row, width - root:] = True
        position_ids = attention.long().cumsum(-1) - 1
        position_ids.masked_fill_(~attention, 0)
        output = model(
            input_ids=prefix,
            attention_mask=attention,
            position_ids=position_ids,
            output_hidden_states=False,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        calls += 1
        processed_tokens += prefix.numel()
        root_cache = output.past_key_values
        if root_cache is None:
            raise RuntimeError(
                "shared-cache collection requires a selectable Cache"
            )
        scheduled = []
        for local_root in range(len(root_chunk)):
            global_root = chunk_start + local_root
            for cell in specifications:
                scheduled.append({
                    "local_root": local_root,
                    "root_index": global_root,
                    "cell": cell,
                    "tokens": [],
                })
        # A root cache is copied and row-selected for each bounded branch
        # minibatch. Duplicate root indices fork every hybrid cache tensor.
        for branch_start in range(
            0, len(scheduled), generation_batch_size
        ):
            active = scheduled[
                branch_start:branch_start + generation_batch_size
            ]
            root_rows = torch.tensor(
                [branch["local_root"] for branch in active],
                dtype=torch.long, device=output.logits.device,
            )
            cache = copy.deepcopy(root_cache)
            _cache_select(cache, root_rows)
            active_attention = attention.index_select(0, root_rows)
            active_logits = output.logits[:, -1].index_select(
                0, root_rows
            )
            for _ in range(MAX_STEP_TOKENS):
                active_specs = [branch["cell"] for branch in active]
                lengths = torch.tensor(
                    [len(branch["tokens"]) for branch in active],
                    dtype=torch.long, device=active_logits.device,
                )
                next_tokens = _sample_supported_tokens(
                    active_logits, active_specs, lengths, generator
                )
                keep = []
                next_token_values = next_tokens.tolist()
                for row, branch in enumerate(active):
                    token = next_token_values[row]
                    branch["tokens"].append(token)
                    eos = token in GENERATION_EOS_TOKEN_IDS
                    end = len(branch["tokens"])
                    delimited = (
                        branch["tokens"][
                            max(0, end - len(newline)):end
                        ] == newline
                    )
                    done = eos or delimited or end == MAX_STEP_TOKENS
                    if done:
                        candidates.append({
                            **roots[branch["root_index"]],
                            "source": branch["cell"].source,
                            "temperature": branch["cell"].temperature,
                            "raw": branch["tokens"],
                        })
                    else:
                        keep.append(row)
                # Terminal sampled tokens need no forward: exact branch
                # states are obtained later by the cache-free re-encoder.
                if not keep:
                    break
                keep_tensor = torch.tensor(
                    keep, dtype=torch.long, device=active_logits.device
                )
                _cache_select(cache, keep_tensor)
                active_attention = active_attention.index_select(
                    0, keep_tensor
                )
                active = [active[index] for index in keep]
                next_tokens = next_tokens.index_select(
                    0, keep_tensor
                )
                active_attention = torch.cat([
                    active_attention,
                    torch.ones(
                        len(active_attention), 1, dtype=torch.bool,
                        device=active_attention.device,
                    ),
                ], -1)
                next_position = (
                    active_attention.long().sum(-1, keepdim=True) - 1
                )
                output_step = model(
                    input_ids=next_tokens[:, None],
                    attention_mask=active_attention,
                    position_ids=next_position,
                    past_key_values=cache,
                    output_hidden_states=False,
                    use_cache=True,
                    return_dict=True,
                    logits_to_keep=1,
                )
                calls += 1
                processed_tokens += next_tokens.numel()
                cache = output_step.past_key_values
                active_logits = output_step.logits[:, -1]
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))
    ledger.add(
        "counterfactual_generation",
        estimated_flops=inference_flops(
            model_parameters, processed_tokens
        ),
        wall_seconds=perf_counter() - start_time,
        calls=calls,
        items=len(candidates),
    )
    return candidates


@torch.no_grad()
def _collect_counterfactuals_shared_cache(
    examples, tokenizer, model, device, seed, ledger,
    generation_batch_size, reencode_batch_size,
):
    model_parameters = _text_parameter_count(model)
    roots = []
    for example in examples:
        ids = example.input_ids.tolist()
        boundaries = example.step_boundaries.tolist()
        for step, root in enumerate(boundaries[:-1]):
            roots.append({
                "example": example, "ids": ids, "boundaries": boundaries,
                "step": step, "root": root,
            })
    observed = [{
        **root,
        "source": "observed",
        "temperature": None,
        "raw": root["ids"][
            root["root"]:root["boundaries"][root["step"] + 1]
        ],
    } for root in roots]
    generated = _generate_with_shared_root_cache(
        roots, tokenizer, model, device, seed,
        generation_batch_size, ledger, model_parameters,
    )
    return _exact_reencode_candidates(
        _materialize_candidates(tokenizer, observed + generated),
        model, device, reencode_batch_size, ledger, model_parameters,
    )


@torch.no_grad()
def collect_counterfactuals(
    examples, tokenizer, model, device: str, seed: int,
    ledger: ComputeLedger | None = None,
    generation_batch_size: int = 32,
    reencode_batch_size: int = 32,
    engine: str = "auto",
):
    """Collect exact branches with a shared-cache engine and safe test fallback."""
    if generation_batch_size < 1 or reencode_batch_size < 1:
        raise ValueError("counterfactual batch sizes must be positive")
    ledger = ComputeLedger() if ledger is None else ledger
    if engine == "auto":
        model_type = getattr(getattr(model, "config", None), "model_type", "")
        engine = (
            "shared-cache" if model_type == "qwen3_5" else "legacy"
        )
    if engine == "shared-cache":
        return _collect_counterfactuals_shared_cache(
            examples, tokenizer, model, device, seed, ledger,
            generation_batch_size, reencode_batch_size,
        )
    if engine == "legacy":
        return _collect_counterfactuals_legacy(
            examples, tokenizer, model, device, seed, ledger,
            generation_batch_size, reencode_batch_size,
        )
    raise ValueError(f"unknown counterfactual engine: {engine}")


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid collection shard")
    counterfactual_example_limit = getattr(
        args, "counterfactual_example_limit", 0
    )
    example_limit = getattr(args, "example_limit", 0)
    if counterfactual_example_limit < 0:
        raise ValueError("counterfactual example limit must be nonnegative")
    if example_limit < 0:
        raise ValueError("example limit must be nonnegative")
    input_fingerprint = sha256_file(args.input)
    if args.resume and args.output.exists() and (
        args.counterfactual_output is None
        or args.counterfactual_output.exists()
    ):
        existing = torch.load(
            args.output, map_location="cpu", weights_only=True
        )
        expected = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "input_fingerprint": input_fingerprint,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "example_limit": example_limit,
        }
        if any(existing.get(name) != value for name, value in expected.items()):
            raise ValueError("cannot resume an incompatible feature shard")
        if args.counterfactual_output is not None:
            replay = torch.load(
                args.counterfactual_output,
                map_location="cpu", weights_only=True,
            )
            accounting = replay.get("candidate_accounting", {})
            replay_expected = {
                **expected,
                "counterfactual_engine": args.counterfactual_engine,
                "seed": args.seed,
                "generation_batch_size": args.generation_batch_size,
                "reencode_batch_size": args.reencode_batch_size,
                "counterfactual_example_limit": (
                    counterfactual_example_limit
                ),
            }
            if any(
                replay.get(name) != value
                for name, value in replay_expected.items()
            ):
                raise ValueError(
                    "cannot resume an incompatible counterfactual shard"
                )
            if replay.get("dataset_fingerprint") != existing.get(
                "dataset_fingerprint"
            ) or accounting.get("empty_or_failed_slots") != 0:
                raise ValueError(
                    "counterfactual resume artifact is incomplete or unbound"
                )
        return
    tokenizer, model = load_reference_model(args.device, args.dtype)
    examples = read_examples(args.input, tokenizer)
    examples = [
        example for example in examples
        if int.from_bytes(hashlib.sha256(
            example.problem_id.encode()
        ).digest()[:8], "big") % args.num_shards == args.shard_index
    ]
    examples.sort(key=lambda example: hashlib.sha256(
        f"{args.seed}:{example.problem_id}".encode()
    ).digest())
    if example_limit:
        examples = examples[:example_limit]
    if not examples:
        raise ValueError("selected collection shard contains no examples")
    ledger = ComputeLedger()
    payload = encode_examples(
        examples, model, args.device, args.batch_size, ledger
    )
    payload["compute"] = ledger.summary()
    payload["shard_index"] = args.shard_index
    payload["num_shards"] = args.num_shards
    payload["dtype"] = args.dtype
    payload["batch_size"] = args.batch_size
    payload["example_limit"] = example_limit
    payload["input_fingerprint"] = input_fingerprint
    payload["backend"] = _backend_metadata()
    payload["text_model_parameters"] = _text_parameter_count(model)
    payload["dataset_fingerprint"] = artifact_fingerprint(payload, (
        "model_id", "model_revision", "transformers_version",
        "input_fingerprint", "shard_index", "num_shards", "dtype",
        "batch_size", "example_limit",
        "input_ids", "attention_mask", "prompt_len", "solution_end",
        "boundaries", "reasoning_depth", "canonical_state_ids",
        "problem_id", "template_family", "graph_family",
        "symbolically_verified", "hidden_states",
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(payload, args.output)
    if args.counterfactual_output is not None:
        counterfactual_examples = examples
        if (
            counterfactual_example_limit
            and len(counterfactual_examples)
            > counterfactual_example_limit
        ):
            counterfactual_examples = sorted(
                counterfactual_examples,
                key=lambda example: hashlib.sha256(
                    f"{args.seed}:{example.problem_id}".encode()
                ).digest(),
            )[:counterfactual_example_limit]
        records = collect_counterfactuals(
            counterfactual_examples,
            tokenizer, model, args.device, args.seed, ledger,
            args.generation_batch_size, args.reencode_batch_size,
            args.counterfactual_engine,
        )
        expected_slots = 8 * sum(
            len(example.step_boundaries) - 1
            for example in counterfactual_examples
        )
        args.counterfactual_output.parent.mkdir(parents=True, exist_ok=True)
        counterfactual_payload = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "input_fingerprint": input_fingerprint,
            "dataset_fingerprint": payload["dataset_fingerprint"],
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "example_limit": example_limit,
            "seed": args.seed,
            "generation_batch_size": args.generation_batch_size,
            "reencode_batch_size": args.reencode_batch_size,
            "counterfactual_example_limit": (
                counterfactual_example_limit
            ),
            "counterfactual_problem_ids": [
                example.problem_id for example in counterfactual_examples
            ],
            "backend": payload["backend"],
            "counterfactual_engine": args.counterfactual_engine,
            "records": records,
            "compute": ledger.summary(),
            "candidate_accounting": {
                "expected_slots": expected_slots,
                "materialized_nonempty_slots": len(records),
                "empty_or_failed_slots": expected_slots - len(records),
            },
        }
        _atomic_torch_save(
            counterfactual_payload, args.counterfactual_output
        )


if __name__ == "__main__":
    main()
