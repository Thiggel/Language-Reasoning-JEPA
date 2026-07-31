"""Regression specifications for defects in the shared-cache collector.

These tests intentionally state the required behavior.  They fail against the
implementation under audit until the corresponding defects are repaired.
"""

from hashlib import sha256
from types import SimpleNamespace

import pytest
import torch

from scripts import collect_hierarchical_language_features as collector
from textjepa.analysis.compute import ComputeLedger, inference_flops
from textjepa.data.language_planning import (
    MODEL_ID,
    MODEL_REVISION,
    TRANSFORMERS_VERSION,
)


class _NewlineTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [10]


class _SelectableCache:
    def __init__(self, batch_size):
        self.rows = torch.arange(batch_size)

    def batch_select_indices(self, indices):
        self.rows = self.rows.index_select(0, indices)


class _CountingCachedLM(torch.nn.Module):
    """Cache-capable LM that records the work submitted to each forward."""

    def __init__(self):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.forward_batch_sizes = []
        self.forward_tensor_tokens = 0
        self.decode_rows = 0

    def forward(
        self,
        input_ids,
        *,
        past_key_values=None,
        logits_to_keep,
        **kwargs,
    ):
        del kwargs
        self.forward_batch_sizes.append(len(input_ids))
        self.forward_tensor_tokens += input_ids.numel()
        if past_key_values is None:
            cache = _SelectableCache(len(input_ids))
        else:
            cache = past_key_values
            self.decode_rows += len(input_ids)
        logits = torch.zeros(
            len(input_ids), min(logits_to_keep, input_ids.shape[1]), 16
        )
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _roots(*lengths):
    result = []
    for index, length in enumerate(lengths):
        result.append(
            {
                "example": SimpleNamespace(problem_id=f"p{index}"),
                "ids": list(range(1, length + 1)),
                "boundaries": [length, length + 1],
                "step": 0,
                "root": length,
            }
        )
    return result


def _always_newline(logits, specifications, generated_lengths, generator):
    del logits, generated_lengths, generator
    return torch.full((len(specifications),), 10, dtype=torch.long)


def test_terminal_first_tokens_are_not_submitted_to_an_unneeded_decode(
    monkeypatch,
):
    """The prefill already supplies token-one logits; a terminal token needs no forward."""
    monkeypatch.setattr(
        collector, "_sample_supported_tokens", _always_newline
    )
    model = _CountingCachedLM()

    candidates = collector._generate_with_shared_root_cache(
        _roots(4),
        _NewlineTokenizer(),
        model,
        "cpu",
        seed=0,
        generation_batch_size=7,
        ledger=ComputeLedger(),
        model_parameters=1,
    )

    assert len(candidates) == 7
    assert model.decode_rows == 0


def test_generation_batch_size_is_a_hard_upper_bound(monkeypatch):
    """A user-provided memory cap must not silently expand to seven rows."""
    monkeypatch.setattr(
        collector, "_sample_supported_tokens", _always_newline
    )
    model = _CountingCachedLM()

    collector._generate_with_shared_root_cache(
        _roots(4),
        _NewlineTokenizer(),
        model,
        "cpu",
        seed=0,
        generation_batch_size=1,
        ledger=ComputeLedger(),
        model_parameters=1,
    )

    assert max(model.forward_batch_sizes) <= 1


def test_generation_flops_include_computed_prefill_padding(monkeypatch):
    """Dense padded positions execute model kernels even when attention-masked."""
    monkeypatch.setattr(
        collector, "_sample_supported_tokens", _always_newline
    )
    model = _CountingCachedLM()
    ledger = ComputeLedger()

    collector._generate_with_shared_root_cache(
        _roots(2, 4),
        _NewlineTokenizer(),
        model,
        "cpu",
        seed=0,
        generation_batch_size=14,
        ledger=ledger,
        model_parameters=1,
    )

    measured = ledger.summary()["components"][
        "counterfactual_generation"
    ]["estimated_flops"]
    assert measured == inference_flops(1, model.forward_tensor_tokens)


def _write_resumable_artifacts(tmp_path, **collection_config):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")
    fingerprint = sha256(input_path.read_bytes()).hexdigest()
    common = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "input_fingerprint": fingerprint,
        "shard_index": 0,
        "num_shards": 1,
    }
    output = tmp_path / "features.pt"
    counterfactual_output = tmp_path / "counterfactuals.pt"
    torch.save({**common, "dataset_fingerprint": "dataset"}, output)
    torch.save(
        {
            **common,
            "dataset_fingerprint": "dataset",
            "counterfactual_engine": "shared-cache",
            "candidate_accounting": {"empty_or_failed_slots": 0},
            **collection_config,
        },
        counterfactual_output,
    )
    return input_path, output, counterfactual_output


@pytest.mark.parametrize(
    ("stored", "requested"),
    [
        ({"seed": 11}, {"seed": 12}),
        (
            {"generation_batch_size": 28},
            {"generation_batch_size": 14},
        ),
        (
            {"reencode_batch_size": 32},
            {"reencode_batch_size": 16},
        ),
        ({"dtype": "bfloat16"}, {"dtype": "float16"}),
    ],
    ids=[
        "seed",
        "generation-batch-size",
        "reencode-batch-size",
        "dtype",
    ],
)
def test_resume_rejects_changed_collection_configuration(
    tmp_path, monkeypatch, stored, requested
):
    """Resume must not bless a shard generated under different semantics."""
    paths = _write_resumable_artifacts(tmp_path, **stored)
    args = SimpleNamespace(
        input=paths[0],
        output=paths[1],
        counterfactual_output=paths[2],
        batch_size=4,
        generation_batch_size=28,
        reencode_batch_size=32,
        counterfactual_engine="shared-cache",
        device="cpu",
        dtype="bfloat16",
        seed=11,
        shard_index=0,
        num_shards=1,
        resume=True,
    )
    for name, value in requested.items():
        setattr(args, name, value)
    monkeypatch.setattr(collector, "parse_args", lambda: args)

    with pytest.raises(ValueError, match="incompatible|unbound"):
        collector.main()
