import json
from dataclasses import asdict

import pytest
import torch
from omegaconf import OmegaConf

from textjepa.data.igsm.dataset import collate
from textjepa.data.observed_action import (
    ObservedActionDataset,
    ObservedActionEpisode,
    build_observed_action_vocab,
    load_observed_action_jsonl,
)
from textjepa.models import DiscourseJEPA
from textjepa.objectives import GeoAdvantageRank, GeoAdvantageRegression
from textjepa.utils.checkpoint import build_dataset, build_vocab_for_config


def _episode():
    return ObservedActionEpisode.from_dict({
        "episode_id": "proof-1",
        "domain": "proofwriter",
        "split": "train",
        "prompt": ["Bob is blue .", "If someone is blue they are kind ."],
        "goal": "Bob is kind .",
        "transitions": [
            {
                "action": "apply blue implies kind to Bob",
                "outcome": "Bob is kind .",
                "catalogue": [
                    "apply blue implies kind to Bob",
                    "apply kind implies round to Bob",
                ],
                "available": [
                    "apply blue implies kind to Bob",
                    "apply kind implies round to Bob",
                ],
                "counterfactuals": [{
                    "action": "apply kind implies round to Bob",
                    "outcome": "rule precondition is not satisfied .",
                    "teacher_rollouts": [["Bob is kind ."]],
                }],
            }
        ],
    })


def test_schema_rejects_oracle_action_missing_from_catalogue():
    value = {
        "action": "hidden expert action",
        "outcome": "done",
        "catalogue": ["public action"],
    }
    with pytest.raises(ValueError, match="absent"):
        ObservedActionEpisode.from_dict({
            "episode_id": "bad", "domain": "x", "split": "train",
            "prompt": ["p"], "goal": "g", "transitions": [value],
        })


def test_jsonl_loader_rejects_duplicate_ids_and_wrong_domain(tmp_path):
    raw = {
        "episode_id": "same", "domain": "proofwriter", "split": "train",
        "prompt": ["p"], "goal": "g",
        "transitions": [{"action": "a", "outcome": "o", "catalogue": ["a"]}],
    }
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(raw) + "\n" + json.dumps(raw) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_observed_action_jsonl(path)
    path.write_text(json.dumps(raw) + "\n")
    with pytest.raises(ValueError, match="expected"):
        load_observed_action_jsonl(path, expected_domain="planbench")


def test_compiled_episode_collates_and_isolates_teacher_rollouts():
    episode = _episode()
    vocab = build_observed_action_vocab([episode])
    dataset = ObservedActionDataset(
        [episode], vocab, geo_rank_k=1, geo_rank_horizon=2, seed=7
    )
    item = dataset[0]
    assert "ga_rollout_steps" in item
    assert "teacher_rollouts" not in item
    batch = collate([item], vocab.pad_id)
    assert batch["ga_alt_action_tokens"].shape[:2] == (1, 1)
    assert batch["ga_rollout_step_tokens"].shape[:3] == (1, 2, 1)


def test_external_config_runs_geometry_value_end_to_end(tmp_path):
    episode = _episode()
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(asdict(episode)) + "\n")
    cfg = OmegaConf.create({"data": {
        "name": "observed_action", "domain": "proofwriter",
        "train_path": str(path), "val_path": str(path),
        "test_path": str(path), "train_seed": 1, "val_seed": 2,
        "geo_rank_k": 1, "geo_rank_horizon": 2,
    }})
    vocab = build_vocab_for_config(cfg)
    dataset = build_dataset(cfg, vocab, "train")
    batch = collate([dataset[0]], vocab.pad_id)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=32,
        chunk_layers=1, chunk_heads=2, state_layers=1, state_heads=2,
        predictor_layers=1, predictor_heads=2, d_action=8, macro_k=0,
        value_detach=False,
    )
    out = model(batch)
    rank = GeoAdvantageRank()(out, batch)
    mse = GeoAdvantageRegression()(out, batch)
    assert torch.isfinite(rank) and torch.isfinite(mse)
    (rank + mse).backward()
    assert any(
        parameter.grad is not None
        for parameter in model.core.value_head.parameters()
    )
