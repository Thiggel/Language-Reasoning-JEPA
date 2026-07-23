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
from textjepa.models.discourse_jepa import prepend_factual_prefix
from textjepa.objectives import (
    CounterfactualStatePrediction,
    GeoAdvantageRank,
    GeoAdvantageRegression,
)
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


def test_schema_allows_observed_invalid_counterfactual_without_feasibility_label():
    episode = ObservedActionEpisode.from_dict({
        "episode_id": "invalid-cf", "domain": "x", "split": "train",
        "prompt": ["room"], "goal": "goal",
        "transitions": [{
            "action": "open door", "outcome": "opened",
            "catalogue": ["open door", "take wall"],
            "available": ["open door"],
            "counterfactuals": [{
                "action": "take wall", "outcome": "nothing happens"
            }],
        }],
    })
    assert episode.transitions[0].counterfactuals[0].action == "take wall"


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


def test_dense_geometry_dataset_exposes_every_counterfactual_anchor():
    base = asdict(_episode())
    base["transitions"] = list(base["transitions"])
    base["transitions"].append({
        "action": "apply kind implies round to Bob",
        "outcome": "Bob is round .",
        "catalogue": [
            "apply blue implies kind to Bob",
            "apply kind implies round to Bob",
        ],
        "available": ["apply kind implies round to Bob"],
        "counterfactuals": [{
            "action": "apply blue implies kind to Bob",
            "outcome": "Bob is kind .",
        }],
    })
    episode = ObservedActionEpisode.from_dict(base)
    vocab = build_observed_action_vocab([episode])
    dataset = ObservedActionDataset(
        [episode], vocab, geo_rank_k=1, dense_geo_anchors=True
    )
    assert len(dataset) == 2
    assert {dataset[index]["ga_t"] for index in range(2)} == {0, 1}


def test_counterfactual_prefix_builder_keeps_only_pre_anchor_history():
    factual = torch.tensor([[[11], [12], [13]]])
    factual_mask = torch.tensor([[True, True, True]])
    anchors = torch.tensor([2])
    candidates = torch.tensor([[[[21], [22]], [[31], [0]]]])
    candidate_mask = torch.tensor([[[True, True], [True, False]]])
    tokens, mask = prepend_factual_prefix(
        factual, factual_mask, anchors, candidates, candidate_mask, pad_id=0
    )
    assert tokens.shape == (1, 2, 5, 1)
    assert tokens[0, 0, :, 0].tolist() == [11, 12, 21, 22, 0]
    assert tokens[0, 1, :, 0].tolist() == [11, 12, 31, 0, 0]
    assert mask[0, 0].tolist() == [True, True, True, True, False]
    assert mask[0, 1].tolist() == [True, True, True, False, False]


def test_geometry_alternatives_use_full_causal_prefix_and_state_target():
    base = asdict(_episode())
    base["transitions"] = list(base["transitions"])
    base["transitions"].append({
        "action": "apply kind implies round to Bob",
        "outcome": "Bob is round .",
        "catalogue": [
            "apply blue implies kind to Bob",
            "apply kind implies round to Bob",
        ],
        "available": ["apply kind implies round to Bob"],
        "counterfactuals": [{
            "action": "apply blue implies kind to Bob",
            "outcome": "Bob remains kind .",
            "teacher_rollouts": [["Bob is round ."]],
        }],
    })
    episode = ObservedActionEpisode.from_dict(base)
    vocab = build_observed_action_vocab([episode])
    dataset = ObservedActionDataset(
        [episode], vocab, geo_rank_k=1, geo_rank_horizon=2,
        dense_geo_anchors=True,
    )
    item = next(dataset[index] for index in range(len(dataset))
                if dataset[index]["ga_t"] == 1)
    batch = collate([item], vocab.pad_id)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=32,
        chunk_layers=1, chunk_heads=2, state_layers=1, state_heads=2,
        predictor_layers=1, predictor_heads=2, d_action=8, macro_k=0,
        value_detach=False,
    ).eval()
    out = model(batch)
    alt = model.encode_actions(batch["ga_alt_action_tokens"])
    all_anchors = alt.unsqueeze(1).expand(-1, out.actions.shape[1], -1, -1)
    expected = model.core._predict_counterfactuals(
        out.prev_states, out.actions, all_anchors, out.step_mask
    )[0, 1]
    torch.testing.assert_close(out.extras["ga_cf_pred"][0], expected)
    legacy = model.core.predictor(out.prev_states[0, 1:2], alt[0])
    assert not torch.allclose(out.extras["ga_cf_pred"][0], legacy)
    assert out.extras["ga_cf_target"].shape == out.extras["ga_cf_pred"].shape
    assert out.extras["ga_cf_valid"].tolist() == [[True]]


def test_counterfactual_state_loss_updates_causal_predictor():
    episode = _episode()
    vocab = build_observed_action_vocab([episode])
    batch = collate(
        [ObservedActionDataset(
            [episode], vocab, geo_rank_k=1, geo_rank_horizon=2
        )[0]],
        vocab.pad_id,
    )
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=32,
        chunk_layers=1, chunk_heads=2, state_layers=1, state_heads=2,
        predictor_layers=1, predictor_heads=2, d_action=8, macro_k=0,
        value_detach=True,
    )
    loss = CounterfactualStatePrediction()(model(batch), batch)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.core.predictor.parameters()
    )


def test_dynamic_catalogue_masks_future_discovered_actions():
    episode = ObservedActionEpisode.from_dict({
        "episode_id": "dynamic-1", "domain": "alfworld-textworld",
        "split": "train", "prompt": ["a room"], "goal": "store apple",
        "transitions": [{
            "action": "open cabinet 1", "outcome": "you see apple 1",
            "catalogue": ["open cabinet 1"],
            "available": ["open cabinet 1"],
        }, {
            "action": "take apple 1 from cabinet 1", "outcome": "taken",
            "catalogue": [
                "open cabinet 1", "take apple 1 from cabinet 1",
            ],
            "available": ["take apple 1 from cabinet 1"],
        }],
    })
    vocab = build_observed_action_vocab([episode])
    batch = collate([ObservedActionDataset([episode], vocab)[0]], vocab.pad_id)
    assert batch["action_candidate_observed"].shape == (1, 2, 2)
    assert batch["action_candidate_observed"][0].tolist() == [
        [True, False], [True, True],
    ]


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


def test_no_prior_model_skips_action_support_computation():
    episode = _episode()
    vocab = build_observed_action_vocab([episode])
    batch = collate(
        [ObservedActionDataset([episode], vocab, geo_rank_k=1)[0]],
        vocab.pad_id,
    )
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, d_model=32,
        chunk_layers=1, chunk_heads=2, state_layers=1, state_heads=2,
        predictor_layers=1, predictor_heads=2, d_action=8, macro_k=0,
        action_support_states="none",
    )
    out = model(batch)
    assert "action_support_logits" not in out.extras
