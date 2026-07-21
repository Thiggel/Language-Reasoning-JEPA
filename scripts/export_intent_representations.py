"""Export aligned frozen states and probe labels from an intent checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from textjepa.data.igsm.dataset import collate
from textjepa.data.lm import (
    IntentSentencePolicyDataset,
    collate_intent_sentence_policy,
)
from textjepa.models.discourse_jepa import DiscourseJEPA
from textjepa.models.lm_baseline import DecoderLM
from textjepa.models.sent_lm import SentenceLM
from omegaconf import OmegaConf

from textjepa.utils.checkpoint import (
    build_dataset,
    build_vocab_for_config,
    load_run,
)


def _device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _pad(sequences: list[list[int]], length: int, pad_id: int) -> np.ndarray:
    output = np.full((len(sequences), length), pad_id, dtype=np.int64)
    for row, sequence in enumerate(sequences):
        used = sequence[:length]
        output[row, :len(used)] = used
    return output


@torch.no_grad()
def _features(model, item: dict, pad_id: int, device: torch.device) -> torch.Tensor:
    if isinstance(model, DiscourseJEPA):
        batch = _device(collate([item], pad_id), device)
        output = model(batch)
        return output.prev_states[0, batch["step_mask"][0]]
    if isinstance(model, SentenceLM):
        wrapped = IntentSentencePolicyDataset([item])
        batch = _device(
            collate_intent_sentence_policy([wrapped[0]], pad_id), device
        )
        contexts = model.contexts(batch)[0]
        return contexts[batch["target_mask"][0]]
    if isinstance(model, DecoderLM):
        tokens = [token for sentence in item["prompt"] for token in sentence]
        endpoints, stream = [], list(tokens)
        for action, outcome in zip(item["actions"], item["steps"]):
            endpoints.append(len(stream) - 1)
            stream.extend(action)
            stream.extend(outcome)
        ids = torch.tensor(stream, device=device).unsqueeze(0)
        hidden = model.hidden(ids)[0]
        return hidden[torch.tensor(endpoints, device=device)]
    raise TypeError(f"unsupported model type: {type(model).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--kind", choices=("jepa", "token_lm", "sentence_lm"), required=True
    )
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--token-length", type=int, default=96)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["cfg"])
    vocab = build_vocab_for_config(cfg)
    if args.kind == "jepa":
        model, vocab, cfg = load_run(args.checkpoint, str(device))
    elif args.kind == "token_lm":
        model = DecoderLM(
            vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    else:
        model = SentenceLM(
            vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
    dataset = build_dataset(cfg, vocab, split=args.split, size=args.samples)
    features, groups = [], []
    numeric = {name: [] for name in (
        "remaining_steps", "resolved_count", "outcome_value", "step_index",
        "trajectory_length", "feasible_action_count", "catalogue_size",
    )}
    categorical = {name: [] for name in ("operation", "necessary")}
    action_tokens, outcome_tokens = [], []
    for episode_index in range(min(args.samples, len(dataset))):
        item = dataset[episode_index]
        state = _features(model, item, vocab.pad_id, device).cpu().numpy()
        length = len(item["steps"])
        if len(state) != length:
            raise RuntimeError("representation and transition counts differ")
        features.extend(state)
        groups.extend([episode_index] * length)
        numeric["remaining_steps"].extend(item["remaining"])
        numeric["resolved_count"].extend(item["resolved_n"])
        numeric["outcome_value"].extend(item["value"])
        numeric["step_index"].extend(range(length))
        numeric["trajectory_length"].extend([length] * length)
        feasible = item.get("action_feasible", [[] for _ in range(length)])
        numeric["feasible_action_count"].extend(
            [sum(row) if row else 0 for row in feasible]
        )
        catalogue_size = len(item.get("action_candidate_tokens", []))
        numeric["catalogue_size"].extend([catalogue_size] * length)
        categorical["operation"].extend(item["op"])
        categorical["necessary"].extend(item["necessary"])
        action_tokens.extend(item["actions"])
        outcome_tokens.extend(item["steps"])
    arrays = {
        "representations": np.asarray(features, dtype=np.float32),
        "problem_id": np.asarray(groups, dtype=np.int64),
        "target_action_tokens": _pad(
            action_tokens, args.token_length, vocab.pad_id
        ),
        "target_outcome_tokens": _pad(
            outcome_tokens, args.token_length, vocab.pad_id
        ),
        "pad_id": np.asarray(vocab.pad_id),
        "vocab_size": np.asarray(len(vocab)),
    }
    arrays.update({f"numeric_{key}": np.asarray(value) for key, value in numeric.items()})
    arrays.update({f"categorical_{key}": np.asarray(value) for key, value in categorical.items()})
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


if __name__ == "__main__":
    main()
