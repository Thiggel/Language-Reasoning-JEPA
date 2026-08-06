"""Diagnose constant and token/action-only collapse in pooled JEPA states.

The current token is the primitive action that produced the current state.
Consequently, healthy token predictability is not enough: this script measures
how much latent variance remains after regressing out current-token identity and
whether that residual remains high-rank across disjoint procedural problems.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.semantic_lm import SemanticBoundaryLMDataset, collate_semantic_lm
from textjepa.models.pooled_sentence_jepa import PooledSentenceJEPA
from textjepa.utils.metrics import effective_rank, feature_std


def _dataset(cfg, vocab, size, seed):
    return SemanticBoundaryLMDataset(
        vocab, size=size, seed=seed, boundary_mode="semantic",
        modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range),
        leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )


@torch.no_grad()
def extract(model, cfg, vocab, examples, seed, batch_size, max_positions, device):
    dataset = _dataset(cfg, vocab, examples, seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=partial(collate_semantic_lm, pad_id=vocab.pad_id),
    )
    states, current, previous, boundary = [], [], [], []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        encoded = model.state_encoder(tokens)
        valid_length = tokens.ne(vocab.pad_id).sum(1)
        for row in range(len(tokens)):
            start, stop = int(batch["prompt_len"][row]), int(valid_length[row])
            positions = torch.arange(start, stop, device=device)
            states.append(encoded[row, positions].float().cpu())
            current.append(tokens[row, positions].cpu())
            previous.append(tokens[row, positions - 1].cpu())
            boundary.append(
                tokens[row, positions].eq(model.period_id).cpu()
                | tokens[row, positions].eq(model.question_id).cpu()
            )
    states = torch.cat(states)[:max_positions]
    current = torch.cat(current)[:max_positions]
    previous = torch.cat(previous)[:max_positions]
    boundary = torch.cat(boundary)[:max_positions]
    return states.numpy(), current.numpy(), previous.numpy(), boundary.numpy()


def one_hot(ids, classes):
    out = np.zeros((len(ids), classes), dtype=np.float32)
    out[np.arange(len(ids)), ids] = 1.0
    return out


def variance_summary(array):
    tensor = torch.from_numpy(array).float()
    return {
        "feature_std": float(feature_std(tensor)),
        "effective_rank": float(effective_rank(tensor[:4096])),
        "mean_squared_radius": float(
            (tensor - tensor.mean(0, keepdim=True)).square().mean()
        ),
    }


def control_regression(train_control, train_state, test_control, test_state):
    scaler = StandardScaler().fit(train_control)
    model = Ridge(alpha=1.0).fit(
        scaler.transform(train_control), train_state
    )
    prediction = model.predict(scaler.transform(test_control))
    residual = test_state - prediction
    return {
        "heldout_variance_explained_r2": float(model.score(
            scaler.transform(test_control), test_state
        )),
        "residual": residual,
    }


def token_decode_accuracy(train_state, train_token, test_state, test_token):
    scaler = StandardScaler().fit(train_state)
    classifier = LogisticRegression(
        max_iter=500, C=1.0, n_jobs=1,
    ).fit(scaler.transform(train_state), train_token)
    return float(classifier.score(scaler.transform(test_state), test_token))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-examples", type=int, default=256)
    parser.add_argument("--test-examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-positions", type=int, default=8192)
    parser.add_argument("--train-seed", type=int, default=510007)
    parser.add_argument("--test-seed", type=int, default=610031)
    args = parser.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = PooledSentenceJEPA(
        len(vocab), vocab.pad_id, period_id=vocab.token_to_id["."],
        question_id=vocab.token_to_id["?"], **cfg.model,
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()

    train = extract(
        model, cfg, vocab, args.train_examples, args.train_seed,
        args.batch_size, args.max_positions, args.device,
    )
    test = extract(
        model, cfg, vocab, args.test_examples, args.test_seed,
        args.batch_size, args.max_positions, args.device,
    )
    train_state, train_current, train_previous, _ = train
    test_state, test_current, test_previous, test_boundary = test
    current = control_regression(
        one_hot(train_current, len(vocab)), train_state,
        one_hot(test_current, len(vocab)), test_state,
    )
    adjacent = control_regression(
        np.concatenate([
            one_hot(train_current, len(vocab)),
            one_hot(train_previous, len(vocab)),
        ], 1),
        train_state,
        np.concatenate([
            one_hot(test_current, len(vocab)),
            one_hot(test_previous, len(vocab)),
        ], 1),
        test_state,
    )
    boundary_state = test_state[test_boundary]
    result = {
        "interpretation": {
            "constant_collapse": (
                "low feature_std and effective_rank indicate global collapse"
            ),
            "action_only_collapse": (
                "current-token R2 near one plus low-rank/low-variance residuals "
                "indicate that the state contains little beyond the primitive action"
            ),
            "healthy_context_dependence": (
                "substantial current-token residual variance/rank and history-probe "
                "performance above token controls refute the action-only explanation"
            ),
        },
        "problem_disjoint_train_test": True,
        "positions": {"train": len(train_state), "test": len(test_state)},
        "all_token_states": variance_summary(test_state),
        "sentence_boundary_states": (
            variance_summary(boundary_state) if len(boundary_state) else None
        ),
        "current_action_only": {
            "heldout_variance_explained_r2":
                current["heldout_variance_explained_r2"],
            "residual": variance_summary(current["residual"]),
        },
        "current_and_previous_token": {
            "heldout_variance_explained_r2":
                adjacent["heldout_variance_explained_r2"],
            "residual": variance_summary(adjacent["residual"]),
        },
        "current_token_linear_decode_accuracy": token_decode_accuracy(
            train_state, train_current, test_state, test_current
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
