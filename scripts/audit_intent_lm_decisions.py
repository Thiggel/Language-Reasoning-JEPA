"""Aligned native decision audit for token and sentence LM baselines.

The audit follows oracle histories so every checkpoint is scored on identical
states.  A deterministic distractor is injected where possible to create a
matched recovery state.  Feasible actions are supplied to all models; query
ancestry is consulted only after scoring.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from textjepa.analysis.intent_decisions import aggregate_rank_metrics, rank_metrics
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import action_phrase, prompt_sentences
from textjepa.models.lm_baseline import DecoderLM
from textjepa.models.sent_lm import SentenceLM
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import (
    build_dataset,
    build_vocab_for_config,
)


def _padded(vocab, texts: list[str], device: torch.device) -> torch.Tensor:
    encoded = [vocab.encode(text) for text in texts]
    width = max(map(len, encoded), default=1)
    result = torch.full(
        (len(encoded), width), vocab.pad_id, dtype=torch.long, device=device
    )
    for row, sequence in enumerate(encoded):
        result[row, :len(sequence)] = torch.tensor(sequence, device=device)
    return result


class TokenAdapter:
    score_names = ("decoder",)

    def __init__(self, model, vocab, device):
        self.model, self.vocab, self.device = model, vocab, device

    @torch.no_grad()
    def score(self, prompt, history, problem, candidates):
        prefix = [
            token for text in prompt for token in self.vocab.encode(text)
        ]
        for action, outcome in history:
            prefix.extend(self.vocab.encode(action_phrase(problem, action)))
            prefix.extend(self.vocab.encode(outcome))
        candidate_tokens = [
            self.vocab.encode(action_phrase(problem, action))
            for action in candidates
        ]
        width = len(prefix) + max(map(len, candidate_tokens))
        tokens = torch.full(
            (len(candidates), width), self.vocab.pad_id, dtype=torch.long,
            device=self.device,
        )
        for row, candidate in enumerate(candidate_tokens):
            sequence = prefix + candidate
            tokens[row, :len(sequence)] = torch.tensor(
                sequence, device=self.device
            )
        log_probability = self.model.sequence_logprob(
            tokens,
            torch.full((len(candidates),), len(prefix), device=self.device),
        )
        lengths = torch.tensor(
            [len(candidate) for candidate in candidate_tokens],
            device=self.device, dtype=log_probability.dtype,
        ).clamp_min(1)
        cost = -(log_probability / lengths)
        hidden = self.model.hidden(tokens[:1, :len(prefix)])[:, -1]
        return {"decoder": cost.cpu().numpy()}, hidden.cpu().numpy()[0]


class SentenceAdapter:
    def __init__(self, model, vocab, device, latent_target):
        self.model, self.vocab, self.device = model, vocab, device
        self.score_names = (
            ("decoder", "latent") if latent_target else ("decoder",)
        )

    @torch.no_grad()
    def score(self, prompt, history, problem, candidates):
        history_texts = []
        for action, outcome in history:
            history_texts.extend([action_phrase(problem, action), outcome])
        prompt_tokens = _padded(self.vocab, prompt, self.device).unsqueeze(0)
        prompt_mask = torch.ones(
            1, len(prompt), dtype=torch.bool, device=self.device
        )
        if history_texts:
            step_tokens = _padded(
                self.vocab, history_texts, self.device
            ).unsqueeze(0)
            step_mask = torch.ones(
                1, len(history_texts), dtype=torch.bool, device=self.device
            )
        else:
            step_tokens = torch.full(
                (1, 1, 1), self.vocab.pad_id, dtype=torch.long,
                device=self.device,
            )
            step_mask = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
        prompt_embedding = self.model.encode_chunks(prompt_tokens)
        step_embedding = self.model.encode_chunks(step_tokens)
        s0, states = self.model.state_model(
            prompt_embedding, prompt_mask, step_embedding, step_mask
        )
        context = states[:, len(history_texts) - 1] if history_texts else s0
        candidate_tokens = _padded(
            self.vocab,
            [action_phrase(problem, action) for action in candidates],
            self.device,
        )
        lengths = (candidate_tokens != self.vocab.pad_id).sum(-1).clamp_min(1)
        scores = {
            "decoder": (
                self.model.decode_ce(
                    context.expand(len(candidates), -1), candidate_tokens
                ) / lengths
            ).cpu().numpy()
        }
        if self.model.latent_target:
            prediction = self.model.latent_head(context)
            target = self.model.chunk_encoder(candidate_tokens)
            scores["latent"] = (
                F.layer_norm(prediction, prediction.shape[-1:])
                - F.layer_norm(target, target.shape[-1:])
            ).abs().mean(-1).cpu().numpy()
        return scores, context.cpu().numpy()[0]


def _collect(
    adapter, problem, prompt, env, history, *, episode, step, trace, group
):
    candidates = env.feasible_actions()
    labels = np.asarray(
        [action in problem.query_ancestors for action in candidates], dtype=bool
    )
    scores, state = adapter.score(prompt, history, problem, candidates)
    rows = []
    for index, action in enumerate(candidates):
        rows.append({
            "episode": episode,
            "step": step,
            "trace": trace,
            "group": group,
            "action": int(action),
            "operation": str(problem.vars[action].op),
            "necessary": bool(labels[index]),
            "remaining_necessary": int(env.remaining_necessary()),
            "n_candidates": len(candidates),
            "n_positive": int(labels.sum()),
            **{name: float(value[index]) for name, value in scores.items()},
        })
    features = {
        "state": np.repeat(state[None, :], len(candidates), axis=0),
        "necessary": labels,
        "group": np.full(len(candidates), group, dtype=np.int64),
        "episode": np.full(len(candidates), episode, dtype=np.int64),
        "step": np.full(len(candidates), step, dtype=np.int64),
        "action": np.asarray(candidates, dtype=np.int64),
        "forced": np.full(len(candidates), trace != "oracle", dtype=bool),
    }
    return rows, features


def _summary(rows, score_names):
    result = {}
    for trace in sorted({row["trace"] for row in rows}):
        selected = [row for row in rows if row["trace"] == trace]
        result[trace] = {}
        for score in score_names:
            metrics = []
            for group in sorted({row["group"] for row in selected}):
                candidates = [row for row in selected if row["group"] == group]
                metrics.append(rank_metrics(
                    [row[score] for row in candidates],
                    [row["necessary"] for row in candidates],
                ))
            result[trace][score] = aggregate_rank_metrics(metrics)
    return result


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kind", choices=("token_lm", "sentence_lm"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--features-out")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7321)
    parser.add_argument("--forced-errors", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["cfg"])
    if cfg.data.get("name", "igsm") != "igsm":
        raise ValueError("decision audit currently requires stylized iGSM")
    vocab = build_vocab_for_config(cfg)
    if args.kind == "token_lm":
        model = DecoderLM(
            vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        adapter = TokenAdapter(model.eval(), vocab, device)
    else:
        model = SentenceLM(
            vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        adapter = SentenceAdapter(
            model.eval(), vocab, device, bool(cfg.model.latent_target)
        )
    dataset = build_dataset(cfg, vocab, split=args.split, size=args.episodes)
    rows = []
    feature_lists = defaultdict(list)
    group = 0
    for episode in range(args.episodes):
        problem, _ = dataset.problem(episode)
        env = SymbolicEnv(problem)
        prompt = prompt_sentences(problem, random.Random(args.seed + episode))
        history = []
        step = 0
        while not env.solved:
            current_rows, current_features = _collect(
                adapter, problem, prompt, env, history, episode=episode,
                step=step, trace="oracle", group=group,
            )
            rows.extend(current_rows)
            for key, value in current_features.items():
                feature_lists[key].append(value)
            group += 1
            distractors = [
                action for action in env.feasible_actions()
                if action not in problem.query_ancestors
            ]
            if args.forced_errors and distractors:
                branch = env.clone()
                forced = min(distractors)
                outcome = branch.step(forced)
                if not branch.solved:
                    branch_rows, branch_features = _collect(
                        adapter, problem, prompt, branch,
                        history + [(forced, outcome)], episode=episode,
                        step=step + 1, trace="forced_error", group=group,
                    )
                    rows.extend(branch_rows)
                    for key, value in branch_features.items():
                        feature_lists[key].append(value)
                    group += 1
            necessary = [
                action for action in env.feasible_actions()
                if action in problem.query_ancestors
            ]
            chosen = min(necessary)
            history.append((chosen, env.step(chosen)))
            step += 1
    features = {
        key: np.concatenate(value, axis=0)
        for key, value in feature_lists.items()
    }
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "kind": args.kind,
        "split": args.split,
        "episodes": args.episodes,
        "seed": args.seed,
        "information_boundary": (
            "All models score the same symbolic feasible-action menu. Query "
            "ancestry is used only for post-hoc metrics. Forced errors are "
            "deterministic diagnostics and are not model-selected."
        ),
        "summary": _summary(rows, adapter.score_names),
        "rows": rows,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    feature_path = Path(args.features_out or destination.with_suffix(".npz"))
    np.savez_compressed(feature_path, **features)
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {destination} and {feature_path}")


if __name__ == "__main__":
    main()
