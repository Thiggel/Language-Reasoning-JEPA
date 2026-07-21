"""Information-matched closed-loop evaluation on compiled action domains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from textjepa.models.lm_baseline import DecoderLM
from textjepa.models.sent_lm import SentenceLM
from textjepa.planning.catalogue import (
    CatalogueEpisodeResult,
    CatalogueLatentPlanner,
    environment_from_episode,
    environment_from_faithful_problem,
)
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import (
    build_dataset,
    build_vocab_for_config,
    load_run,
)


def _flat(vocab, texts):
    return [token for text in texts for token in vocab.encode(text)]


def _padded(vocab, sequences, device):
    width = max((len(value) for value in sequences), default=1)
    result = torch.full(
        (len(sequences), width), vocab.pad_id, dtype=torch.long,
        device=device,
    )
    for index, value in enumerate(sequences):
        result[index, :len(value)] = torch.tensor(value, device=device)
    return result


class RandomPolicy:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def run_episode(self, environment, excess_actions=0):
        steps = 0
        budget = environment.optimal_length + excess_actions
        while not environment.solved and steps < budget:
            environment.step(self.rng.choice(environment.catalogue))
            steps += 1
        return CatalogueEpisodeResult(
            environment.solved, steps, environment.optimal_length,
            environment.invalid_actions,
        )


class TokenLMPolicy:
    def __init__(self, model, vocab, device):
        self.model, self.vocab, self.device = model, vocab, device

    @torch.no_grad()
    def run_episode(self, environment, excess_actions=0):
        history = _flat(self.vocab, environment.prompt)
        steps = 0
        budget = environment.optimal_length + excess_actions
        while not environment.solved and steps < budget:
            candidates = [self.vocab.encode(value) for value in environment.catalogue]
            width = len(history) + max(map(len, candidates))
            tokens = torch.full(
                (len(candidates), width), self.vocab.pad_id,
                dtype=torch.long, device=self.device,
            )
            for index, candidate in enumerate(candidates):
                sequence = history + candidate
                tokens[index, :len(sequence)] = torch.tensor(
                    sequence, device=self.device
                )
            log_probability = self.model.sequence_logprob(
                tokens,
                torch.full(
                    (len(candidates),), len(history), device=self.device
                ),
            )
            length = torch.tensor(
                [len(value) for value in candidates],
                dtype=log_probability.dtype, device=self.device,
            ).clamp_min(1)
            selected = int((log_probability / length).argmax().item())
            action = environment.catalogue[selected]
            outcome = environment.step(action)
            history += self.vocab.encode(action) + self.vocab.encode(outcome)
            steps += 1
        return CatalogueEpisodeResult(
            environment.solved, steps, environment.optimal_length,
            environment.invalid_actions,
        )


class SentenceLMPolicy:
    def __init__(self, model, vocab, device, score):
        self.model, self.vocab, self.device, self.score = (
            model, vocab, device, score
        )

    def _chunks(self, texts):
        sequences = [self.vocab.encode(value) for value in texts]
        return _padded(self.vocab, sequences, self.device).unsqueeze(0)

    @torch.no_grad()
    def run_episode(self, environment, excess_actions=0):
        history = []
        steps = 0
        budget = environment.optimal_length + excess_actions
        while not environment.solved and steps < budget:
            prompt = self._chunks(environment.prompt)
            prompt_mask = torch.ones(
                1, len(environment.prompt), dtype=torch.bool,
                device=self.device,
            )
            if history:
                step_tokens = self._chunks(history)
                step_mask = torch.ones(
                    1, len(history), dtype=torch.bool, device=self.device
                )
            else:
                step_tokens = torch.full(
                    (1, 1, 1), self.vocab.pad_id, dtype=torch.long,
                    device=self.device,
                )
                step_mask = torch.zeros(
                    1, 1, dtype=torch.bool, device=self.device
                )
            prompt_emb = self.model.encode_chunks(prompt)
            step_emb = self.model.encode_chunks(step_tokens)
            s0, states = self.model.state_model(
                prompt_emb, prompt_mask, step_emb, step_mask
            )
            context = states[:, len(history) - 1] if history else s0
            candidates = [
                self.vocab.encode(value) for value in environment.catalogue
            ]
            candidate_tokens = _padded(
                self.vocab, candidates, self.device
            )
            if self.score == "latent":
                prediction = self.model.latent_head(context)
                embedding = self.model.chunk_encoder(candidate_tokens)
                normalize = lambda value: F.layer_norm(
                    value, value.shape[-1:]
                )
                cost = (
                    normalize(prediction) - normalize(embedding)
                ).abs().mean(-1)
            else:
                cost = self.model.decode_ce(
                    context.expand(len(candidates), -1), candidate_tokens
                ) / (candidate_tokens != self.vocab.pad_id).sum(-1).clamp_min(1)
            action = environment.catalogue[int(cost.argmin().item())]
            outcome = environment.step(action)
            history.extend([action, outcome])
            steps += 1
        return CatalogueEpisodeResult(
            environment.solved, steps, environment.optimal_length,
            environment.invalid_actions,
        )


def _load_policy(args):
    device = torch.device(args.device)
    if args.kind == "random":
        if not args.data_config:
            raise ValueError("random evaluation requires --data-config")
        cfg = OmegaConf.load(args.data_config)
        vocab = build_vocab_for_config(OmegaConf.create({"data": cfg}))
        return RandomPolicy(args.seed), vocab, OmegaConf.create({"data": cfg})
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["cfg"])
    vocab = build_vocab_for_config(cfg)
    if args.kind == "jepa":
        model, vocab, cfg = load_run(args.checkpoint, args.device)
        return CatalogueLatentPlanner(
            model, vocab, device,
            simulation_depth=args.simulation_depth,
            proposal_top_m=args.proposal_top_m,
            beam_width=args.beam_width,
            prior_only=args.prior_only,
            prior_weight=args.prior_weight,
        ), vocab, cfg
    if args.kind == "token_lm":
        model = DecoderLM(
            vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        if args.eval_loops is not None:
            if not hasattr(model.blocks, "eval_loops"):
                raise ValueError("--eval-loops needs a recurrent token LM")
            model.blocks.eval_loops = args.eval_loops
        return TokenLMPolicy(model, vocab, device), vocab, cfg
    model = SentenceLM(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if args.eval_loops is not None:
        encoder = model.state_model.encoder
        if not hasattr(encoder, "eval_loops"):
            raise ValueError("--eval-loops needs a recurrent sentence LM")
        encoder.eval_loops = args.eval_loops
    if args.sentence_score == "latent" and not cfg.model.latent_target:
        raise ValueError("latent scoring needs a latent-target checkpoint")
    return SentenceLMPolicy(
        model, vocab, device, args.sentence_score
    ), vocab, cfg


def _aggregate(results):
    count = len(results)
    total_actions = sum(value.steps for value in results)
    solved = [value for value in results if value.solved]
    return {
        "success": sum(value.solved for value in results) / max(count, 1),
        "mean_steps": sum(value.steps for value in results) / max(count, 1),
        "mean_excess_actions_solved": (
            sum(value.steps - value.optimal_length for value in solved)
            / max(len(solved), 1)
        ),
        "invalid_action_rate": (
            sum(value.invalid_actions for value in results)
            / max(total_actions, 1)
        ),
        "episodes": count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True,
                        choices=("random", "jepa", "token_lm", "sentence_lm"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--data-config")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--excess-actions", type=int, nargs="+", default=(0, 1, 2, 4))
    parser.add_argument("--seed", type=int, default=7321)
    parser.add_argument("--simulation-depth", type=int, default=1)
    parser.add_argument("--proposal-top-m", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--prior-only", action="store_true")
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--sentence-score", choices=("decoder", "latent"),
                        default="decoder")
    parser.add_argument("--eval-loops", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.kind != "random" and not args.checkpoint:
        parser.error("learned policies require --checkpoint")
    seed_everything(args.seed)
    policy, vocab, cfg = _load_policy(args)
    dataset = build_dataset(cfg, vocab, args.split)
    if hasattr(dataset, "episodes"):
        evaluation_items = dataset.episodes[:args.episodes]
        make_environment = environment_from_episode
    elif cfg.data.get("name") == "igsm_real":
        evaluation_items = [
            dataset.problem(index)[0]
            for index in range(min(args.episodes, len(dataset)))
        ]
        make_environment = environment_from_faithful_problem
    else:
        raise ValueError(
            "this evaluator supports compiled domains and faithful iGSM"
        )
    payload = {
        "kind": args.kind,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "candidate_interface": "non_oracle_full_catalogue",
        "candidate_order_seed": args.seed,
        "metrics_by_excess_actions": {},
    }
    for excess in args.excess_actions:
        results = []
        for index, item in enumerate(evaluation_items):
            environment = make_environment(item)
            shuffled = list(environment.catalogue)
            random.Random(
                f"{args.seed}:{index}"
            ).shuffle(shuffled)
            environment.catalogue = tuple(shuffled)
            results.append(policy.run_episode(environment, excess))
        payload["metrics_by_excess_actions"][str(excess)] = _aggregate(results)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
