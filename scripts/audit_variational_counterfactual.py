"""Same-state counterfactual coverage for action-free sentence-stream VJEPA.

For each visited state with at least two feasible environment actions, encode
the true rendered next state for every candidate.  Posterior means provide an
outcome-informed reconstruction upper bound.  Samples from p(u|s), available
before the outcome, are then evaluated by how well and how broadly they cover
that fixed candidate set.  This directly tests reusable action modes without
using same-context prior/posterior retrieval as evidence.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import prompt_sentences
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, load_run


def normalized_l1_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.layer_norm(x, x.shape[-1:])
    y = F.layer_norm(y, y.shape[-1:])
    return torch.cdist(x, y, p=1) / x.shape[-1]


class Tokenizer:
    def __init__(self, vocab, device: torch.device):
        self.vocab = vocab
        self.device = device

    def __call__(self, sentences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = [self.vocab.encode(sentence) for sentence in sentences]
        width = max(map(len, encoded))
        tokens = torch.full(
            (1, len(encoded), width), self.vocab.pad_id, dtype=torch.long,
            device=self.device,
        )
        for index, ids in enumerate(encoded):
            tokens[0, index, : len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=self.device
            )
        mask = torch.ones(
            1, len(encoded), dtype=torch.bool, device=self.device
        )
        return tokens, mask


@torch.no_grad()
def encode_last(model, tokenizer: Tokenizer, sentences: list[str], teacher: bool):
    tokens, mask = tokenizer(sentences)
    return model._encode(tokens, mask, teacher=teacher)[:, -1]


def average(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


@torch.no_grad()
def audit(args) -> dict:
    device = torch.device(args.device)
    model, vocab, cfg = load_run(args.ckpt, args.device)
    if model.__class__.__name__ != "SentenceStreamVJEPA":
        raise TypeError("counterfactual latent-action audit requires SentenceStreamVJEPA")
    dataset = build_dataset(cfg, vocab, split="val")
    tokenizer = Tokenizer(vocab, device)
    use_teacher = model.target_mode == "ema"

    posterior_match = [0, 0]
    state_candidate_counts: list[float] = []
    posterior_diagonal: list[float] = []
    prior_nearest: list[float] = []
    prior_best_per_candidate: list[float] = []
    prior_to_posterior_ratio: list[float] = []
    assignment_coverage: list[float] = []
    distinct_candidate_coverage: list[float] = []
    posterior_error_coverage: list[float] = []
    posterior_distinct_match: list[float] = []
    posterior_mode_separation: list[float] = []
    prior_mode_coverage: list[float] = []
    necessary_coverage: list[float] = []
    distractor_coverage: list[float] = []
    necessary_candidate_fraction: list[float] = []
    necessary_assignment_rate: list[float] = []
    candidate_separation: list[float] = []

    for episode in range(min(args.n_episodes, len(dataset))):
        problem, _ = dataset.problem(episode)
        faithful = hasattr(problem, "prompt_sentences")
        if faithful:
            from textjepa.data.faithful import FaithfulEnv

            env = FaithfulEnv(problem)
            prompt = problem.prompt_sentences
            necessary = problem.necessary
        else:
            env = SymbolicEnv(problem)
            prompt = prompt_sentences(problem, random.Random(args.seed + episode))
            necessary = problem.query_ancestors
        history: list[str] = []

        while not env.solved:
            feasible = env.feasible_actions()
            if len(feasible) >= 2:
                current = encode_last(
                    model, tokenizer, prompt + history, teacher=False
                )
                true_next = torch.cat([
                    encode_last(
                        model,
                        tokenizer,
                        prompt + history + [env.clone().step(action)],
                        teacher=use_teacher,
                    )
                    for action in feasible
                ])
                n_candidates = len(feasible)
                repeated = current.expand(n_candidates, -1)

                q_mu, _ = model.var_action._split(
                    model.var_action.post(torch.cat([repeated, true_next], -1))
                )
                posterior_pred, _ = model.transition(repeated, q_mu)
                posterior_dist = normalized_l1_matrix(
                    posterior_pred, true_next
                )
                posterior_match[0] += int(
                    (posterior_dist.argmin(1) == torch.arange(
                        n_candidates, device=device
                    )).sum()
                )
                posterior_match[1] += n_candidates
                diagonal = posterior_dist.diagonal()

                samples = model.var_action.sample_prior(
                    current, k=args.prior_samples
                ).squeeze(0)
                prior_pred, _ = model.transition(
                    current.expand(args.prior_samples, -1), samples
                )
                prior_dist = normalized_l1_matrix(prior_pred, true_next)
                assignments = prior_dist.argmin(1)
                min_per_candidate = prior_dist.amin(0)
                hit = torch.bincount(
                    assignments, minlength=n_candidates
                ).bool()
                target_pairwise = normalized_l1_matrix(true_next, true_next)
                target_pairwise.fill_diagonal_(float("inf"))
                target_radius = 0.5 * target_pairwise.amin(1)
                # A prediction inside half the distance to the nearest
                # competing outcome is unambiguously associated with this
                # candidate.  Unlike a multiple of posterior error, this
                # criterion cannot become easier when the posterior is poor.
                distinct_hit = min_per_candidate < target_radius

                posterior_nearest = posterior_dist.argmin(1)
                posterior_distinct = (
                    (posterior_nearest == torch.arange(
                        n_candidates, device=device
                    ))
                    & (diagonal < target_radius)
                )
                posterior_distinct_match.extend(
                    posterior_distinct.float().cpu().tolist()
                )
                posterior_error_hit = (
                    (min_per_candidate <= diagonal + 1e-3)
                    & posterior_distinct
                )

                # Test whether the prior at least covers the modes learned by
                # the outcome-informed posterior, independently of whether
                # those modes already reconstruct the true outcomes.
                mode_pairwise = normalized_l1_matrix(
                    posterior_pred, posterior_pred
                )
                mode_pairwise.fill_diagonal_(float("inf"))
                mode_radius = 0.5 * mode_pairwise.amin(1)
                prior_to_modes = normalized_l1_matrix(
                    prior_pred, posterior_pred
                )
                mode_hit = prior_to_modes.amin(0) < mode_radius
                is_necessary = torch.tensor(
                    [action in necessary for action in feasible],
                    dtype=torch.bool, device=device,
                )
                is_distractor = ~is_necessary

                state_candidate_counts.append(float(n_candidates))
                posterior_diagonal.extend(diagonal.cpu().tolist())
                prior_nearest.extend(prior_dist.amin(1).cpu().tolist())
                prior_best_per_candidate.extend(min_per_candidate.cpu().tolist())
                prior_to_posterior_ratio.extend(
                    (min_per_candidate / diagonal.clamp(min=1e-6)).cpu().tolist()
                )
                assignment_coverage.append(float(hit.float().mean()))
                distinct_candidate_coverage.append(
                    float(distinct_hit.float().mean())
                )
                posterior_error_coverage.append(
                    float(posterior_error_hit.float().mean())
                )
                posterior_mode_separation.extend(
                    mode_pairwise[torch.isfinite(mode_pairwise)].cpu().tolist()
                )
                prior_mode_coverage.append(float(mode_hit.float().mean()))
                necessary_candidate_fraction.append(
                    float(is_necessary.float().mean())
                )
                if is_necessary.any():
                    necessary_coverage.append(
                        float(distinct_hit[is_necessary].float().mean())
                    )
                if is_distractor.any():
                    distractor_coverage.append(
                        float(distinct_hit[is_distractor].float().mean())
                    )
                necessary_assignment_rate.append(
                    float(is_necessary[assignments].float().mean())
                )
                if n_candidates >= 2:
                    pairwise = torch.pdist(
                        F.layer_norm(true_next, true_next.shape[-1:]), p=1
                    ) / true_next.shape[-1]
                    candidate_separation.extend(pairwise.cpu().tolist())

            next_necessary = [
                action for action in env.feasible_actions()
                if action in necessary
            ]
            history.append(env.step(next_necessary[0]))

    return {
        "checkpoint": args.ckpt,
        "episodes": min(args.n_episodes, len(dataset)),
        "states": len(state_candidate_counts),
        "prior_samples_per_state": args.prior_samples,
        "mean_candidates": average(state_candidate_counts),
        "posterior_candidate_match": posterior_match[0] / max(posterior_match[1], 1),
        "posterior_diagonal_l1": average(posterior_diagonal),
        "mean_true_candidate_separation_l1": average(candidate_separation),
        "prior_nearest_candidate_l1": average(prior_nearest),
        "prior_best_per_candidate_l1": average(prior_best_per_candidate),
        "prior_to_posterior_error_ratio": average(prior_to_posterior_ratio),
        "prior_assignment_coverage": average(assignment_coverage),
        "posterior_distinct_candidate_match": average(
            posterior_distinct_match
        ),
        "prior_distinct_candidate_coverage": average(
            distinct_candidate_coverage
        ),
        "prior_within_posterior_error_coverage": average(
            posterior_error_coverage
        ),
        "mean_posterior_mode_separation_l1": average(
            posterior_mode_separation
        ),
        "prior_distinct_posterior_mode_coverage": average(prior_mode_coverage),
        "necessary_candidate_fraction": average(necessary_candidate_fraction),
        "prior_necessary_distinct_coverage": average(necessary_coverage),
        "prior_distractor_distinct_coverage": average(distractor_coverage),
        "prior_necessary_assignment_rate": average(necessary_assignment_rate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--prior-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    args = parser.parse_args()
    seed_everything(args.seed)
    results = audit(args)
    destination = Path(args.out) if args.out else (
        Path(args.ckpt).parent / "variational_counterfactual_audit.json"
    )
    destination.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
