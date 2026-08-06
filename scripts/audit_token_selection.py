"""Diagnose whether token selection or latent dynamics limits planning.

The audit enumerates the complete vocabulary.  It uses no symbolic
feasibility labels and no auxiliary language model.  For each held-out state,
it asks where the observed next token ranks under four scores: next-state
geometry, terminal-goal geometry, the learned value, and the optional
state-conditioned token prior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.lm import LMDataset, collate_lm
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA


def rank_of_true(scores: torch.Tensor, true_index: int) -> int:
    """One-indexed rank for a score where smaller is better."""
    true = scores[true_index]
    return int((scores < true).sum()) + 1


def summarize(ranks: list[int]) -> dict[str, float]:
    value = torch.tensor(ranks, dtype=torch.float)
    return {
        "n": len(ranks),
        "top1": float((value <= 1).float().mean()),
        "top5": float((value <= 5).float().mean()),
        "mrr": float((1.0 / value).mean()),
        "mean_rank": float(value.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--positions", type=int, default=256)
    args = parser.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    dataset = LMDataset(
        vocab, size=args.examples, seed=cfg.data.val_seed + 104729,
        modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range),
        leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    loader = DataLoader(
        dataset, batch_size=1,
        collate_fn=lambda rows: collate_lm(rows, pad_id=vocab.pad_id),
    )
    candidate_ids = torch.tensor(
        [i for i in range(len(vocab)) if i != vocab.pad_id],
        device=args.device,
    )
    ranks = {
        "next_state": [], "terminal_goal": [], "remaining_value": [],
        "geometric_value": [], "prior": [],
    }
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["tokens"].to(args.device),
                batch["prompt_len"].to(args.device),
            )
            n = int(out["valid"][0].sum())
            # Deterministic coverage from early, middle, and late positions.
            order = torch.linspace(0, max(n - 1, 0), min(n, 4)).long().unique()
            for position in order.tolist():
                true_id = int(out["action_ids"][0, position])
                true_candidate = int((candidate_ids == true_id).nonzero()[0])
                actions = model.token_action(candidate_ids).unsqueeze(1)
                count = len(candidate_ids)
                state_history = out["prev"][:, :position + 1].expand(count, -1, -1)
                action_history = out["token_actions"][:, :position].expand(count, -1, -1)
                predicted = model.low_predictor.rollout(
                    out["prev"][:, position].expand(count, -1), actions,
                    state_history=state_history,
                    action_history=action_history,
                )[:, 0]
                target = out["target"][:, position].expand_as(predicted)
                goal = out["final_target"].expand_as(predicted)
                next_score = F.mse_loss(
                    F.layer_norm(predicted, predicted.shape[-1:]),
                    F.layer_norm(target, target.shape[-1:]), reduction="none",
                ).mean(-1)
                goal_score = F.mse_loss(
                    F.layer_norm(predicted, predicted.shape[-1:]),
                    F.layer_norm(goal, goal.shape[-1:]), reduction="none",
                ).mean(-1)
                value_score = model.low_value(
                    predicted, out["prompt_state"].expand(count, -1)
                )
                geometric_value = model.low_goal_value(predicted, goal)
                ranks["next_state"].append(rank_of_true(next_score, true_candidate))
                ranks["terminal_goal"].append(rank_of_true(goal_score, true_candidate))
                ranks["remaining_value"].append(
                    rank_of_true(value_score, true_candidate)
                )
                ranks["geometric_value"].append(
                    rank_of_true(geometric_value, true_candidate)
                )
                if out["token_prior_logits"] is not None:
                    logits = out["token_prior_logits"][0, position, candidate_ids]
                    ranks["prior"].append(rank_of_true(-logits, true_candidate))
                if len(ranks["next_state"]) >= args.positions:
                    break
            if len(ranks["next_state"]) >= args.positions:
                break
    result = {
        name: summarize(values) for name, values in ranks.items() if values
    }
    result.update(
        vocabulary_candidates=len(candidate_ids),
        uses_symbolic_feasibility=False,
        uses_auxiliary_lm=False,
    )
    destination = Path(args.ckpt).parent / "token_selection_audit.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
