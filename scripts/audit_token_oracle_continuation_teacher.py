"""Audit non-symbolic N-step GAR teacher alignment before training.

For each held-out prefix, the factual token and prior-supported alternatives
are followed by a small beam of prior-supported continuations. Exact EMA
outcomes prune the beam by terminal-goal distance. This is a privileged target
audit only; it does not use symbolic feasibility or an auxiliary language
model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

try:
    from scripts.train_token_hierarchy_v2 import primitive_oracle_beam_distances
except ModuleNotFoundError:  # Direct execution as ``python scripts/...``.
    from train_token_hierarchy_v2 import primitive_oracle_beam_distances
from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.lm import LMDataset, collate_lm
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA


def rank_of_first(costs: torch.Tensor) -> int:
    return int((costs < costs[0]).sum()) + 1


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float)
    return {
        "mean": float(tensor.mean()),
        "p90": float(torch.quantile(tensor, 0.9)),
        "n": int(len(tensor)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--horizon", type=int, choices=[1, 2, 4, 8], required=True)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--positions", type=int, default=16)
    parser.add_argument("--alternatives", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--beam-branch", type=int, default=4)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    if model.token_prior is None:
        raise ValueError("oracle-continuation audit requires a token prior")

    dataset = LMDataset(
        vocab, size=args.examples, seed=cfg.data.val_seed + 262147,
        modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range), leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    loader = DataLoader(
        dataset, batch_size=1,
        collate_fn=lambda rows: collate_lm(rows, pad_id=vocab.pad_id),
    )
    ranks, regrets, gaps, spreads = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(args.device)
            prompt_len = batch["prompt_len"].to(args.device)
            out = model(tokens, prompt_len)
            length = int(out["valid"][0].sum())
            available = length - (args.horizon - 1)
            if available <= 0:
                continue
            positions = torch.linspace(
                0, available - 1, min(available, 4)
            ).long().unique().tolist()
            for position in positions:
                factual = out["action_ids"][:, position]
                logits = out["token_prior_logits"][:, position].clone()
                logits[:, model.pad_id] = -torch.inf
                logits.scatter_(1, factual[:, None], -torch.inf)
                alternatives = logits.topk(args.alternatives, 1).indices
                roots = torch.cat([factual[:, None], alternatives], 1)
                distance = primitive_oracle_beam_distances(
                    model, tokens, prompt_len, roots, position,
                    out["final_target"].detach(), args.horizon,
                    args.beam_width, args.beam_branch,
                )[0]
                rank = rank_of_first(distance)
                ranks.append(float(rank))
                regrets.append(float(distance[0] - distance.min()))
                sorted_distance = distance.sort().values
                gaps.append(float(sorted_distance[1] - sorted_distance[0]))
                spreads.append(float(distance.std(unbiased=False)))
                if len(ranks) >= args.positions:
                    break
            if len(ranks) >= args.positions:
                break

    rank_tensor = torch.tensor(ranks)
    result = {
        "horizon": args.horizon,
        "reference_rank": {
            "top1": float((rank_tensor <= 1).float().mean()),
            "top5": float((rank_tensor <= 5).float().mean()),
            "mean_rank": float(rank_tensor.mean()),
            "n": int(len(rank_tensor)),
        },
        "reference_regret": summarize(regrets),
        "best_second_gap": summarize(gaps),
        "label_spread": summarize(spreads),
        "metadata": {
            "alternatives": args.alternatives,
            "beam_width": args.beam_width,
            "beam_branch": args.beam_branch,
            "uses_symbolic_feasibility": False,
            "uses_auxiliary_lm": False,
            "privileged_target_construction": True,
        },
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
