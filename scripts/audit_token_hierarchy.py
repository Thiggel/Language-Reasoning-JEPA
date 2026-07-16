"""Prediction, support, and subgoal audits for the token hierarchy."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.lm import LMDataset, collate_lm
from textjepa.models.token_hierarchy import TokenHierarchyJEPA


def ln_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (F.layer_norm(x, x.shape[-1:]) - F.layer_norm(
        y, y.shape[-1:]
    )).abs().mean(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument("--prior-samples", type=int, default=32)
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = TokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    dataset = LMDataset(
        vocab,
        size=max(args.examples, 512),
        seed=cfg.data.val_seed,
        modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range),
        leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=partial(collate_lm, pad_id=vocab.pad_id),
    )
    totals = {
        "direct_l1": 0.0,
        "recursive_l1": 0.0,
        "prior_mean_l1": 0.0,
        "prior_best_l1": 0.0,
        "posterior_prior_l1": 0.0,
        "retrieval_correct": 0.0,
        "retrieval_n": 0.0,
        "n": 0.0,
    }
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= args.examples:
                break
            out = model(
                batch["tokens"].to(args.device),
                batch["prompt_len"].to(args.device),
            )
            mask = out["high_valid"].reshape(-1)
            if not mask.any():
                continue
            prev = out["high_prev"].reshape(-1, model.d_model)[mask]
            target = out["high_target"].reshape(-1, model.d_model)[mask]
            direct = out["high_pred"].reshape(-1, model.d_model)[mask]
            codes = out["macro_codes"].reshape(
                -1, out["macro_codes"].shape[-1]
            )[mask]
            windows = out["macro_action_windows"].reshape(
                -1,
                model.macro_span,
                out["macro_action_windows"].shape[-1],
            )[mask]
            pm = out["macro_p_mu"].reshape(-1, codes.shape[-1])[mask]
            n = prev.shape[0]

            # Compose the causal token predictor through the complete span,
            # feeding each predicted state back as the next input state.
            state_seq = prev.new_zeros(n, model.macro_span, model.d_model)
            state_seq[:, 0] = prev
            valid = torch.ones(
                n, model.macro_span, dtype=torch.bool, device=prev.device
            )
            final = prev
            for step in range(model.macro_span):
                if step:
                    state_seq[:, step] = final
                pred_seq = model.low_predictor(state_seq, windows, valid)
                final = pred_seq[:, step]

            prior_mean_pred = model.high_predictor(
                prev.unsqueeze(1), pm.unsqueeze(1), valid[:, :1]
            )[:, 0]
            samples = model.macro.sample_prior(prev, args.prior_samples)
            flat_prev = prev.unsqueeze(1).expand(-1, args.prior_samples, -1)
            flat_valid = torch.ones(
                n * args.prior_samples,
                1,
                dtype=torch.bool,
                device=prev.device,
            )
            sample_pred = model.high_predictor(
                flat_prev.reshape(-1, 1, model.d_model),
                samples.reshape(-1, 1, samples.shape[-1]),
                flat_valid,
            )[:, 0].reshape(n, args.prior_samples, model.d_model)
            best = ln_l1(sample_pred, target.unsqueeze(1)).amin(1)

            # In-batch discrete-span retrieval: select which observed macro
            # span produces the true next waypoint from this state.
            cap = min(n, 32)
            candidate = codes[:cap]
            anchors = min(n, cap)
            rep_prev = prev[:anchors].unsqueeze(1).expand(-1, cap, -1)
            rep_code = candidate.unsqueeze(0).expand(anchors, -1, -1)
            pred = model.high_predictor(
                rep_prev.reshape(-1, 1, model.d_model),
                rep_code.reshape(-1, 1, codes.shape[-1]),
                torch.ones(
                    anchors * cap, 1, dtype=torch.bool, device=prev.device
                ),
            )[:, 0].reshape(anchors, cap, model.d_model)
            choice = ln_l1(pred, target[:anchors].unsqueeze(1)).argmin(1)
            retrieval = choice.eq(torch.arange(anchors, device=choice.device))

            totals["direct_l1"] += float(ln_l1(direct, target).sum())
            totals["recursive_l1"] += float(ln_l1(final, target).sum())
            totals["prior_mean_l1"] += float(ln_l1(prior_mean_pred, target).sum())
            totals["prior_best_l1"] += float(best.sum())
            totals["posterior_prior_l1"] += float((codes - pm).abs().mean(-1).sum())
            totals["retrieval_correct"] += float(retrieval.sum())
            totals["retrieval_n"] += anchors
            totals["n"] += n
            seen += batch["tokens"].shape[0]
    n = max(totals.pop("n"), 1.0)
    retrieval_n = max(totals.pop("retrieval_n"), 1.0)
    retrieval_correct = totals.pop("retrieval_correct")
    metrics = {name: value / n for name, value in totals.items()}
    metrics["retrieval_accuracy"] = retrieval_correct / retrieval_n
    metrics["examples"] = seen
    dest = Path(args.ckpt).parent / "hierarchy_audit.json"
    dest.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
