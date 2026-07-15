"""Teacher-forced and open-loop drift curves for every hierarchy predictor."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.lm import LMDataset, collate_lm
from textjepa.models.token_hierarchy_v2 import MultilevelTokenHierarchyJEPA
from textjepa.planning.token_cem import latent_l1


def summarize(values):
    return {
        str(h): {"mean": float(np.mean(v)), "p90": float(np.quantile(v, .9)), "n": len(v)}
        for h, v in sorted(values.items()) if v
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--max-horizon", type=int, default=16)
    args = parser.parse_args()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(args.device)
    model.load_state_dict(payload["model"]); model.eval()
    dataset = LMDataset(
        vocab, size=args.examples, seed=cfg.data.val_seed,
        modulus=cfg.data.modulus, n_vars_range=tuple(cfg.data.n_vars_range),
        leaf_prob=cfg.data.leaf_prob, steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    loader = DataLoader(
        dataset, batch_size=16, collate_fn=partial(collate_lm, pad_id=vocab.pad_id)
    )
    curves = None
    with torch.no_grad():
        for batch in loader:
            out = model(batch["tokens"].to(args.device), batch["prompt_len"].to(args.device))
            if curves is None:
                curves = [dict(one_step={}, open_loop={}, composed_low={})
                          for _ in range(1 + len(out["levels"]))]
            sequences = [dict(
                prev=out["prev"], target=out["target"], actions=out["token_actions"],
                valid=out["valid"], predictor=model.low_predictor,
            )]
            for level, module in zip(out["levels"], model.levels):
                sequences.append(dict(
                    prev=level["prev"], target=level["target"],
                    actions=level["codes"], valid=level["valid"],
                    predictor=module.predictor,
                    composed=level["recursive_low_endpoint"],
                ))
            for level_index, sequence in enumerate(sequences):
                valid = sequence["valid"]
                one = sequence["predictor"](
                    sequence["prev"], sequence["actions"], valid
                )
                one_error = latent_l1(one, sequence["target"])
                # A teacher-forced prediction always has rollout horizon one.
                # Pooling it by sequence position and calling that position a
                # horizon would confound non-stationarity with compounding drift.
                if valid.any():
                    curves[level_index]["one_step"].setdefault(1, []).extend(
                        one_error[valid].cpu().tolist()
                    )
                for row in range(valid.shape[0]):
                    length = min(int(valid[row].sum()), args.max_horizon)
                    if not length:
                        continue
                    predicted = sequence["predictor"].rollout(
                        sequence["prev"][row:row + 1, 0],
                        sequence["actions"][row:row + 1, :length],
                    )[0]
                    error = latent_l1(predicted, sequence["target"][row, :length])
                    for horizon, value in enumerate(error, 1):
                        curves[level_index]["open_loop"].setdefault(horizon, []).append(float(value))
                if "composed" in sequence:
                    error = latent_l1(sequence["composed"], sequence["target"])
                    for horizon in range(1, min(args.max_horizon, valid.shape[1]) + 1):
                        mask = valid[:, horizon - 1]
                        if mask.any():
                            curves[level_index]["composed_low"].setdefault(horizon, []).extend(
                                error[:, horizon - 1][mask].cpu().tolist()
                            )
    results = {}
    for level_index, curve in enumerate(curves or []):
        name = "token" if level_index == 0 else f"level{level_index}"
        results[name] = {key: summarize(value) for key, value in curve.items()}
    dest = Path(args.ckpt).parent / "predictor_drift_curves.json"
    dest.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
