"""Verify the horizon-conditioned Energy losses are live on a compiled domain.

The headline recipe scores actions with a horizon-conditioned endpoint Energy.
The model only builds that Energy when the batch carries teacher rollout
action tokens, so a compiled domain that does not record them trains with the
recipe's main ranking loss silently at zero.  This check builds real batches
from a compiled dataset, runs the model, and fails unless the horizon ranking
and root-advantage terms are present and non-degenerate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from textjepa.data.igsm.dataset import collate
from textjepa.models import DiscourseJEPA
from textjepa.objectives.ranking import (
    GeoAdvantageRegression,
    GeoHorizonRank,
)
from textjepa.utils.checkpoint import build_dataset, build_vocab_for_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--geo-rank-k", type=int, default=2)
    parser.add_argument("--geo-rank-horizon", type=int, default=8)
    parser.add_argument("--geo-rank-horizons", default="1,2,4,8")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = OmegaConf.load(args.data_config)
    data.geo_rank_k = args.geo_rank_k
    data.geo_rank_horizon = args.geo_rank_horizon
    data.geo_rank_horizons = [
        int(value) for value in args.geo_rank_horizons.split(",") if value
    ]
    data.dense_geo_anchors = True
    cfg = OmegaConf.create({"data": data})
    vocab = build_vocab_for_config(cfg)
    dataset = build_dataset(cfg, vocab, args.split, size=args.episodes)

    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parent.parent / "configs"),
        version_base=None,
    ):
        full = compose(
            config_name="config",
            overrides=[
                "+experiment=paper_gar_scoring_screen",
                "model.d_model=64", "model.chunk_layers=1",
                "model.chunk_heads=4", "model.state_layers=2",
                "model.state_heads=4", "model.predictor_layers=2",
                "model.predictor_heads=4", "model.d_action=16",
                "model.max_chunk_len=96", "model.max_chunks=64",
                "model.geo_horizon_input=false",
                "model.geo_rank_score_mode=horizon",
                "model.dense_rollout_depth=0",
                "model.observed_action_ldad=true",
            ],
        )
    model_cfg = OmegaConf.to_container(full.model, resolve=True)
    model_cfg.pop("_target_", None)
    model = DiscourseJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **model_cfg
    ).train()

    rank = GeoHorizonRank(kind="logistic")
    regression = GeoAdvantageRegression()
    report = {
        "data_config": str(args.data_config),
        "split": args.split,
        "batches": [],
    }
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda items: collate(items, vocab.pad_id),
    )
    for index, batch in enumerate(loader):
        if index >= args.batches:
            break
        if "ga_rollout_action_tokens" not in batch:
            raise SystemExit(
                "batch has no ga_rollout_action_tokens: the horizon Energy "
                "would be silently skipped on this domain"
            )
        out = model(batch)
        if "ga_horizon_energy" not in out.extras:
            raise SystemExit("model did not build a horizon Energy")
        rank_loss = rank(out, batch)
        regression_loss = regression(out, batch)
        report["batches"].append({
            "requested_horizons": sorted(set(
                batch["ga_requested_horizon"].tolist()
            )),
            "rollout_action_steps": int(
                batch["ga_rollout_action_mask"].sum().item()
            ),
            "candidates": int(batch["ga_rollout_action_tokens"].shape[1]),
            "geo_horizon_rank": float(rank_loss),
            "geo_advantage_mse": float(regression_loss),
        })
    if not report["batches"]:
        raise SystemExit("no batches produced")
    if all(value["geo_horizon_rank"] == 0.0 for value in report["batches"]):
        raise SystemExit("horizon ranking loss is identically zero")
    report["horizon_ranking_active"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
