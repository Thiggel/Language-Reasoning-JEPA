#!/usr/bin/env python3
"""Fit fresh frozen-checkpoint probes for horizons and history-window gap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from textjepa.data.predictive_state import load_token_blocks
from textjepa.models.action_transition import (
    FrozenSufficiencyProbe,
    ResidualCapture,
    decoder_layers,
)
from textjepa.objectives.predictive_state import transition_loss
from textjepa.training.predictive_state import (
    architecture_defaults,
    load_stage1_checkpoint,
    teacher_forward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-blocks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", action="append", help="history,horizon")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    return parser.parse_args()


def parse_cells(values):
    values = values or ["1,1", "8,1", "1,2", "1,4", "1,8"]
    result = []
    for value in values:
        history, horizon = map(int, value.split(","))
        if min(history, horizon) < 1:
            raise ValueError("probe history and horizon must be positive")
        result.append((history, horizon))
    return result


@torch.no_grad()
def sample_features(model, capture, dataset, *, source_layers, target_layer,
                    history, horizon, batch_size, generator, device):
    for _ in range(128):
        indices = torch.randint(len(dataset), (batch_size,), generator=generator)
        tokens = torch.stack([
            dataset[int(index)]["input_ids"] for index in indices
        ]).to(device)
        target_mask = torch.stack([
            dataset[int(index)]["target_mask"] for index in indices
        ])
        low = history - 1
        high = tokens.shape[1] - horizon - 1
        if high < low:
            raise ValueError("token block is too short for probe cell")
        anchor = int(torch.randint(low, high + 1, (), generator=generator))
        begin = anchor - history + 1
        if not bool(target_mask[:, begin:anchor + horizon].all()):
            continue
        exact = teacher_forward(
            model, tokens, capture=capture, attention_mask=None, use_cache=False
        )
        state_history = torch.cat([
            exact.states[layer][:, begin:anchor + 1]
            for layer in source_layers
        ], dim=-1).detach()
        action_ids = tokens[:, anchor + 1:anchor + horizon + 1]
        action = model.get_input_embeddings()(action_ids).detach()
        target = exact.states[target_layer][:, anchor + horizon].detach()
        return state_history, action, target
    raise RuntimeError("could not sample a boundary-safe probe batch")


@torch.no_grad()
def evaluate(probe, model, capture, dataset, *, cell, source_layers,
             target_layer, args, generator):
    values = []
    probe.eval()
    for _ in range(args.eval_batches):
        state, action, target = sample_features(
            model, capture, dataset, source_layers=source_layers,
            target_layer=target_layer, history=cell[0], horizon=cell[1],
            batch_size=args.batch_size, generator=generator,
            device=args.device,
        )
        prediction = probe(state, action)
        loss = transition_loss(
            prediction, target,
            torch.ones(len(target), dtype=torch.bool, device=target.device),
        )
        values.append({
            "cosine": float(loss.cosine),
            "scale": float(loss.scale),
            "normalized_mse": float(loss.normalized_mse),
        })
    probe.train()
    return {
        key: sum(row[key] for row in values) / len(values)
        for key in values[0]
    }


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    model, predictor, payload = load_stage1_checkpoint(
        args.checkpoint, device=args.device, dtype=dtype
    )
    model.requires_grad_(False)
    model.eval()
    defaults = architecture_defaults(payload["model_id"], len(decoder_layers(model)))
    if predictor is None:
        source_layers = tuple(defaults["sources"])
        target_layer = int(defaults["target"])
    else:
        source_layers = predictor.config.source_layers
        target_layer = predictor.config.target_layer
    train_data, metadata = load_token_blocks(args.token_blocks, "train")
    validation_data, _ = load_token_blocks(args.token_blocks, "validation")
    capture = ResidualCapture(model, set(source_layers) | {target_layer})
    results = {}
    for history, horizon in parse_cells(args.cell):
        # Reset both module initialization and sampling for matched probes.
        torch.manual_seed(args.seed + 404)
        train_generator = torch.Generator().manual_seed(args.seed + 505)
        eval_generator = torch.Generator().manual_seed(args.seed + 606)
        probe = FrozenSufficiencyProbe(
            predictor.config.hidden_size if predictor is not None
            else int(getattr(model.config, "text_config", model.config).hidden_size),
            len(source_layers),
        ).to(device=args.device, dtype=dtype)
        optimizer = torch.optim.AdamW(
            probe.parameters(), lr=args.learning_rate, weight_decay=0.1
        )
        for step in range(1, args.steps + 1):
            state, action, target = sample_features(
                model, capture, train_data, source_layers=source_layers,
                target_layer=target_layer, history=history, horizon=horizon,
                batch_size=args.batch_size, generator=train_generator,
                device=args.device,
            )
            prediction = probe(state, action)
            loss = transition_loss(
                prediction, target,
                torch.ones(len(target), dtype=torch.bool, device=target.device),
            ).total
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
        key = f"history_{history}_horizon_{horizon}"
        results[key] = evaluate(
            probe, model, capture, validation_data,
            cell=(history, horizon), source_layers=source_layers,
            target_layer=target_layer, args=args, generator=eval_generator,
        )
        results[key]["probe_parameters"] = sum(
            parameter.numel() for parameter in probe.parameters()
        )
        print(json.dumps({key: results[key]}), flush=True)
    one = results.get("history_1_horizon_1")
    eight = results.get("history_8_horizon_1")
    history_gap = None if one is None or eight is None else (
        one["cosine"] - eight["cosine"]
    )
    report = {
        "schema_version": 1,
        "kind": "fresh_frozen_transition_sufficiency_probes",
        "checkpoint": str(args.checkpoint),
        "checkpoint_variant": payload["variant"],
        "source_layers": list(source_layers),
        "target_layer": target_layer,
        "results": results,
        "history_gap_cosine": history_gap,
        "dataset_metadata": metadata,
        "fresh_probe": True,
        "predictor_reused": False,
        "oracle_information": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    capture.__exit__(None, None, None)


if __name__ == "__main__":
    main()
