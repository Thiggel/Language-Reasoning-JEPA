#!/usr/bin/env python3
"""Score three systems on the same lm-eval tasks.

  original  the unmodified pretrained checkpoint
  full      the adapted checkpoint decoded normally, all layers
  jump      the adapted checkpoint with the lower stack skipped, optionally
            re-materializing the block every `refresh` tokens

The jump condition advances the predicted residual through the upper stack for
the scored continuation, so a task's accuracy measures how much capability
survives skipping the lower half. `original` and `full` differ only by Stage 1
training, which is the untested half of the Stage 1 gate; `full` and `jump`
differ only by the decoding path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from textjepa.models.action_transition import ResidualCapture
from textjepa.training.predictive_state import (
    UpperStackRunner,
    crop_cache,
    load_stage1_checkpoint,
    load_stage2_checkpoint,
    require_transformers_runtime,
    teacher_forward,
    transition_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default="lambada_openai,piqa,arc_easy,winogrande")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--refresh", type=int, action="append")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"),
                        default="bfloat16")
    return parser.parse_args()


def build_jump_model(base_class):
    class JumpLM(base_class):
        """An lm-eval model whose scoring runs through the predicted state."""

        def __init__(self, *args, predictor=None, capture=None,
                     refresh=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._predictor = predictor
            self._capture = capture
            self._refresh = refresh
            self._runner = UpperStackRunner(
                self.model, target_layer=predictor.config.target_layer,
                source_layers=predictor.config.source_layers,
            )

        def _model_call(self, inps, attn_mask=None, labels=None):
            """Return logits for every position, produced by the jump path.

            Position 0 is prefilled exactly; each later position is reached by
            predicting the target residual from the previous position's state
            and the realized token, then running only the upper stack.
            """
            del attn_mask, labels
            outputs = []
            for row in range(inps.shape[0]):
                tokens = inps[row:row + 1]
                exact = teacher_forward(
                    self.model, tokens[:, :1], capture=self._capture,
                    attention_mask=None, use_cache=True,
                )
                cache = exact.past_key_values
                sources = {
                    layer: exact.states[layer][:, -1:]
                    for layer in self._predictor.config.used_source_layers
                }
                logits = [exact.logits[:, -1:]]
                for index in range(1, tokens.shape[1]):
                    action = tokens[:, index:index + 1]
                    if (
                        self._refresh is not None
                        and index % self._refresh == 0
                    ):
                        # Re-materialize the whole block just decoded, not only
                        # the current token: the upper cache entries for the
                        # jumped block were built from predicted states and are
                        # what a refresh exists to replace.
                        start = max(0, index - self._refresh + 1)
                        crop_cache(cache, start)
                        positions = torch.arange(
                            start, index + 1, device=tokens.device
                        )[None]
                        exact = teacher_forward(
                            self.model, tokens[:, start:index + 1],
                            capture=self._capture,
                            attention_mask=None, position_ids=positions,
                            use_cache=True, past_key_values=cache,
                        )
                        cache = exact.past_key_values
                        sources = {
                            layer: exact.states[layer][:, -1:]
                            for layer in self._predictor.config.used_source_layers
                        }
                        logits.append(exact.logits[:, -1:])
                        continue
                    predicted = transition_prediction(
                        self._predictor, self.model, sources, action
                    )
                    sources, step_logits = self._runner.step(
                        predicted, past_key_values=cache, position_index=index,
                    )
                    logits.append(step_logits[:, -1:])
                outputs.append(torch.cat(logits, dim=1))
            return F.log_softmax(torch.cat(outputs, dim=0).float(), dim=-1)

    return JumpLM


def main() -> None:
    args = parse_args()
    require_transformers_runtime()
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    dtype = getattr(torch, args.dtype)
    header = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    loader = (
        load_stage2_checkpoint
        if header.get("kind") == "action_conditioned_cross_layer_stage2"
        else load_stage1_checkpoint
    )
    model, predictor, payload = loader(
        args.checkpoint, device=args.device, dtype=dtype
    )
    model.eval()
    stage1 = payload.get("stage1_payload", payload)
    model_id, revision = stage1["model_id"], stage1["model_revision"]
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]

    report = {
        "schema_version": 1,
        "kind": "lm_harness_three_way",
        "checkpoint": str(args.checkpoint),
        "model_id": model_id,
        "model_revision": revision,
        "tasks": tasks,
        "limit": args.limit,
        "results": {},
    }

    def record(name, results):
        report["results"][name] = {
            task: {
                metric: value for metric, value in values.items()
                if isinstance(value, (int, float))
            }
            for task, values in results["results"].items()
        }
        print(json.dumps({name: report["results"][name]}), flush=True)

    # 1. The unmodified pretrained checkpoint.
    record("original", lm_eval.simple_evaluate(
        model=HFLM(pretrained=model_id, revision=revision, dtype=dtype,
                   batch_size=args.batch_size, device=args.device),
        tasks=tasks, limit=args.limit,
    ))

    # 2. The adapted checkpoint, decoded normally.
    record("adapted_full", lm_eval.simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=model_id,
                   batch_size=args.batch_size, device=args.device),
        tasks=tasks, limit=args.limit,
    ))

    # 3. The adapted checkpoint with the lower stack skipped.
    JumpLM = build_jump_model(HFLM)
    layers = set(predictor.config.used_source_layers) | {
        predictor.config.target_layer
    }
    for refresh in [None] + sorted(set(args.refresh or [4, 16])):
        label = "adapted_jump" if refresh is None else f"adapted_jump_refresh_{refresh}"
        with ResidualCapture(model, layers) as capture:
            record(label, lm_eval.simple_evaluate(
                model=JumpLM(pretrained=model, tokenizer=model_id,
                             batch_size=1, device=args.device,
                             predictor=predictor, capture=capture,
                             refresh=refresh),
                tasks=tasks, limit=args.limit,
            ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
