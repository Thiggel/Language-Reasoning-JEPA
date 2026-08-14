"""Evaluate outcome-scoring and intent-policy LM baselines.

The historical outcome LM scores rendered candidate step sentences, which
include their computed consequences.  The information-matched intent policy
instead scores the same outcome-free action phrases as the JEPA planner.  It
then appends the selected intent and observed outcome to its causal history.

    python scripts/plan_lm.py ckpt=runs/lm_9m/best.pt slack=0
"""

from __future__ import annotations

import json
import random
from contextlib import nullcontext
from pathlib import Path

import hydra
import math
import torch
from omegaconf import DictConfig, OmegaConf

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import action_phrase, prompt_sentences, step_sentence
from textjepa.models.lm_baseline import DecoderLM
from textjepa.planning.evaluate import aggregate_episodes
from textjepa.planning.search import EpisodeResult
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import apply_eval_data_overrides, build_dataset


@hydra.main(config_path="../configs", config_name="plan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    ckpt = torch.load(cfg.ckpt, map_location=cfg.device, weights_only=False)
    run_cfg = OmegaConf.create(ckpt["cfg"])
    apply_eval_data_overrides(run_cfg, cfg)
    score_kind = cfg.get("score_kind") or run_cfg.train.get(
        "target_kind", "outcome"
    )
    if score_kind not in {"outcome", "intent"}:
        raise ValueError(f"unknown LM score_kind: {score_kind}")
    candidate_interface = cfg.get("candidate_interface", "feasible_menu")
    if candidate_interface not in {"feasible_menu", "full_catalogue"}:
        raise ValueError(f"unknown candidate_interface: {candidate_interface}")
    device = torch.device(cfg.device)
    faithful = run_cfg.data.get("name", "igsm") == "igsm_real"
    if candidate_interface == "full_catalogue" and score_kind != "intent":
        raise ValueError(
            "full_catalogue candidate_interface requires score_kind=intent "
            "(outcome-scoring needs a feasible action to render the outcome)"
        )
    if faithful:
        from textjepa.data.faithful import cached_faithful_vocab

        vocab = cached_faithful_vocab()
    else:
        vocab = build_vocab(run_cfg.data.modulus)
    model = DecoderLM(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **run_cfg.model
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    if cfg.get("eval_loops") is not None:
        if not hasattr(model.blocks, "eval_loops"):
            raise ValueError("eval_loops requires a recurrent LM checkpoint")
        model.blocks.eval_loops = int(cfg.eval_loops)
    split = cfg.get("split", "val")
    dataset = build_dataset(run_cfg, vocab, split=split)

    results = []
    measure_flops = bool(cfg.get("measure_flops", False))
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except (ImportError, AttributeError):
        FlopCounterMode = None
    flop_counter = (
        FlopCounterMode(display=False)
        if measure_flops and FlopCounterMode is not None else nullcontext()
    )
    with torch.no_grad(), flop_counter:
        for ep in range(cfg.n_episodes):
            problem, _ = dataset.problem(ep)
            if faithful:
                from textjepa.data.faithful import FaithfulEnv

                env = FaithfulEnv(problem)
                prompt = problem.prompt_sentences
                necessary = problem.necessary
            else:
                env = SymbolicEnv(problem)
                prompt = prompt_sentences(problem, random.Random(cfg.seed + ep))
                necessary = problem.query_ancestors
            history = [t for s in prompt for t in vocab.encode(s)]
            n_necessary = len(necessary)
            # Proportional slack: same budget rule as FaithfulPlanner
            # (budget = necessary + slack + ceil(slack_frac * necessary)).
            # slack_frac defaults to 0, leaving historical runs unchanged.
            budget = n_necessary + cfg.slack + math.ceil(
                float(cfg.get("slack_frac", 0.0)) * n_necessary
            )
            steps = n_distr = n_invalid = 0
            # Faithful full_catalogue only: the policy's OWN attempted-invalid
            # actions, masked until progress (same state-scoped semantics as
            # FaithfulPlanner; without it a deterministic argmax loops on a
            # no-op forever).
            mask_attempted = bool(cfg.get("mask_attempted", True))
            attempted: set = set()
            while not env.solved and steps < budget:
                if candidate_interface == "full_catalogue":
                    if faithful:
                        from textjepa.planning.faithful_search import (
                            faithful_catalogue,
                        )

                        feas = faithful_catalogue(
                            env,
                            frozenset(attempted) if mask_attempted
                            else frozenset(),
                        )
                        if not feas:
                            # Mask exhausted the catalogue: stall (unsolved)
                            # rather than re-propose a known-dead action.
                            break
                    else:
                        feas = list(range(len(problem.vars)))
                else:
                    feas = env.feasible_actions()
                if score_kind == "intent":
                    cands = [
                        vocab.encode(env.action_text(a)) if faithful
                        else vocab.encode(action_phrase(problem, a))
                        for a in feas
                    ]
                else:
                    cands = [
                        vocab.encode(env.clone().step(a)) if faithful
                        else vocab.encode(step_sentence(problem, a))
                        for a in feas
                    ]
                L = len(history) + max(len(c) for c in cands)
                toks = torch.full(
                    (len(cands), L), vocab.pad_id, dtype=torch.long
                )
                for i, c in enumerate(cands):
                    seq = history + c
                    toks[i, : len(seq)] = torch.tensor(seq)
                lp = model.sequence_logprob(
                    toks.to(device),
                    torch.full((len(cands),), len(history), device=device),
                )
                if cfg.get("length_normalize", True):
                    lengths = torch.tensor(
                        [len(c) for c in cands], device=device, dtype=lp.dtype
                    ).clamp_min(1)
                    lp = lp / lengths
                pick = feas[int(lp.argmax().item())]
                n_distr += int(pick not in necessary)
                if score_kind == "intent":
                    history += (
                        vocab.encode(env.action_text(pick)) if faithful
                        else vocab.encode(action_phrase(problem, pick))
                    )
                if candidate_interface == "full_catalogue":
                    invalid = pick not in env.feasible_actions()
                    n_invalid += int(invalid)
                    history += vocab.encode(env.step_or_invalid(pick))
                    if faithful:
                        # Mask only WHILE the state is unchanged; reset to the
                        # resolved set on progress (an action infeasible now
                        # may become feasible after its dependencies resolve).
                        if invalid:
                            attempted.add(pick)
                        else:
                            attempted = set(env.resolved)
                else:
                    history += vocab.encode(env.step(pick))
                steps += 1
            results.append(
                EpisodeResult(
                    env.solved, steps, n_necessary, n_distr, n_invalid,
                    # The greedy policy stops the moment the goal is reached,
                    # so the executed-step count IS the solved-at step.
                    solved_at=steps if env.solved else None,
                )
            )
    n = len(results)
    slack_curve = bool(cfg.get("slack_curve", False))
    metrics = aggregate_episodes(
        results, slack_curve=slack_curve, slack=cfg.slack
    )
    # ``invalid_rate`` is the historical field name in the LM baseline JSONs;
    # keep it as an alias of the shared ``invalid_action_rate``.
    metrics["invalid_rate"] = metrics["invalid_action_rate"]
    metrics.update({
        "length_normalized": bool(cfg.get("length_normalize", True)),
        "candidate_interface": candidate_interface,
    })
    out = {f"lm_{score_kind}_policy": metrics}
    out[f"lm_{score_kind}_policy"]["flop_measurement_supported"] = (
        FlopCounterMode is not None
    )
    if measure_flops and FlopCounterMode is not None:
        total_flops = int(flop_counter.get_total_flops())
        out[f"lm_{score_kind}_policy"].update({
            "measured_eval_flops": total_flops,
            "measured_flops_per_episode": total_flops / n,
        })
    for k, v in out[f"lm_{score_kind}_policy"].items():
        print(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
    split_suffix = "" if split == "val" else f"_{split}"
    ci_suffix = (
        "" if candidate_interface == "feasible_menu" else f"_{candidate_interface}"
    )
    # A slack curve is scored from one generous-budget run, so it must not
    # overwrite the fixed-slack run at the same nominal slack.
    curve_suffix = "_slackcurve" if slack_curve else ""
    dest = Path(
        cfg.out or Path(cfg.ckpt).parent
        / f"plan_slack{cfg.slack}_lm_{score_kind}{ci_suffix}"
        f"{curve_suffix}{split_suffix}.json"
    )
    dest.write_text(json.dumps(out, indent=2))
    print(f"saved to {dest}")


if __name__ == "__main__":
    main()
