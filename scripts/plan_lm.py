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
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import action_phrase, prompt_sentences, step_sentence
from textjepa.models.lm_baseline import DecoderLM
from textjepa.planning.search import EpisodeResult
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset


@hydra.main(config_path="../configs", config_name="plan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    ckpt = torch.load(cfg.ckpt, map_location=cfg.device, weights_only=False)
    run_cfg = OmegaConf.create(ckpt["cfg"])
    score_kind = cfg.get("score_kind") or run_cfg.train.get(
        "target_kind", "outcome"
    )
    if score_kind not in {"outcome", "intent"}:
        raise ValueError(f"unknown LM score_kind: {score_kind}")
    device = torch.device(cfg.device)
    faithful = run_cfg.data.get("name", "igsm") == "igsm_real"
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
    with torch.no_grad():
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
            budget = n_necessary + cfg.slack
            steps = n_distr = 0
            while not env.solved and steps < budget:
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
                history += vocab.encode(env.step(pick))
                steps += 1
            results.append(
                EpisodeResult(env.solved, steps, n_necessary, n_distr)
            )
    n = len(results)
    out = {
        f"lm_{score_kind}_policy": {
            "success": sum(r.solved for r in results) / n,
            "mean_steps": sum(r.steps for r in results) / n,
            "mean_necessary": sum(r.n_necessary for r in results) / n,
            "distractor_rate": sum(r.n_distractor for r in results)
            / max(sum(r.steps for r in results), 1),
            "length_normalized": bool(cfg.get("length_normalize", True)),
        }
    }
    for k, v in out[f"lm_{score_kind}_policy"].items():
        print(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
    split_suffix = "" if split == "val" else f"_{split}"
    dest = Path(
        cfg.out or Path(cfg.ckpt).parent
        / f"plan_slack{cfg.slack}_lm_{score_kind}{split_suffix}.json"
    )
    dest.write_text(json.dumps(out, indent=2))
    print(f"saved to {dest}")


if __name__ == "__main__":
    main()
