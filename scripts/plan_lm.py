"""Evaluate the LM baseline with action-selection parity.

At each state: score every feasible action's rendered step SENTENCE by
LM log-likelihood given [prompt + steps so far]; execute the argmax in
the environment; same budget/success criterion as the latent planner.

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
from textjepa.data.igsm.render import prompt_sentences, step_sentence
from textjepa.models.lm_baseline import DecoderLM
from textjepa.planning.search import EpisodeResult
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset


@hydra.main(config_path="../configs", config_name="plan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    ckpt = torch.load(cfg.ckpt, map_location=cfg.device, weights_only=False)
    run_cfg = OmegaConf.create(ckpt["cfg"])
    device = torch.device(cfg.device)
    vocab = build_vocab(run_cfg.data.modulus)
    model = DecoderLM(
        vocab_size=len(vocab), pad_id=vocab.pad_id,
        d_model=run_cfg.model.d_model, n_layers=run_cfg.model.n_layers,
        n_heads=run_cfg.model.n_heads, ff_mult=run_cfg.model.ff_mult,
        max_len=run_cfg.model.max_len,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dataset = build_dataset(run_cfg, vocab, split="val")

    results = []
    with torch.no_grad():
        for ep in range(cfg.n_episodes):
            problem, _ = dataset.problem(ep)
            env = SymbolicEnv(problem)
            prompt = prompt_sentences(problem, random.Random(cfg.seed + ep))
            history = [t for s in prompt for t in vocab.encode(s)]
            budget = problem.n_necessary_steps + cfg.slack
            steps = n_distr = 0
            while not env.solved and steps < budget:
                feas = env.feasible_actions()
                cands = [vocab.encode(step_sentence(problem, a)) for a in feas]
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
                pick = feas[int(lp.argmax().item())]
                n_distr += int(pick not in problem.query_ancestors)
                history += vocab.encode(env.step(pick))
                steps += 1
            results.append(
                EpisodeResult(env.solved, steps, problem.n_necessary_steps, n_distr)
            )
    n = len(results)
    out = {
        "lm_policy": {
            "success": sum(r.solved for r in results) / n,
            "mean_steps": sum(r.steps for r in results) / n,
            "mean_necessary": sum(r.n_necessary for r in results) / n,
            "distractor_rate": sum(r.n_distractor for r in results)
            / max(sum(r.steps for r in results), 1),
        }
    }
    for k, v in out["lm_policy"].items():
        print(f"{k}={v:.3f}")
    dest = Path(cfg.out or Path(cfg.ckpt).parent / f"plan_slack{cfg.slack}_lm.json")
    dest.write_text(json.dumps(out, indent=2))
    print(f"saved to {dest}")


if __name__ == "__main__":
    main()
