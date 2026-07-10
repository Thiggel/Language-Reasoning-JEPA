"""Parity evaluation for the sentence-latent LM baseline.

Selection energies:
- ``score=decoder``: CE of each candidate's step sentence decoded from
  the context latent (reconstruction likelihood — both variants).
- ``score=latent``: LN-L1 distance between the predicted next latent and
  each candidate's encoded latent (only meaningful for latent_target
  models).

    python scripts/plan_sentlm.py ckpt=runs/sent_lm/best.pt slack=0
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import prompt_sentences, step_sentence
from textjepa.models.sent_lm import SentenceLM
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset


@hydra.main(config_path="../configs", config_name="plan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    ckpt = torch.load(cfg.ckpt, map_location=cfg.device, weights_only=False)
    run_cfg = OmegaConf.create(ckpt["cfg"])
    device = torch.device(cfg.device)
    vocab = build_vocab(run_cfg.data.modulus)
    model = SentenceLM(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **run_cfg.model
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dataset = build_dataset(run_cfg, vocab, split="val")
    score = cfg.get("score", "decoder")

    def tokens(texts):
        ids = [vocab.encode(t) for t in texts]
        L = max(len(i) for i in ids)
        out = torch.full((1, len(ids), L), vocab.pad_id, dtype=torch.long)
        for c, i in enumerate(ids):
            out[0, c, : len(i)] = torch.tensor(i)
        return out.to(device)

    solved = steps_sum = distr = 0
    with torch.no_grad():
        for ep in range(cfg.n_episodes):
            problem, _ = dataset.problem(ep)
            env = SymbolicEnv(problem)
            prompt = prompt_sentences(problem, random.Random(cfg.seed + ep))
            step_texts: list[str] = []
            budget = problem.n_necessary_steps + cfg.slack
            n = 0
            while not env.solved and n < budget:
                batch = {
                    "prompt_tokens": tokens(prompt),
                    "prompt_mask": torch.ones(1, len(prompt), dtype=torch.bool,
                                              device=device),
                    "step_tokens": tokens(step_texts or ["."]),
                    "step_mask": torch.tensor(
                        [[bool(step_texts)] * max(len(step_texts), 1)],
                        device=device,
                    ),
                }
                ctx_all = model.contexts(batch)
                ctx = ctx_all[:, len(step_texts) if step_texts else 0]
                feas = env.feasible_actions()
                cand_tok = tokens(
                    [step_sentence(problem, a) for a in feas]
                ).squeeze(0)
                k = len(feas)
                if score == "latent":
                    pred = model.latent_head(ctx)
                    emb = model.chunk_encoder(cand_tok)
                    ln = lambda x: F.layer_norm(x, x.shape[-1:])
                    s = (ln(pred) - ln(emb)).abs().mean(-1)
                else:
                    s = model.decode_ce(ctx.expand(k, -1), cand_tok)
                pick = feas[int(s.argmin().item())]
                distr += int(pick not in problem.query_ancestors)
                step_texts.append(env.step(pick))
                n += 1
            solved += int(env.solved)
            steps_sum += n
    out = {
        f"sentlm_{score}": {
            "success": solved / cfg.n_episodes,
            "mean_steps": steps_sum / cfg.n_episodes,
            "distractor_rate": distr / max(steps_sum, 1),
        }
    }
    print(json.dumps(out, indent=2))
    dest = Path(
        cfg.out
        or Path(cfg.ckpt).parent / f"plan_slack{cfg.slack}_sentlm_{score}.json"
    )
    dest.write_text(json.dumps(out, indent=2))
    print(f"saved to {dest}")


if __name__ == "__main__":
    main()
