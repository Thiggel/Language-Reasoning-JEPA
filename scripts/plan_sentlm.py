"""Evaluation for the sentence-latent LM baselines.

Selection energies:
- ``score=decoder``: CE of each candidate sentence decoded from
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
from textjepa.data.igsm.render import action_phrase, prompt_sentences, step_sentence
from textjepa.models.sent_lm import SentenceLM
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset


@hydra.main(config_path="../configs", config_name="plan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    ckpt = torch.load(cfg.ckpt, map_location=cfg.device, weights_only=False)
    run_cfg = OmegaConf.create(ckpt["cfg"])
    device = torch.device(cfg.device)
    faithful = run_cfg.data.get("name", "igsm") == "igsm_real"
    if faithful:
        from textjepa.data.faithful import cached_faithful_vocab

        vocab = cached_faithful_vocab()
    else:
        vocab = build_vocab(run_cfg.data.modulus)
    model = SentenceLM(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **run_cfg.model
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    split = cfg.get("split", "val")
    dataset = build_dataset(run_cfg, vocab, split=split)
    score = cfg.get("score", "decoder")
    target_kind = run_cfg.train.get("target_kind", "outcome")
    if target_kind not in {"outcome", "intent"}:
        raise ValueError(f"unknown sentence LM target_kind: {target_kind}")

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
            if faithful:
                from textjepa.data.faithful import FaithfulEnv

                env = FaithfulEnv(problem)
                prompt = problem.prompt_sentences
                necessary = problem.necessary
            else:
                env = SymbolicEnv(problem)
                prompt = prompt_sentences(problem, random.Random(cfg.seed + ep))
                necessary = problem.query_ancestors
            history_texts: list[str] = []
            budget = len(necessary) + cfg.slack
            n = 0
            while not env.solved and n < budget:
                batch = {
                    "prompt_tokens": tokens(prompt),
                    "prompt_mask": torch.ones(1, len(prompt), dtype=torch.bool,
                                              device=device),
                    "step_tokens": tokens(history_texts or ["."]),
                    "step_mask": torch.tensor(
                        [[bool(history_texts)] * max(len(history_texts), 1)],
                        device=device,
                    ),
                }
                prompt_emb = model.encode_chunks(batch["prompt_tokens"])
                step_emb = model.encode_chunks(batch["step_tokens"])
                s0, states = model.state_model(
                    prompt_emb, batch["prompt_mask"], step_emb,
                    batch["step_mask"],
                )
                ctx = states[:, len(history_texts) - 1] if history_texts else s0
                feas = env.feasible_actions()
                if target_kind == "intent":
                    cand_texts = [
                        env.action_text(a) if faithful
                        else action_phrase(problem, a)
                        for a in feas
                    ]
                else:
                    cand_texts = [
                        env.clone().step(a) if faithful
                        else step_sentence(problem, a)
                        for a in feas
                    ]
                cand_tok = tokens(cand_texts).squeeze(0)
                k = len(feas)
                if score == "latent":
                    pred = model.latent_head(ctx)
                    emb = model.chunk_encoder(cand_tok)
                    ln = lambda x: F.layer_norm(x, x.shape[-1:])
                    s = (ln(pred) - ln(emb)).abs().mean(-1)
                else:
                    s = model.decode_ce(ctx.expand(k, -1), cand_tok)
                    if cfg.get("length_normalize", True):
                        lengths = (cand_tok != vocab.pad_id).sum(-1).clamp_min(1)
                        s = s / lengths
                pick = feas[int(s.argmin().item())]
                distr += int(pick not in necessary)
                if target_kind == "intent":
                    history_texts.append(
                        env.action_text(pick) if faithful
                        else action_phrase(problem, pick)
                    )
                history_texts.append(env.step(pick))
                n += 1
            solved += int(env.solved)
            steps_sum += n
    out = {
        f"sentlm_{target_kind}_{score}": {
            "success": solved / cfg.n_episodes,
            "mean_steps": steps_sum / cfg.n_episodes,
            "distractor_rate": distr / max(steps_sum, 1),
            "length_normalized": bool(
                cfg.get("length_normalize", True) and score == "decoder"
            ),
        }
    }
    print(json.dumps(out, indent=2))
    split_suffix = "" if split == "val" else f"_{split}"
    dest = Path(
        cfg.out or Path(cfg.ckpt).parent
        / f"plan_slack{cfg.slack}_sentlm_{target_kind}_{score}{split_suffix}.json"
    )
    dest.write_text(json.dumps(out, indent=2))
    print(f"saved to {dest}")


if __name__ == "__main__":
    main()
