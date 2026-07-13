"""Sample-based planning with variational (unobserved) actions.

At each state: sample M codes from the prior p(a|s), roll F, score
cost = 1 + V; decode the best code via the detached readout to an
intent-anchor embedding; execute the nearest feasible symbolic action
(matching in the frozen anchor space). No intent phrases enter the
model as inputs.

    python scripts/plan_var.py ckpt=runs/disc_var/best.pt slack=0
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import action_phrase, prompt_sentences
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, load_run


def _ln(x):
    return F.layer_norm(x, x.shape[-1:])


@hydra.main(config_path="../configs", config_name="plan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    model, vocab, run_cfg = load_run(cfg.ckpt, cfg.device)
    dataset = build_dataset(run_cfg, vocab, split="val")
    device = torch.device(cfg.device)
    M = cfg.get("n_samples_var", 64)

    def tok(texts):
        ids = [vocab.encode(t) for t in texts]
        L = max(len(i) for i in ids)
        out = torch.full((1, len(ids), L), vocab.pad_id, dtype=torch.long)
        for c, i in enumerate(ids):
            out[0, c, : len(i)] = torch.tensor(i)
        return out.to(device)

    def enc_state(pt, pm, texts):
        if not texts:
            empty = torch.full((1, 1, 1), vocab.pad_id, dtype=torch.long,
                               device=device)
            return model.encode_states(
                pt, pm, empty,
                torch.zeros(1, 1, dtype=torch.bool, device=device))[0]
        st = tok(texts)
        sm = torch.ones(1, st.shape[1], dtype=torch.bool, device=device)
        return model.encode_states(pt, pm, st, sm)[1][:, -1]

    solved = steps_sum = distr = 0
    with torch.no_grad():
        for ep in range(cfg.n_episodes):
            problem, _ = dataset.problem(ep)
            env = SymbolicEnv(problem)
            prompt = prompt_sentences(problem, random.Random(cfg.seed + ep))
            pt = tok(prompt)
            pm = torch.ones(1, pt.shape[1], dtype=torch.bool, device=device)
            texts: list[str] = []
            budget = problem.n_necessary_steps + cfg.slack
            s0 = enc_state(pt, pm, [])
            n = 0
            while not env.solved and n < budget:
                s = enc_state(pt, pm, texts) if texts else s0
                codes = model.var_action.sample_prior(s, k=M).squeeze(0)
                preds = model.predictor(s.expand(M, -1), codes)
                cost = 1.0 + model.value_head(preds, s0.expand(M, -1))
                a_star = codes[int(cost.argmin().item())]
                dec = model.act_decode(a_star.unsqueeze(0))
                feas = env.feasible_actions()
                itoks = tok([action_phrase(problem, a) for a in feas]).squeeze(0)
                anchors = model.chunk_anchor(itoks)
                d = (_ln(dec) - _ln(anchors)).abs().mean(-1)
                pick = feas[int(d.argmin().item())]
                distr += int(pick not in problem.query_ancestors)
                texts.append(env.step(pick))
                n += 1
            solved += int(env.solved)
            steps_sum += n
    out = {"var_planner": {
        "success": solved / cfg.n_episodes,
        "mean_steps": steps_sum / cfg.n_episodes,
        "distractor_rate": distr / max(steps_sum, 1),
    }}
    print(json.dumps(out, indent=2))
    dest = Path(cfg.out or Path(cfg.ckpt).parent / f"plan_slack{cfg.slack}_var.json")
    dest.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
