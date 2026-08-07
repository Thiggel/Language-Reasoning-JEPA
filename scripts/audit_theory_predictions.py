"""Pre-registered theory tests for the endpoint-ranking Energy head.

Runs the falsifiable predictions T1, T2, T4 from
research/reports/intent_phrase/2026-08-07-theory-propositions/REPORT.md
against a trained checkpoint. All symbolic quantities (remaining necessary
steps, necessary/distractor identity) are oracle measurement labels only;
nothing here feeds training.

    .venv/bin/python scripts/audit_theory_predictions.py \
        ckpt=<run>/model/best.pt out=<dir>/theory_probes.json

T1  Ordinal faithfulness: Kendall tau between EMA-latent goal distance and
    symbolic steps-to-go over the prefix states of random-policy episodes.
T2  Monotone invariance: planner metrics must be bit-identical when the
    Energy head output passes through strictly increasing transforms
    (exp, cube, affine). Any deviation is an implementation bug, not a
    property of the model (Proposition 2 is exact).
T4  Label quality: agreement of the best-of-R latent rollout label with the
    exact H-step optimal root ordering (necessary before distractor,
    Proposition 3), as a function of R and H, restricted to anchors where
    the exact ordering is strict (>= H remaining necessary steps).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import prompt_sentences
from textjepa.planning import LatentPlanner, evaluate_planning
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, load_run


def _kendall_tau(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = (xs[i] - xs[j]) * (ys[i] - ys[j])
            if a > 0:
                concordant += 1
            elif a < 0:
                discordant += 1
    total = concordant + discordant  # ties contribute nothing
    return (concordant - discordant) / total if total else None


class _Probe:
    """Teacher-encoding helpers shared by the tests (training conventions)."""

    def __init__(self, model, vocab, device):
        self.model, self.vocab, self.device = model, vocab, device

    def _tokens(self, texts: list[str]) -> torch.Tensor:
        ids = [self.vocab.encode(t) for t in texts]
        length = max(len(t) for t in ids)
        out = torch.full((1, len(ids), length), self.vocab.pad_id, dtype=torch.long)
        for c, t in enumerate(ids):
            out[0, c, : len(t)] = torch.tensor(t)
        return out.to(self.device)

    @torch.no_grad()
    def encode_prefix(self, prompt_texts, step_texts) -> torch.Tensor:
        """EMA-teacher latent of the state after ``step_texts`` (s0 if empty)."""
        prompt_tokens = self._tokens(prompt_texts)
        prompt_mask = torch.ones(
            1, len(prompt_texts), dtype=torch.bool, device=self.device
        )
        if not step_texts:
            empty = torch.full(
                (1, 1, 1), self.vocab.pad_id, dtype=torch.long, device=self.device
            )
            no_steps = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
            s0, _ = self.model.encode_states(
                prompt_tokens, prompt_mask, empty, no_steps, teacher=True
            )
            return s0
        step_tokens = self._tokens(step_texts)
        step_mask = torch.ones(
            1, len(step_texts), dtype=torch.bool, device=self.device
        )
        _, states = self.model.encode_states(
            prompt_tokens, prompt_mask, step_tokens, step_mask, teacher=True
        )
        return states[:, -1]

    @staticmethod
    def goal_distance(z: torch.Tensor, goal: torch.Tensor) -> float:
        ln = lambda x: torch.nn.functional.layer_norm(x, x.shape[-1:])
        return float((ln(z) - ln(goal)).abs().mean(-1).item())

    def solved_texts(self, problem) -> list[str]:
        env = SymbolicEnv(problem)
        texts = []
        while not env.solved:
            necessary = [
                a for a in env.feasible_actions() if a in problem.query_ancestors
            ]
            texts.append(env.step(min(necessary)))
        return texts


def t1_ordinal_faithfulness(probe, dataset, n_episodes, seed) -> dict:
    rng = random.Random(seed)
    taus = []
    for i in range(n_episodes):
        problem, _ = dataset.problem(i)
        prompt = prompt_sentences(problem, random.Random(seed + i))
        goal = probe.encode_prefix(prompt, probe.solved_texts(problem))
        env = SymbolicEnv(problem)
        texts: list[str] = []
        dists = [probe.goal_distance(probe.encode_prefix(prompt, texts), goal)]
        steps_to_go = [env.remaining_necessary()]
        while not env.solved:
            texts.append(env.step(rng.choice(env.feasible_actions())))
            dists.append(
                probe.goal_distance(probe.encode_prefix(prompt, texts), goal)
            )
            steps_to_go.append(env.remaining_necessary())
        tau = _kendall_tau(dists, [float(s) for s in steps_to_go])
        if tau is not None:
            taus.append(tau)
    return {
        "mean_kendall_tau": sum(taus) / len(taus),
        "min_kendall_tau": min(taus),
        "frac_tau_above_0_8": sum(t > 0.8 for t in taus) / len(taus),
        "n_episodes": len(taus),
        "protocol": "random-feasible episodes; prefix states vs symbolic "
        "steps-to-go (oracle measurement label)",
    }


def t2_monotone_invariance(
    model, vocab, device, dataset, cfg, n_episodes
) -> dict:
    """Prop 2(i) wiring check: g∘E must leave planner metrics unchanged."""

    def run(transform):
        head = model.core.horizon_energy_head
        original = head.forward
        if transform == "identity":
            wrapped = original
        elif transform == "exp":
            wrapped = lambda *a, **k: torch.exp(original(*a, **k) / 4.0)
        elif transform == "cube":
            wrapped = lambda *a, **k: original(*a, **k) ** 3
        elif transform == "affine":
            wrapped = lambda *a, **k: 7.0 * original(*a, **k) - 3.0
        head.forward = wrapped
        try:
            planner = LatentPlanner(
                model, vocab, device, lookahead=cfg.lookahead,
                max_expand=cfg.max_expand, energy="value",
                search_algorithm="root_balanced_beam",
                candidate_interface="feasible_menu",
                allow_oracle_future_actions=cfg.lookahead > 1,
            )
            return evaluate_planning(
                planner, dataset, n_episodes, slack=2, seed=cfg.seed
            )
        finally:
            head.forward = original

    baseline = run("identity")
    key = next(iter(baseline))
    results = {"identity": baseline[key]}
    identical = True
    for g in ("exp", "cube", "affine"):
        metrics = run(g)[key]
        results[g] = metrics
        identical = identical and all(
            metrics[m] == baseline[key][m]
            for m in ("success", "mean_steps", "distractor_rate")
        )
    return {"planner_metrics": results, "all_identical": identical}


def t4_label_quality(probe, dataset, n_anchors, seed, horizons, rs) -> dict:
    """Best-of-R rollout label vs the exact H-step optimal root ordering.

    Restricted to anchors with >= H remaining necessary steps, where the
    exact ordering is strict: any necessary root strictly beats any
    distractor root (Prop 3(ii)). Agreement = the label's argmin root is a
    necessary action.
    """
    rng = random.Random(seed)
    max_r = max(rs)
    agree = {h: {r: [0, 0] for r in rs} for h in horizons}
    i = 0
    collected = 0
    while collected < n_anchors and i < 10 * n_anchors:
        problem, _ = dataset.problem(i)
        i += 1
        prompt = prompt_sentences(problem, random.Random(seed + i))
        goal = probe.encode_prefix(prompt, probe.solved_texts(problem))
        # anchor: a random prefix of a random-feasible trajectory
        env = SymbolicEnv(problem)
        texts: list[str] = []
        depth = rng.randrange(0, max(1, problem.n_necessary_steps))
        ok = True
        for _ in range(depth):
            if env.solved:
                ok = False
                break
            texts.append(env.step(rng.choice(env.feasible_actions())))
        if not ok or env.solved:
            continue
        candidates = env.feasible_actions()
        necessary = set(problem.query_ancestors)
        if not any(c in necessary for c in candidates) or all(
            c in necessary for c in candidates
        ):
            continue  # ordering not strict without both kinds
        collected += 1
        for h in horizons:
            if env.remaining_necessary() < h:
                continue
            # per candidate: max_r true rollouts; endpoint teacher distances
            dist = {}
            for c in candidates:
                samples = []
                for _ in range(max_r):
                    roll = env.clone()
                    rtexts = list(texts)
                    rtexts.append(roll.step(c))
                    for _ in range(h - 1):
                        if roll.solved:
                            break
                        feas = roll.feasible_actions()
                        if not feas:
                            break
                        rtexts.append(roll.step(rng.choice(feas)))
                    z = probe.encode_prefix(prompt, rtexts)
                    samples.append(probe.goal_distance(z, goal))
                dist[c] = samples
            for r in rs:
                best = min(candidates, key=lambda c: min(dist[c][:r]))
                agree[h][r][0] += int(best in necessary)
                agree[h][r][1] += 1
    return {
        "agreement": {
            str(h): {
                str(r): (a / n if n else None)
                for r, (a, n) in by_r.items()
            }
            for h, by_r in agree.items()
        },
        "n_anchors": collected,
        "protocol": "mixed-candidate anchors with >= H necessary steps; "
        "exact optimal ordering from Prop 3(ii) (oracle measurement label)",
    }


@hydra.main(config_path="../configs", config_name="plan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    model, vocab, run_cfg = load_run(cfg.ckpt, cfg.device)
    dataset = build_dataset(run_cfg, vocab, split=cfg.get("split", "val"))
    device = torch.device(cfg.device)
    probe = _Probe(model, vocab, device)

    n = int(cfg.get("n_episodes", 100))
    results = {
        "checkpoint": str(cfg.ckpt),
        "t1_ordinal_faithfulness": t1_ordinal_faithfulness(
            probe, dataset, n, cfg.seed
        ),
        "t2_monotone_invariance": t2_monotone_invariance(
            model, vocab, device, dataset, cfg, n_episodes=min(n, 60)
        ),
        "t4_label_quality": t4_label_quality(
            probe, dataset, n_anchors=n, seed=cfg.seed,
            horizons=(1, 2, 4, 8), rs=(1, 2, 4, 8),
        ),
    }
    out = Path(cfg.out or Path(cfg.ckpt).parent / "theory_probes.json")
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
