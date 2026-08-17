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
from contextlib import nullcontext
from pathlib import Path

import hydra
import math
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import action_phrase, prompt_sentences, step_sentence
from textjepa.models.sent_lm import SentenceLM
from textjepa.planning.evaluate import aggregate_episodes
from textjepa.planning.search import EpisodeResult
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import apply_eval_data_overrides, build_dataset


def _percentile(sorted_vals: list, q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, int(math.ceil(q * len(sorted_vals))) - 1)
    return float(sorted_vals[max(idx, 0)])


def free_generation_eval(cfg, run_cfg, model, vocab, dataset, device) -> None:
    """Menu-free eval matching the sentence LM's training-time inference mode.

    The intent-target sentence LM was trained to decode the next intent
    sentence from the discourse context latent.  Its natural inference mode is
    therefore FREE GENERATION: greedily decode the next intent sentence
    token-by-token from the context latent (no candidate catalogue, no scored
    budget), ground it against the faithful environment's action texts,
    execute it, and append the observed outcome sentence.  Grading follows the
    original iGSM paper's free-generated solutions: goal reached; the only cap
    is a runaway stop at ``gen_step_cap_mult`` x necessary steps.
    ``gen_invalid_policy=fail`` (default) ends the episode on an invalid or
    unparseable generated definition; ``ignore`` appends the
    invalid-definition outcome sentence and continues.
    """
    from textjepa.data.faithful import INVALID_DEFINITION_OUTCOME, FaithfulEnv

    max_chunk_len = int(run_cfg.model.get("max_chunk_len", 48))
    cap_mult = float(cfg.get("gen_step_cap_mult", 4.0))
    invalid_policy = cfg.get("gen_invalid_policy", "fail")
    if invalid_policy not in {"fail", "ignore"}:
        raise ValueError(f"unknown gen_invalid_policy: {invalid_policy}")

    def tokens(texts):
        ids = [vocab.encode(t) for t in texts]
        L = max(len(i) for i in ids)
        out = torch.full((1, len(ids), L), vocab.pad_id, dtype=torch.long)
        for c, i in enumerate(ids):
            out[0, c, : len(i)] = torch.tensor(i)
        return out.to(device)

    def greedy_sentence(ctx) -> str:
        """Greedy autoregressive decode of one sentence from a context latent
        with the trained per-sentence decoder (same net as decode_ce)."""
        import torch.nn as nn

        generated: list[int] = []
        for _ in range(max_chunk_len):
            inp = torch.tensor(
                [[model.pad_id] + generated], dtype=torch.long, device=device
            )
            L = inp.shape[1]
            x = model.dec_tok(inp) + model.dec_pos[:, :L]
            causal = nn.Transformer.generate_square_subsequent_mask(
                L, device=device
            )
            h = model.decoder(x, ctx.unsqueeze(1), tgt_mask=causal)
            logits = model.dec_head(h)[0, -1]
            logits[vocab.pad_id] = float("-inf")
            nxt = int(logits.argmax().item())
            generated.append(nxt)
            if vocab.id_to_token[nxt].endswith("."):
                break
        return vocab.decode(generated).strip()

    episodes = []
    with torch.no_grad():
        for ep in range(cfg.n_episodes):
            problem, _ = dataset.problem(ep)
            env = FaithfulEnv(problem)
            prompt = problem.prompt_sentences
            n_necessary = len(problem.necessary)
            step_cap = int(math.ceil(cap_mult * n_necessary))
            action_by_text = {env.action_text(q): q for q in problem.params}
            history_texts: list[str] = []
            steps = n_invalid = n_unparseable = 0
            while not env.solved and steps < step_cap:
                batch = {
                    "prompt_tokens": tokens(prompt),
                    "prompt_mask": torch.ones(
                        1, len(prompt), dtype=torch.bool, device=device
                    ),
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
                ctx = (
                    states[:, len(history_texts) - 1] if history_texts else s0
                )
                text = greedy_sentence(ctx)
                pick = action_by_text.get(text)
                invalid = pick is None or pick not in env.feasible_actions()
                n_unparseable += int(pick is None)
                n_invalid += int(invalid)
                steps += 1
                history_texts.append(text)
                if invalid:
                    if invalid_policy == "fail":
                        break
                    history_texts.append(INVALID_DEFINITION_OUTCOME)
                else:
                    history_texts.append(env.step(pick))
            episodes.append({
                "success": bool(env.solved),
                "steps": steps,
                "necessary": n_necessary,
                "invalid": n_invalid,
                "unparseable": n_unparseable,
            })
            if (ep + 1) % 10 == 0:
                sr = sum(e["success"] for e in episodes) / len(episodes)
                print(
                    f"[ep {ep + 1}/{cfg.n_episodes}] success_rate={sr:.3f}",
                    flush=True,
                )
    n = len(episodes)
    total_steps = sum(e["steps"] for e in episodes)
    all_steps = sorted(e["steps"] for e in episodes)
    solved_steps = sorted(e["steps"] for e in episodes if e["success"])
    metrics = {
        "success_rate": sum(e["success"] for e in episodes) / n,
        "invalid_step_rate": (
            sum(e["invalid"] for e in episodes) / max(total_steps, 1)
        ),
        "unparseable_step_rate": (
            sum(e["unparseable"] for e in episodes) / max(total_steps, 1)
        ),
        "steps_mean": sum(all_steps) / n,
        "steps_median": _percentile(all_steps, 0.5),
        "steps_p90": _percentile(all_steps, 0.9),
        "solved_steps_mean": (
            sum(solved_steps) / len(solved_steps) if solved_steps
            else float("nan")
        ),
        "solved_steps_median": _percentile(solved_steps, 0.5),
        "solved_steps_p90": _percentile(solved_steps, 0.9),
        "necessary_mean": sum(e["necessary"] for e in episodes) / n,
        "n_episodes": n,
        "gen_step_cap_mult": cap_mult,
        "gen_invalid_policy": invalid_policy,
        "episode_invalid_rate": sum(e["invalid"] > 0 for e in episodes) / n,
    }
    out = {"sentlm_intent_freegen": metrics, "episodes": episodes}
    for k, v in metrics.items():
        print(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
    split = cfg.get("split", "val")
    split_suffix = "" if split == "val" else f"_{split}"
    dest = Path(
        cfg.out or Path(cfg.ckpt).parent
        / f"plan_freegen_sentlm_intent{split_suffix}.json"
    )
    dest.write_text(json.dumps(out, indent=2))
    print(f"saved to {dest}")


@hydra.main(config_path="../configs", config_name="plan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    ckpt = torch.load(cfg.ckpt, map_location=cfg.device, weights_only=False)
    run_cfg = OmegaConf.create(ckpt["cfg"])
    apply_eval_data_overrides(run_cfg, cfg)
    candidate_interface = cfg.get("candidate_interface", "feasible_menu")
    if candidate_interface not in {"feasible_menu", "full_catalogue"}:
        raise ValueError(f"unknown candidate_interface: {candidate_interface}")
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
    if cfg.get("eval_loops") is not None:
        encoder = model.state_model.encoder
        if not hasattr(encoder, "eval_loops"):
            raise ValueError(
                "eval_loops requires a recurrent sentence-LM checkpoint"
            )
        encoder.eval_loops = int(cfg.eval_loops)
    split = cfg.get("split", "val")
    dataset = build_dataset(run_cfg, vocab, split=split)
    score = cfg.get("score", "decoder")
    target_kind = run_cfg.train.get("target_kind", "outcome")
    if target_kind not in {"outcome", "intent"}:
        raise ValueError(f"unknown sentence LM target_kind: {target_kind}")
    if candidate_interface == "full_catalogue" and target_kind != "intent":
        raise ValueError(
            "full_catalogue candidate_interface requires target_kind=intent "
            "(outcome-scoring needs a feasible action to render the outcome)"
        )

    mode = cfg.get("mode", "candidate_ranking")
    if mode == "free_generation":
        if not faithful:
            raise ValueError(
                "free_generation mode is implemented for faithful iGSM only"
            )
        if target_kind != "intent":
            raise ValueError("free_generation requires target_kind=intent")
        free_generation_eval(cfg, run_cfg, model, vocab, dataset, device)
        return
    if mode != "candidate_ranking":
        raise ValueError(f"unknown mode: {mode}")

    def tokens(texts):
        ids = [vocab.encode(t) for t in texts]
        L = max(len(i) for i in ids)
        out = torch.full((1, len(ids), L), vocab.pad_id, dtype=torch.long)
        for c, i in enumerate(ids):
            out[0, c, : len(i)] = torch.tensor(i)
        return out.to(device)

    results: list[EpisodeResult] = []
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
            history_texts: list[str] = []
            n_necessary = len(necessary)
            # Proportional slack: same budget rule as FaithfulPlanner
            # (budget = necessary + slack + ceil(slack_frac * necessary)).
            # slack_frac defaults to 0, leaving historical runs unchanged.
            budget = n_necessary + cfg.slack + math.ceil(
                float(cfg.get("slack_frac", 0.0)) * n_necessary
            )
            n = n_invalid = n_distr = 0
            # Faithful full_catalogue only: the policy's OWN attempted-invalid
            # actions, masked until progress (same state-scoped semantics as
            # FaithfulPlanner; without it a deterministic argmin loops on a
            # no-op forever).
            mask_attempted = bool(cfg.get("mask_attempted", True))
            attempted: set = set()
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
                n_distr += int(pick not in necessary)
                if target_kind == "intent":
                    history_texts.append(
                        env.action_text(pick) if faithful
                        else action_phrase(problem, pick)
                    )
                if candidate_interface == "full_catalogue":
                    invalid = pick not in env.feasible_actions()
                    n_invalid += int(invalid)
                    history_texts.append(env.step_or_invalid(pick))
                    if faithful:
                        # Mask only WHILE the state is unchanged; reset to the
                        # resolved set on progress (an action infeasible now
                        # may become feasible after its dependencies resolve).
                        if invalid:
                            attempted.add(pick)
                        else:
                            attempted = set(env.resolved)
                else:
                    history_texts.append(env.step(pick))
                n += 1
            results.append(
                EpisodeResult(
                    env.solved, n, n_necessary, n_distr, n_invalid,
                    # The greedy policy stops the moment the goal is reached,
                    # so the executed-step count IS the solved-at step.
                    solved_at=n if env.solved else None,
                )
            )
    slack_curve = bool(cfg.get("slack_curve", False))
    metrics = aggregate_episodes(
        results, slack_curve=slack_curve, slack=cfg.slack
    )
    # ``invalid_rate`` is the historical field name in the LM baseline JSONs;
    # keep it as an alias of the shared ``invalid_action_rate``.
    metrics["invalid_rate"] = metrics["invalid_action_rate"]
    metrics.update({
        "length_normalized": bool(
            cfg.get("length_normalize", True) and score == "decoder"
        ),
        "candidate_interface": candidate_interface,
    })
    key = f"sentlm_{target_kind}_{score}"
    out = {key: metrics}
    out[key]["flop_measurement_supported"] = FlopCounterMode is not None
    if measure_flops and FlopCounterMode is not None:
        total_flops = int(flop_counter.get_total_flops())
        out[key].update({
            "measured_eval_flops": total_flops,
            "measured_flops_per_episode": total_flops / cfg.n_episodes,
        })
    print(json.dumps(out, indent=2))
    split_suffix = "" if split == "val" else f"_{split}"
    ci_suffix = (
        "" if candidate_interface == "feasible_menu" else f"_{candidate_interface}"
    )
    # A slack curve is scored from one generous-budget run, so it must not
    # overwrite the fixed-slack run at the same nominal slack.
    curve_suffix = "_slackcurve" if slack_curve else ""
    dest = Path(
        cfg.out or Path(cfg.ckpt).parent
        / f"plan_slack{cfg.slack}_sentlm_{target_kind}_{score}{ci_suffix}"
        f"{curve_suffix}{split_suffix}.json"
    )
    dest.write_text(json.dumps(out, indent=2))
    print(f"saved to {dest}")


if __name__ == "__main__":
    main()
