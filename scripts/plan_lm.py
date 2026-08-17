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


def _percentile(sorted_vals: list, q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, int(math.ceil(q * len(sorted_vals))) - 1)
    return float(sorted_vals[max(idx, 0)])


def free_generation_eval(
    cfg, run_cfg, model, vocab, dataset, device, score_kind
) -> None:
    """Menu-free eval matching the LM's training-time inference mode.

    The intent-policy LM was trained on causal streams
    ``prompt, intent_1, outcome_1, ...`` with next-token CE on intent tokens.
    Its natural inference mode is therefore FREE GENERATION of the next intent
    phrase — no candidate catalogue, no scored budget.  We greedily decode one
    phrase at a time (a phrase ends at a token ending in "."), ground it
    against the faithful environment's action texts, execute it, and append
    the observed outcome; an unparseable or infeasible phrase yields the
    invalid-definition outcome sentence (env unchanged).  An episode is graded
    like a generated solution in the original iGSM paper: goal reached
    (query defined => answer computed).  The only cap is a runaway stop at
    ``gen_step_cap_mult`` x necessary steps — it is NOT a scored budget.
    """
    from textjepa.data.faithful import INVALID_DEFINITION_OUTCOME, FaithfulEnv

    max_len = int(run_cfg.model.get("max_len", 4096))
    cap_mult = float(cfg.get("gen_step_cap_mult", 4.0))
    phrase_tok_cap = int(cfg.get("gen_phrase_token_cap", 24))
    # "fail": an invalid/unparseable generated definition ends the episode
    # unsolved — matching how a free-generated solution containing an invalid
    # step would be graded in the original iGSM paper (the solution is simply
    # wrong; there is no retry loop).  "ignore": append the invalid-definition
    # outcome sentence and keep generating (note: that sentence never occurs
    # in the training streams, so continuation is off-distribution).
    invalid_policy = cfg.get("gen_invalid_policy", "fail")
    if invalid_policy not in {"fail", "ignore"}:
        raise ValueError(f"unknown gen_invalid_policy: {invalid_policy}")
    # "env" (default, historical): after each grounded intent the TRUE
    # environment outcome sentence is appended — matching the intent-policy
    # LM's training streams.  "model": the LM free-generates its own outcome
    # sentence (definition + arithmetic) which is kept in the context, and
    # only the intent (definition) sentences are grounded/stepped through the
    # environment — the original iGSM paper's fully free-generated solution.
    outcome_source = cfg.get("gen_outcome", "env")
    if outcome_source not in {"env", "model"}:
        raise ValueError(f"unknown gen_outcome: {outcome_source}")
    outcome_tok_cap = int(cfg.get("gen_outcome_token_cap", 96))

    def decode_sentence(history: list[int], cap: int) -> list[int]:
        """Greedy autoregressive decode until a token ending in '.'."""
        phrase: list[int] = []
        for _ in range(cap):
            ctx = (history + phrase)[-max_len:]
            toks = torch.tensor(
                ctx, dtype=torch.long, device=device
            ).unsqueeze(0)
            logits = model(toks)[0, -1]
            logits[vocab.pad_id] = float("-inf")
            nxt = int(logits.argmax().item())
            phrase.append(nxt)
            if vocab.id_to_token[nxt].endswith("."):
                break
        return phrase

    def last_int(text: str):
        ints = [t for t in text.split() if t.lstrip("-").isdigit()]
        return int(ints[-1]) if ints else None

    episodes = []
    with torch.no_grad():
        for ep in range(cfg.n_episodes):
            problem, _ = dataset.problem(ep)
            env = FaithfulEnv(problem)
            history = [
                t for s in problem.prompt_sentences for t in vocab.encode(s)
            ]
            n_necessary = len(problem.necessary)
            step_cap = int(math.ceil(cap_mult * n_necessary))
            action_by_text = {
                env.action_text(q): q for q in problem.params
            }
            steps = n_invalid = n_unparseable = 0
            n_valid = n_value_match = 0
            answer_correct = None
            while not env.solved and steps < step_cap:
                phrase = decode_sentence(history, phrase_tok_cap)
                history += phrase
                text = vocab.decode(phrase).strip()
                pick = action_by_text.get(text)
                invalid = pick is None or pick not in env.feasible_actions()
                n_unparseable += int(pick is None)
                n_invalid += int(invalid)
                steps += 1
                if invalid and invalid_policy == "fail":
                    break
                true_outcome = (
                    INVALID_DEFINITION_OUTCOME if invalid else env.step(pick)
                )
                if outcome_source == "model":
                    gen_out = decode_sentence(history, outcome_tok_cap)
                    history += gen_out
                    if not invalid:
                        n_valid += 1
                        # Grade the model's OWN computed value (last integer
                        # of its generated outcome sentence) against the true
                        # value — robust to arbitrary temp-variable letters.
                        ok = last_int(vocab.decode(gen_out)) == last_int(
                            true_outcome
                        )
                        n_value_match += int(ok)
                        if env.solved:
                            answer_correct = bool(
                                ok and last_int(vocab.decode(gen_out))
                                == problem.answer
                            )
                else:
                    history += vocab.encode(true_outcome)
            episodes.append({
                "success": bool(env.solved),
                "steps": steps,
                "necessary": n_necessary,
                "invalid": n_invalid,
                "unparseable": n_unparseable,
                "valid_steps": n_valid,
                "value_match": n_value_match,
                "answer_correct": answer_correct,
                "success_answer": bool(env.solved)
                and (answer_correct is True or outcome_source == "env"),
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
        "gen_outcome": outcome_source,
        # Model-outcome mode only: does the model's own arithmetic match?
        "outcome_value_match_rate": (
            sum(e["value_match"] for e in episodes)
            / max(sum(e["valid_steps"] for e in episodes), 1)
        ),
        # Success requiring the model's OWN final answer to be correct
        # (equals success_rate in gen_outcome=env mode).
        "success_answer_rate": (
            sum(e["success_answer"] for e in episodes) / n
        ),
        "episode_invalid_rate": sum(
            e["invalid"] > 0 for e in episodes
        ) / n,
    }
    out = {f"lm_{score_kind}_freegen": metrics, "episodes": episodes}
    for k, v in metrics.items():
        print(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
    split = cfg.get("split", "val")
    split_suffix = "" if split == "val" else f"_{split}"
    dest = Path(
        cfg.out or Path(cfg.ckpt).parent
        / f"plan_freegen_lm_{score_kind}{split_suffix}.json"
    )
    dest.write_text(json.dumps(out, indent=2))
    print(f"saved to {dest}")


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
        from textjepa.utils.checkpoint import build_vocab_for_config

        # Uses the checkpoint's own vocab caps (data.vocab_max_op/max_edge),
        # NOT the eval overrides — the embedding table is indexed by it.
        vocab = build_vocab_for_config(run_cfg)
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

    mode = cfg.get("mode", "candidate_ranking")
    if mode == "free_generation":
        if not faithful:
            raise ValueError("free_generation mode is implemented for faithful iGSM only")
        free_generation_eval(cfg, run_cfg, model, vocab, dataset, device, score_kind)
        return
    if mode != "candidate_ranking":
        raise ValueError(f"unknown mode: {mode}")

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
                # Micro-batch: at long-budget OOD evals the history grows to
                # hundreds of sentences and scoring the whole catalogue in one
                # batch OOMs an 80 GB card (attention is quadratic in length).
                chunk = int(cfg.get("candidate_chunk", 4))
                lp = torch.cat([
                    model.sequence_logprob(
                        toks[i:i + chunk].to(device),
                        torch.full(
                            (min(chunk, len(cands) - i),), len(history),
                            device=device,
                        ),
                    )
                    for i in range(0, len(cands), chunk)
                ])
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
