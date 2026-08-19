"""Test-time-compute baselines for the token / sentence LM.

Why this exists
---------------
Our JEPA planner reports success as a function of *planning depth*
{1, 4, 16}.  A reviewer will immediately ask whether the JEPA number is just
"more compute at inference time".  The honest answer requires giving the LM
baseline the same knob: a family of inference procedures indexed by an
explicit compute parameter N (or K), evaluated under exactly the protocol the
headline LM row uses -- free generation, no scored budget, runaway cap at
``cap_mult`` x necessary steps, success + steps-used distribution
(``scripts/plan_lm.py +mode=free_generation``).

Three baselines, all standard, none invented here:

``sample_rerank``   (N in {1,2,4,8,16})
    Draw N complete solutions by nucleus sampling, score each by the LM's own
    sequence log-probability over the tokens *it generated*, execute the best.
    Reported for both sum log-prob and length-normalised (mean) log-prob,
    because sum log-prob is biased toward solutions that terminated early.
    ``pass_at_n`` (any of the N solved) is reported as the oracle-selection
    ceiling -- labelled as an ORACLE row, not a baseline.

``self_consistency`` (N in {1,2,4,8,16})
    Same N samples; majority vote over the *final answer* the model wrote for
    the queried variable.  iGSM has a checkable final answer, so this is the
    textbook self-consistency baseline.  The episode-level "success" reported
    for this method is the success of the rollout carrying the winning
    answer; ``answer_accuracy`` (majority answer == true answer) is the metric
    self-consistency is actually about.

``looped`` (K in {1,2,4,8,16})
    Extra *forward-pass depth* rather than extra samples.
    - If the checkpoint is a recurrent (weight-shared / looped) LM
      (``model.recurrent=true`` -> ``LoopedTransformerEncoder``), K is the
      genuine number of shared-block loops per forward: ``eval_loops=K``.
      This is the architecture the campaign log's "looped LM" plan means.
    - Otherwise the checkpoint has a fixed depth and there is no loop count to
      turn.  We then run the *cheap standard substitute* and label it exactly:
      ``looped_reread`` -- before decoding each solution sentence the model's
      own solution-so-far is appended once more to the context, K-1 extra
      times, so the model re-reads its own draft K times before committing to
      the next step.  This is prefix re-reading / self-conditioning, NOT a
      recurrent-depth model, and it is off-distribution for a checkpoint that
      never saw a repeated prefix.  It is reported so the compute axis is
      populated, with that caveat carried in the JSON
      (``looped_variant`` field).

Compute accounting
------------------
Every method reports a ``ComputeCounter`` (see
``textjepa/utils/compute_counter.py``): backbone forwards, backbone token
positions (the FLOP-proportional axis), and generated tokens.  The JEPA
planner reports the same object from ``scripts/plan_flat.py``, so success can
be plotted against one comparable x-axis for both families.

Usage
-----
    .venv/bin/python scripts/plan_lm_testtime.py \
        --ckpt .../model/best.pt --method sample_rerank --n 1 2 4 8 16 \
        --n-episodes 200 --out out.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
from omegaconf import OmegaConf

from textjepa.data.faithful import INVALID_DEFINITION_OUTCOME, FaithfulEnv
from textjepa.models.lm_baseline import DecoderLM
from textjepa.utils import seed_everything
from textjepa.utils.checkpoint import build_dataset, build_vocab_for_config
from textjepa.utils.compute_counter import ComputeCounter

METHODS = ("sample_rerank", "self_consistency", "looped")


def _percentile(vals: list, q: float) -> float:
    v = sorted(vals)
    if not v:
        return float("nan")
    idx = min(len(v) - 1, int(math.ceil(q * len(v))) - 1)
    return float(v[max(idx, 0)])


def _last_int(text: str):
    ints = [
        t.rstrip(".;,") for t in text.split()
        if t.rstrip(".;,").lstrip("-").isdigit()
    ]
    return int(ints[-1]) if ints else None


class Rollout:
    """One complete free-generated solution attempt, already graded."""

    def __init__(self, solved, steps, necessary, invalid, unparseable,
                 valid_steps, value_match, answer, answer_correct, logp,
                 n_gen_tokens):
        self.solved = solved
        self.steps = steps
        self.necessary = necessary
        self.invalid = invalid
        self.unparseable = unparseable
        self.valid_steps = valid_steps
        self.value_match = value_match
        self.answer = answer                # model's own final answer, or None
        self.answer_correct = answer_correct
        self.logp_sum = logp
        self.n_gen_tokens = n_gen_tokens

    @property
    def logp_mean(self) -> float:
        return self.logp_sum / max(self.n_gen_tokens, 1)

    def as_dict(self) -> dict:
        return {
            "solved": self.solved, "steps": self.steps,
            "necessary": self.necessary, "invalid": self.invalid,
            "unparseable": self.unparseable, "answer": self.answer,
            "answer_correct": self.answer_correct,
            "logp_sum": self.logp_sum, "logp_mean": self.logp_mean,
            "n_gen_tokens": self.n_gen_tokens,
        }


class Generator:
    """Free-generation rollouts under the plan_lm.py free_generation protocol.

    ``gen_outcome=model`` and ``gen_invalid_policy=fail`` are hard-wired: this
    is the headline protocol (the model writes its own arithmetic; a generated
    definition that is unparseable or infeasible ends the attempt unsolved,
    exactly as an invalid step would invalidate a generated iGSM solution).
    The environment is used for grounding/grading only -- never to choose a
    token, and never to choose which of the N samples to execute.
    """

    def __init__(self, model, vocab, device, max_len, counter,
                 phrase_cap=24, outcome_cap=96, cap_mult=4.0,
                 temperature=1.0, top_p=0.95, reread_loops=1,
                 num_loops=None):
        self.model = model
        self.vocab = vocab
        self.device = device
        self.max_len = int(max_len)
        self.counter = counter
        self.phrase_cap = int(phrase_cap)
        self.outcome_cap = int(outcome_cap)
        self.cap_mult = float(cap_mult)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.reread_loops = int(reread_loops)
        self.num_loops = num_loops
        self.last_reread_repeats = 0

    def _logits(self, ctx: list[int]) -> torch.Tensor:
        toks = torch.tensor(ctx[-self.max_len:], dtype=torch.long,
                            device=self.device).unsqueeze(0)
        kw = {} if self.num_loops is None else {"num_loops": self.num_loops}
        out = self.model(toks, **kw)[0, -1].float()
        # A looped forward with K loops costs K times a single-layer-stack
        # pass; charge it that way so the compute axis is honest.
        mult = 1 if self.num_loops is None else int(self.num_loops)
        self.counter.backbone(mult, toks.shape[1], generated=1)
        out[self.vocab.pad_id] = float("-inf")
        return out

    def _reread_context(self, prompt: list[int], solution: list[int],
                        pending: list[int]) -> list[int]:
        """looped_reread: repeat the model's own solution-so-far K-1 extra
        times before the live continuation.  K=1 is the plain context."""
        base = len(prompt) + len(solution) + len(pending)
        room = self.max_len - base
        # Never let re-reading push the PROMPT out of the window (the context
        # is truncated from the left): drop repeats that do not fit.
        repeats = max(self.reread_loops - 1, 0)
        if solution:
            repeats = min(repeats, max(room // len(solution), 0))
        self.last_reread_repeats = repeats
        return list(prompt) + solution * repeats + solution + pending

    def _decode(self, prompt, solution, cap, gen, greedy):
        """Decode one sentence; returns (tokens, summed logprob)."""
        out: list[int] = []
        logp = 0.0
        for _ in range(cap):
            logits = self._logits(self._reread_context(prompt, solution, out))
            lp = torch.log_softmax(logits, -1)
            if greedy:
                nxt = int(logits.argmax().item())
            else:
                probs = torch.softmax(logits / self.temperature, -1)
                sp, si = probs.sort(descending=True)
                keep = (sp.cumsum(0) - sp) < self.top_p
                keep[0] = True
                sp = sp * keep
                sp = sp / sp.sum()
                nxt = int(si[torch.multinomial(sp, 1, generator=gen)].item())
            logp += float(lp[nxt].item())
            out.append(nxt)
            if self.vocab.id_to_token[nxt].endswith("."):
                break
        return out, logp

    @torch.no_grad()
    def rollout(self, problem, gen, greedy: bool) -> Rollout:
        env = FaithfulEnv(problem)
        prompt = [t for s in problem.prompt_sentences
                  for t in self.vocab.encode(s)]
        solution: list[int] = []
        n_nec = len(problem.necessary)
        cap = int(math.ceil(self.cap_mult * n_nec))
        steps = n_invalid = n_unparseable = n_valid = n_match = 0
        logp = 0.0
        answer = None
        answer_correct = None
        action_by_text = {env.action_text(q): q for q in problem.params}
        while not env.solved and steps < cap:
            phrase, lp = self._decode(prompt, solution, self.phrase_cap, gen,
                                      greedy)
            solution += phrase
            logp += lp
            text = self.vocab.decode(phrase).strip()
            pick = action_by_text.get(text)
            invalid = pick is None or pick not in env.feasible_actions()
            n_unparseable += int(pick is None)
            n_invalid += int(invalid)
            steps += 1
            if invalid:
                break  # gen_invalid_policy = fail
            true_outcome = env.step(pick)
            gen_out, lp2 = self._decode(prompt, solution, self.outcome_cap,
                                        gen, greedy)
            solution += gen_out
            logp += lp2
            n_valid += 1
            written = _last_int(self.vocab.decode(gen_out))
            ok = written == _last_int(true_outcome)
            n_match += int(ok)
            if env.solved:
                answer = written
                answer_correct = bool(ok and written == problem.answer)
        return Rollout(bool(env.solved), steps, n_nec, n_invalid,
                       n_unparseable, n_valid, n_match, answer,
                       answer_correct, logp, len(solution))


def summarize(selected: list[Rollout], extra: dict) -> dict:
    n = max(len(selected), 1)
    solved = [r for r in selected if r.solved]
    total_steps = sum(r.steps for r in selected)
    out = {
        "success_rate": sum(r.solved for r in selected) / n,
        "success_answer_rate": sum(bool(r.answer_correct)
                                   for r in selected) / n,
        "steps_mean": total_steps / n,
        "steps_median": _percentile([r.steps for r in selected], 0.5),
        "steps_p90": _percentile([r.steps for r in selected], 0.9),
        "necessary_mean": sum(r.necessary for r in selected) / n,
        "solved_steps_mean": (sum(r.steps for r in solved) / len(solved)
                              if solved else float("nan")),
        "solved_steps_over_necessary_mean": (
            sum(r.steps / r.necessary for r in solved) / len(solved)
            if solved else float("nan")),
        "solved_steps_over_necessary_median": _percentile(
            [r.steps / r.necessary for r in solved], 0.5),
        "solved_exact_necessary_frac": (
            sum(r.steps == r.necessary for r in solved) / len(solved)
            if solved else float("nan")),
        "invalid_step_rate": sum(r.invalid for r in selected)
        / max(total_steps, 1),
        "unparseable_step_rate": sum(r.unparseable for r in selected)
        / max(total_steps, 1),
        "outcome_value_match_rate": sum(r.value_match for r in selected)
        / max(sum(r.valid_steps for r in selected), 1),
        "n_episodes": len(selected),
    }
    out.update(extra)
    return out


def run_method(args, model, vocab, dataset, device, max_len) -> dict:
    """Returns {setting_label: {metrics..., compute: ...}}.

    For the sample-based methods the N-axis is NESTED: max(N) samples are
    drawn once per episode and the N=k row uses the first k of them (sample 0
    is the greedy decode, so N=1 reproduces the headline free-generation row
    exactly).  Nesting halves the cost of the sweep and is the usual way
    best-of-N / self-consistency curves are produced.
    """
    results = {}
    recurrent = hasattr(getattr(model, "blocks", None), "eval_loops")
    if args.method == "looped":
        axis = args.k
        settings = []
        for val in axis:
            if recurrent:
                settings.append((val, "recurrent_eval_loops", int(val), 1))
            else:
                settings.append((
                    val, "looped_reread (NOT a recurrent-depth model)",
                    None, int(val)))
    else:
        axis = sorted(args.n)
        settings = [(max(axis), None, None, 1)]

    for val, looped_variant, num_loops, reread in settings:
        n_draw = 1 if args.method == "looped" else int(val)
        greedy = args.method == "looped"
        gen_engine = Generator(
            model, vocab, device, max_len, None,
            phrase_cap=args.phrase_token_cap,
            outcome_cap=args.outcome_token_cap, cap_mult=args.cap_mult,
            temperature=args.temperature, top_p=args.top_p,
            reread_loops=reread, num_loops=num_loops,
        )
        # rolls_by_ep[ep] = list of n_draw graded rollouts
        rolls_by_ep, answers = [], []
        for ep in range(args.n_episodes):
            problem, _ = dataset.problem(ep)
            g = torch.Generator(device=device).manual_seed(
                args.seed * 100003 + ep)
            rolls = []
            for i in range(n_draw):
                gen_engine.counter = ComputeCounter()
                r = gen_engine.rollout(problem, g, greedy=greedy or i == 0)
                r.compute = gen_engine.counter.totals()
                rolls.append(r)
            rolls_by_ep.append(rolls)
            answers.append(problem.answer)
            if (ep + 1) % 10 == 0:
                sr = sum(rr[0].solved for rr in rolls_by_ep) / len(rolls_by_ep)
                print(f"[{args.method} {val}] ep {ep+1}/{args.n_episodes} "
                      f"greedy_success={sr:.3f}", flush=True)

        ns = [val] if args.method == "looped" else axis
        for nn in ns:
            take = 1 if args.method == "looped" else int(nn)
            counter = ComputeCounter()
            selected, per_ep = [], []
            for rolls_all, true_ans in zip(rolls_by_ep, answers):
                rolls = rolls_all[:take]
                counter.new_episode()
                for r in rolls:
                    counter.backbone_forwards += r.compute["backbone_forwards"]
                    counter.backbone_token_positions += r.compute[
                        "backbone_token_positions"]
                    counter.generated_tokens += r.compute["generated_tokens"]
                votes = Counter(r.answer for r in rolls if r.answer is not None)
                win = votes.most_common(1)[0][0] if votes else None
                sc_pick = (next(r for r in rolls if r.answer == win)
                           if win is not None else rolls[0])
                rr_mean = max(rolls, key=lambda r: r.logp_mean)
                rr_sum = max(rolls, key=lambda r: r.logp_sum)
                pick = {"self_consistency": sc_pick,
                        "sample_rerank": rr_mean}.get(args.method, rolls[0])
                selected.append(pick)
                per_ep.append({
                    **pick.as_dict(),
                    "majority_answer": win,
                    "majority_correct": bool(win is not None
                                             and win == true_ans),
                    "sc_solved": sc_pick.solved,
                    "rerank_mean_solved": rr_mean.solved,
                    "rerank_mean_answer": bool(rr_mean.answer_correct),
                    "rerank_sum_solved": rr_sum.solved,
                    "rerank_sum_answer": bool(rr_sum.answer_correct),
                    "pass_at_n": any(r.solved for r in rolls),
                    "answer_pass_at_n": any(bool(r.answer_correct)
                                            for r in rolls),
                })
            n = max(len(per_ep), 1)
            extra = {
                # ORACLE rows: selection BY THE ENVIRONMENT, i.e. the ceiling
                # any selection rule could reach.  Never quote as a baseline.
                "oracle_pass_at_n": sum(e["pass_at_n"] for e in per_ep) / n,
                "oracle_answer_pass_at_n": sum(
                    e["answer_pass_at_n"] for e in per_ep) / n,
            }
            if args.method != "looped":
                extra.update({
                    "self_consistency_answer_accuracy": sum(
                        e["majority_correct"] for e in per_ep) / n,
                    "self_consistency_success_rate": sum(
                        e["sc_solved"] for e in per_ep) / n,
                    "rerank_meanlogp_success_rate": sum(
                        e["rerank_mean_solved"] for e in per_ep) / n,
                    "rerank_meanlogp_success_answer_rate": sum(
                        e["rerank_mean_answer"] for e in per_ep) / n,
                    "rerank_sumlogp_success_rate": sum(
                        e["rerank_sum_solved"] for e in per_ep) / n,
                    "rerank_sumlogp_success_answer_rate": sum(
                        e["rerank_sum_answer"] for e in per_ep) / n,
                })
            m = summarize(selected, extra)
            m["compute"] = counter.report()
            axis_name = "K" if args.method == "looped" else "N"
            m["setting"] = {"method": args.method, axis_name: int(nn),
                            "looped_variant": looped_variant,
                            "temperature": args.temperature,
                            "top_p": args.top_p}
            label = f"{args.method}_{axis_name}{nn}"
            print(f"== {label}: success={m['success_rate']:.3f} "
                  f"answer={m['success_answer_rate']:.3f} "
                  f"pass@n={m['oracle_pass_at_n']:.3f} gen_tokens/ep="
                  f"{m['compute']['per_episode']['generated_tokens_per_episode']:.0f}",
                  flush=True)
            m["episodes"] = per_ep
            results[label] = m
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--n", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--cap-mult", type=float, default=4.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--phrase-token-cap", type=int, default=24)
    ap.add_argument("--outcome-token-cap", type=int, default=96)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="val")
    ap.add_argument("--eval-max-op", type=int, default=None)
    ap.add_argument("--eval-max-edge", type=int, default=None)
    ap.add_argument("--eval-op-lo", type=int, default=None)
    ap.add_argument("--eval-op-hi", type=int, default=None)
    ap.add_argument("--eval-sample-max-tries", type=int, default=40000)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    run_cfg = OmegaConf.create(ckpt["cfg"])
    if run_cfg.data.get("name", "igsm") != "igsm_real":
        raise ValueError("test-time-compute baselines are faithful-iGSM only")
    for key, val in (("max_op", args.eval_max_op),
                     ("max_edge", args.eval_max_edge)):
        if val is not None:
            run_cfg.data[key] = val
    if args.eval_op_lo is not None or args.eval_op_hi is not None:
        lo, hi = list(run_cfg.data.op_range)
        run_cfg.data.op_range = [args.eval_op_lo or lo, args.eval_op_hi or hi]
    run_cfg.data.distractor_prob = 0
    run_cfg.data.sample_max_tries = int(args.eval_sample_max_tries)
    vocab = build_vocab_for_config(run_cfg)
    model = DecoderLM(vocab_size=len(vocab), pad_id=vocab.pad_id,
                      **run_cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dataset = build_dataset(run_cfg, vocab, split=args.split)
    max_len = int(run_cfg.model.get("max_len", 4096))

    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if args.precision == "bf16" and device.type == "cuda"
           else torch.autocast("cpu", enabled=False))
    with torch.no_grad(), ctx:
        results = run_method(args, model, vocab, dataset, device, max_len)
    results["protocol"] = {
        "ckpt": args.ckpt, "method": args.method,
        "n_episodes": args.n_episodes, "cap_mult": args.cap_mult,
        "gen_outcome": "model", "gen_invalid_policy": "fail",
        "budget": "none (runaway cap only)",
        "selection_signal": {
            "sample_rerank": "LM sequence log-probability (model-internal)",
            "self_consistency": "majority vote over model-written final answers",
            "looped": "none (single rollout, extra forward-pass depth)",
        }[args.method],
        "recurrent_checkpoint": hasattr(
            getattr(model, "blocks", None), "eval_loops"),
        "data": {"max_op": run_cfg.data.max_op,
                 "max_edge": run_cfg.data.max_edge,
                 "op_range": list(run_cfg.data.op_range),
                 "split": args.split},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
