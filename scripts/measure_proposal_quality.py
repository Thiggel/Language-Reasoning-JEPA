"""Proposal-quality bench: token-head sampling vs a learned action density.

    .venv/bin/python scripts/measure_proposal_quality.py \
        --ckpt <best.pt> --flow-prior flow.pt --n-problems 100 --out q.json

Planning success confounds proposal with search and scoring.  This script
isolates the proposer.  For every state along a problem's demonstrated
solution trace it asks each arm for K candidates and records:

* ``recall_feasible``  - is ANY currently-feasible action among the K?
* ``recall_necessary`` - is any feasible action that is also on the solution
                         path among the K?  (the one that lets a step make
                         progress)
* ``recall_true_next`` - is the action the trace actually took among the K?
* ``parse_rate``       - fraction of the K that ground to executable text
* ``unique_per_state`` - distinct grounded actions per state (the diversity
                         number: ``prior_propose`` sits near 1.6)
* ``unique_text_per_state`` - distinct decoded strings, grounded or not
* ``decodes_per_state``- token-decode cost, so budgets can be matched on
                         compute rather than only on K.

All recall numbers use the environment's feasible set FOR MEASUREMENT ONLY;
no arm sees it.  Arms:

  token_head   the current proposer (greedy + nucleus, exact-text grounding)
  codebook_ground  k-means codebook over TRAINING action vectors, each row
               snapped to the nearest vector of THIS problem's catalogue and
               then cycle-ranked.  CATALOGUE-PRIVILEGED: it needs every intent
               of the problem enumerated, so its recall is an upper bound that
               a free proposer does not get for free.  Labelled as such.
  codebook_free  the same codebook rows decoded to text through the LDAD
               decoder with no catalogue -- the genuinely menu-free codebook,
               and the one the 2026-08-11 screen found at parse rate ~0.
  flow_rerank  oversample from the token head, keep K by p(a | s) + spread
  flow_dens    same, ranked by p(a | s) alone (diversity ablation)
  flow_decode  sample K codes from p(a | s), decode via the LDAD decoder
  code_prior   THE TARGET ARCHITECTURE: a learned state-conditioned DISCRETE
               prior proposes K action codes (distinct by construction), and
               a DETACHED decoder conditioned on the action code AND the
               problem context renders each to text.  No menu, no catalogue,
               no feasibility signal.  Requires --code-prior and
               --action-decoder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan_flat import build_eval_dataset, load_flat_run  # noqa: E402

from textjepa.planning.flat_search import FlatPlanner  # noqa: E402
from textjepa.utils import seed_everything  # noqa: E402

ARMS = ("token_head", "codebook_ground", "codebook_free",
        "flow_rerank", "flow_dens", "flow_decode", "code_prior")
FLOW_ARMS = ("flow_rerank", "flow_dens", "flow_decode")
CODEBOOK_ARMS = ("codebook_ground", "codebook_free")


@torch.no_grad()
def _codebook_ground(planner, history, env, fp, k, stats):
    """The planner's own codebook_ground root selection, isolated."""
    roots = list(fp.action_order)
    phrases = [planner.vocab.encode(env.action_text(q)) for q in roots]
    codes = planner._codes(history, phrases)
    code_of = {q: codes[i] for i, q in enumerate(roots)}
    state, _ = planner._state(history)
    # _filter_roots dispatches on planner.candidate_interface, so it must be
    # set here -- the bench builds the planner with the default interface.
    old_k, old_iface = planner.prior_top_k, planner.candidate_interface
    planner.prior_top_k, planner.candidate_interface = k, "codebook_ground"
    try:
        out = planner._filter_roots(roots, env, state, code_of)
    finally:
        planner.prior_top_k, planner.candidate_interface = old_k, old_iface
    stats["n_proposed"] += len(out)
    stats["n_parseable"] += len(out)   # catalogue actions are executable text
    stats["n_unique"] += len(out)
    return out


@torch.no_grad()
def _codebook_free(planner, history, env, k, stats):
    """Codebook rows decoded straight to text: no catalogue, no menu."""
    from textjepa.planning.ldad_decode import greedy_phrases

    state, _ = planner._state(history)
    codes = planner.codebook[:k].to(state.dtype)
    s_rep = state.unsqueeze(0).expand(codes.shape[0], -1)
    logits = planner.model.observed_action_decoder(
        planner.model.predict(s_rep, codes) - s_rep)
    phrases, _ = greedy_phrases(logits.float(), planner.vocab)
    stats["n_decoded"] += len(phrases)
    by_text = {env.action_text(q): q for q in env.fp.params}
    out, seen = [], set()
    for text in phrases:
        text = text.strip()
        if text in seen:
            continue
        seen.add(text)
        q = by_text.get(text)
        if q is not None:
            out.append(q)
    stats["n_proposed"] += len(phrases)
    stats["n_parseable"] += len(out)
    stats["n_unique"] += len(out)
    return out


@torch.no_grad()
def _code_prior(planner, history, env, k, stats, prior, decoder, temperature,
                sample, rng_seed):
    """Learned discrete prior -> detached context-conditioned decoder -> text.

    Nothing about the current problem's action set enters: the prior sees only
    the state, and the decoder sees only the action code plus the problem text
    that is already in the context window.  Grounding is the same exact-text
    lookup every menu-free arm gets, purely so the environment can execute the
    string.
    """
    state, ctx = planner._state(history)
    ctx = ctx[-planner.code_prior_max_ctx:]
    gen = None
    if sample:
        gen = torch.Generator(device=state.device)
        gen.manual_seed(int(rng_seed))
    idx, vecs = prior.propose(state.float().unsqueeze(0), k,
                              temperature=temperature, sample=sample,
                              generator=gen, ctx=ctx.float().unsqueeze(0))
    vecs = vecs[0].to(ctx.dtype)
    ctx_rep = ctx.unsqueeze(0).expand(k, -1, -1).float()
    mask = torch.zeros(k, ctx.shape[0], dtype=torch.bool, device=ctx.device)
    toks = decoder.generate(vecs.float(), ctx_rep, mask,
                            eos_id=planner.vocab.pad_id)
    texts = [planner.vocab.decode(t).strip() for t in toks]
    stats["n_decoded"] += k
    stats["n_proposed"] += k
    stats["n_unique_text"] = len(set(texts))
    by_text = {env.action_text(q): q for q in env.fp.params}
    out, seen = [], set()
    for text in texts:
        q = by_text.get(text)
        if q is None:
            continue
        stats["n_parseable"] += 1
        if q in seen:
            continue
        seen.add(q)
        out.append(q)
    stats["n_unique"] += len(out)
    return out


def _new_stats() -> dict:
    return {"n_proposed": 0, "n_parseable": 0, "n_unique": 0, "n_decoded": 0}


@torch.no_grad()
def run_arm(planner: FlatPlanner, arm: str, k: int, dataset, n_problems: int,
            vocab, seed: int) -> dict:
    planner.prior_samples = k
    planner.flow_diversity = (arm == "flow_rerank")
    acc = {"states": 0, "feas": 0, "nec": 0, "true": 0, "parse": 0,
           "proposed": 0, "unique": 0, "unique_text": 0, "decodes": 0,
           "empty": 0}
    for i in range(n_problems):
        fp = dataset.problem(i)[0]
        env = fp.make_env()
        history = [t for s in fp.prompt_sentences for t in vocab.encode(s)]
        step = 0
        while not env.solved:
            feasible = set(env.feasible_actions())
            todo = [q for q in feasible if q in fp.necessary]
            if not todo:
                break
            true_next = todo[0]
            stats = _new_stats()
            rng_seed = (seed * 1_000_003 + i * 997 + step) % (1 << 30)
            if arm == "token_head":
                out = planner._propose(history, env, rng_seed, stats)
                stats["n_decoded"] += k  # the baseline decodes exactly K phrases
            elif arm == "codebook_ground":
                out = _codebook_ground(planner, history, env, fp, k, stats)
            elif arm == "codebook_free":
                out = _codebook_free(planner, history, env, k, stats)
            elif arm == "code_prior":
                out = _code_prior(
                    planner, history, env, k, stats, planner.code_prior,
                    planner.action_decoder, planner.code_prior_temperature,
                    planner.code_prior_sample, rng_seed)
            elif arm == "flow_decode":
                out = planner._propose_flow_decode(history, env, rng_seed, stats)
            else:
                out = planner._propose_flow_rerank(history, env, rng_seed, stats)
            uniq = list(dict.fromkeys(out))
            acc["states"] += 1
            acc["feas"] += int(any(q in feasible for q in uniq))
            acc["nec"] += int(any(q in feasible and q in fp.necessary for q in uniq))
            acc["true"] += int(true_next in uniq)
            acc["parse"] += stats["n_parseable"]
            acc["proposed"] += stats["n_proposed"]
            acc["unique"] += len(uniq)
            acc["unique_text"] += stats.get("n_unique_text", len(uniq))
            acc["decodes"] += stats["n_decoded"]
            acc["empty"] += int(not uniq)
            history = history + vocab.encode(env.action_text(true_next))
            history = history + vocab.encode(env.step(true_next))
            step += 1
    n = max(acc["states"], 1)
    return {
        "arm": arm, "K": k, "n_states": acc["states"],
        "recall_feasible": acc["feas"] / n,
        "recall_necessary": acc["nec"] / n,
        "recall_true_next": acc["true"] / n,
        "parse_rate": acc["parse"] / max(acc["proposed"], 1),
        "unique_per_state": acc["unique"] / n,
        "decodes_per_state": acc["decodes"] / n,
        "no_proposal_state_rate": acc["empty"] / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--flow-prior", default=None)
    ap.add_argument("--code-prior", default=None,
                    help="scripts/train_code_prior.py artifact")
    ap.add_argument("--action-decoder", default=None,
                    help="scripts/train_action_decoder.py artifact")
    ap.add_argument("--code-prior-temperature", type=float, default=1.0)
    ap.add_argument("--code-prior-sample", action="store_true")
    ap.add_argument("--code-prior-max-ctx", type=int, default=768)
    ap.add_argument("--n-problems", type=int, default=100)
    ap.add_argument("--split-seed", type=int, default=2)
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--flow-oversample", type=int, default=64)
    ap.add_argument("--codebook-k", type=int, default=64)
    ap.add_argument("--codebook-problems", type=int, default=64)
    ap.add_argument("--prior-temperature", type=float, default=1.3)
    ap.add_argument("--prior-top-p", type=float, default=0.95)
    ap.add_argument("--flow-temperature", type=float, default=1.0)
    ap.add_argument("--op-lo", type=int, default=None)
    ap.add_argument("--op-hi", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    model, vocab, cfg = load_flat_run(args.ckpt, str(device))
    flow = None
    if args.flow_prior:
        from textjepa.planning.flow_prior import ConditionalFlowPrior
        flow = ConditionalFlowPrior.load(args.flow_prior, map_location=device).to(device)
    code_prior = action_decoder = None
    if args.code_prior or args.action_decoder:
        if not (args.code_prior and args.action_decoder):
            raise SystemExit(
                "the code_prior arm needs BOTH --code-prior and "
                "--action-decoder")
        from textjepa.planning.action_decoder import ContextActionDecoder
        from textjepa.planning.code_prior import CodePrior
        code_prior = CodePrior.load(args.code_prior,
                                    map_location=device).to(device).eval()
        action_decoder = ContextActionDecoder.load(
            args.action_decoder, map_location=device).to(device).eval()
    dataset, caps = build_eval_dataset(cfg, vocab, args.n_problems,
                                       args.split_seed, op_lo=args.op_lo,
                                       op_hi=args.op_hi)
    planner = FlatPlanner(
        model, vocab, device, max_len=int(cfg["model"]["max_len"]),
        prior_top_p=args.prior_top_p, prior_temperature=args.prior_temperature,
        flow_prior=flow, flow_oversample=args.flow_oversample,
        flow_temperature=args.flow_temperature,
        codebook_k=args.codebook_k,
    )
    # Attached to the planner (not constructor arguments) so no existing
    # FlatPlanner signature or behaviour changes.
    planner.code_prior = code_prior
    planner.action_decoder = action_decoder
    planner.code_prior_temperature = args.code_prior_temperature
    planner.code_prior_sample = bool(args.code_prior_sample)
    planner.code_prior_max_ctx = args.code_prior_max_ctx
    if any(a in CODEBOOK_ARMS for a in args.arms):
        train_ds, _ = build_eval_dataset(cfg, vocab, args.codebook_problems,
                                         cfg["data"]["train_seed"])
        planner.fit_action_prior(
            [train_ds.problem(i)[0] for i in range(args.codebook_problems)])
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if args.precision == "bf16" and device.type == "cuda"
           else torch.autocast("cpu", enabled=False))
    rows = []
    with torch.no_grad(), ctx:
        for arm in args.arms:
            if arm in FLOW_ARMS and flow is None:
                continue
            if arm == "code_prior" and code_prior is None:
                continue
            for k in args.ks:
                row = run_arm(planner, arm, k, dataset, args.n_problems,
                              vocab, args.seed)
                rows.append(row)
                print(json.dumps(row), flush=True)
    Path(args.out).write_text(json.dumps({
        "protocol": {
            "ckpt": args.ckpt, "flow_prior": args.flow_prior,
            "code_prior": args.code_prior,
            "action_decoder": args.action_decoder,
            "code_prior_temperature": args.code_prior_temperature,
            "code_prior_sample": bool(args.code_prior_sample),
            "n_problems": args.n_problems, "split_seed": args.split_seed,
            "caps": caps, "flow_oversample": args.flow_oversample,
            "codebook_k": args.codebook_k,
            "codebook_problems": args.codebook_problems,
            "prior_temperature": args.prior_temperature,
            "prior_top_p": args.prior_top_p,
            "note": "recall_* use the environment feasible set FOR MEASUREMENT "
                    "ONLY; no arm is shown a menu",
        },
        "rows": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
