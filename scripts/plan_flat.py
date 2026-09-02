"""Planning eval for flat-backbone intent JEPA checkpoints (faithful iGSM).

    .venv/bin/python scripts/plan_flat.py --ckpt <best.pt> --interface feasible_menu \
        --lookahead 1 --n-episodes 100 --out out.json

No scored budget: episodes run until solved or a runaway cap of
``--cap-mult`` x necessary steps; success + steps-used distribution are
reported together with random / first-feasible reference rows under the same
rules.  Lookahead > 1 is oracle-free (deeper slots are drawn from the root
pool, never from the environment's future menus).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from textjepa.data.faithful import FaithfulDataset, cached_faithful_vocab
from textjepa.data.igsm.dataset import IGSMDataset
from textjepa.data.stylized_flat import StylizedFlatDataset, build_flat_stylized_vocab
from textjepa.models.flat_intent_jepa import FlatIntentJEPA
from textjepa.planning.flat_search import (
    CANDIDATE_INTERFACES, FlatPlanner, evaluate_flat_planning,
)
from textjepa.utils import seed_everything


def load_flat_run(ckpt_path: str, device: str = "cuda:0"):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    if ckpt.get("data_name", "faithful") == "stylized":
        vocab = build_flat_stylized_vocab(*ckpt["vocab_caps"])
    else:
        vocab = cached_faithful_vocab(*ckpt["vocab_caps"])
    mcfg = dict(cfg["model"])
    mcfg["init_from_lm"] = None  # weights come from the checkpoint itself
    mcfg["lm_detach_state"] = bool(
        dict(cfg.get("objective", {})).get("intent_prior_lm", {})
        .get("detach_state", False)
    )
    model = FlatIntentJEPA(vocab_size=len(vocab), pad_id=vocab.pad_id, **mcfg)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval(), vocab, cfg


def build_eval_dataset(cfg, vocab, size: int, seed: int, max_op=None,
                       max_edge=None, op_lo=None, op_hi=None,
                       nec_lo=None, nec_hi=None):
    """Evaluation problems in the checkpoint's data setting.

    faithful: caps/op_range (overridable for the OOD band).  stylized: the
    generator's own knobs, with ``op_lo/op_hi`` reinterpreted as the
    necessary-steps range so the same CLI drives both.
    """
    dc = cfg["data"]
    if dc.get("name", "faithful") == "stylized":
        steps = list(dc["steps_range"])
        base = IGSMDataset(
            vocab, size=size, seed=seed, modulus=dc["modulus"],
            n_vars_range=tuple(dc["n_vars_range"]), leaf_prob=dc["leaf_prob"],
            steps_range=(op_lo or steps[0], op_hi or steps[1]),
            distractor_prob=0.0, max_distractors=dc["max_distractors"],
            all_action_supervision=True,
        )
        return StylizedFlatDataset(base), {
            "generator": "stylized", "modulus": dc["modulus"],
            "n_vars_range": list(dc["n_vars_range"]),
            "steps_range": [op_lo or steps[0], op_hi or steps[1]],
        }
    mo = max_op or dc["max_op"]
    me = max_edge or dc["max_edge"]
    orange = (op_lo or dc["op_range"][0], op_hi or dc["op_range"][1])
    return FaithfulDataset(
        vocab, size=size, seed=seed, max_op=mo, max_edge=me,
        op_range=orange, distractor_prob=0.0,
        necessary_range=(nec_lo, nec_hi),
    ), {"generator": "faithful", "max_op": mo, "max_edge": me,
        "op_range": list(orange),
        "necessary_range": [nec_lo, nec_hi]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--interface", default="feasible_menu", choices=CANDIDATE_INTERFACES)
    ap.add_argument("--lookahead", type=int, default=1)
    ap.add_argument("--max-expand", type=int, default=64)
    ap.add_argument("--aggregate", default="mean_prefix",
                    choices=["endpoint", "mean_prefix", "movement"],
                    help="rollout score: endpoint energy only (legacy); mean energy over "
                         "imagined prefixes; or endpoint energy plus a geometric penalty "
                         "for steps whose imagined state barely moves (movement)")
    ap.add_argument("--expansion", default="beam", choices=["beam", "random"],
                    help="depth>1 continuations: energy-guided beam (current) or the "
                         "LEGACY uniform random tails")
    ap.add_argument("--no-beam-diagnostics", action="store_true",
                    help="skip the ORACLE beam-vs-depth-1 measurement (goal vector + 3 extra "
                         "scoring passes per step).  Plans are bit-identical, several times faster; "
                         "beam_closer_to_goal_than_d1_frac is then not reported")
    ap.add_argument("--root-agg", default="best", choices=["best", "mean"],
                help="first-action choice at depth>1: best = argmin rollout (historical); mean = argmin over MEAN energy of each root's surviving beams (variance reduction against the winner's curse)")
    ap.add_argument("--movement-weight", type=float, default=1.0,
                    help="aggregate=movement: weight of the no-movement penalty, in units "
                         "of the candidate-batch std of the endpoint energy")
    ap.add_argument("--scorer", default="energy",
                    choices=["energy", "oracle_distance", "symbolic_oracle", "context_distance"],
                    help="DIAGNOSTIC (candidate-privileged oracle rows, never a headline): "
                         "oracle_distance ranks by latent distance to the encoded TRUE solved "
                         "state (correct goal, LEARNED ruler); symbolic_oracle ranks by the "
                         "number of necessary-and-unresolved actions left after executing the "
                         "candidate in a CLONE of the env (correct goal, EXACT ruler) -- this is "
                         "the genuine upper bound on the search procedure")
    ap.add_argument("--distance-metric", default="raw", choices=["raw", "ln_l1", "cos"],
                    help="metric for scorer=oracle_distance.  raw = plain L2 "
                         "(historical; WRONG when endpoints are imagined, since "
                         "latent_pred only matches the encoder up to LayerNorm "
                         "and predicted states sit at ~2.2x the encoder norm). "
                         "ln_l1 = LN-L1, the space the predictor is trained in")
    ap.add_argument("--true-render-avg", type=int, default=1,
                    help="DIAGNOSTIC (endpoints=true): re-render each real endpoint K times "
                         "with perturbed global RNG and average the encodings, knocking out "
                         "rendering noise from the distance ruler")
    ap.add_argument("--goal-set-samples", type=int, default=0,
                    help="DIAGNOSTIC (scorer=oracle_distance): K>0 builds, per goal "
                         "computation, K alternative valid terminal states (random "
                         "topological orders of the remaining necessary actions, "
                         "executed + canonically rendered in env clones and encoded) "
                         "in addition to the reference terminal; the oracle distance "
                         "becomes the MIN over this goal set.  0 = single reference "
                         "goal (historical behavior, byte-identical)")
    ap.add_argument("--endpoints", default="imagined", choices=["imagined", "true"],
                    help="DIAGNOSTIC: true = execute candidates in a copy of the env and encode the REAL state")
    ap.add_argument("--branch", type=int, default=4,
                    help="energy-guided continuations kept per beam at depth>1")
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--episode-start", type=int, default=0,
                    help="shard offset: run episodes [start, start+n) of the same deterministic set")
    ap.add_argument("--cap-mult", type=float, default=4.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=2, help="problem seed (2 = val)")
    ap.add_argument("--max-op", type=int, default=None)
    ap.add_argument("--max-edge", type=int, default=None)
    ap.add_argument("--op-lo", type=int, default=None)
    ap.add_argument("--op-hi", type=int, default=None)
    ap.add_argument("--nec-lo", type=int, default=None,
                    help="reject problems whose solution needs fewer than this many steps "
                         "(faithful generator only)")
    ap.add_argument("--nec-hi", type=int, default=None,
                    help="reject problems whose solution needs more than this many steps; "
                         "together with --op-lo/--op-hi this controls how many catalogue "
                         "actions are USELESS (not on the solution path)")
    ap.add_argument("--prior-samples", type=int, default=16)
    ap.add_argument("--prior-top-p", type=float, default=0.95)
    ap.add_argument("--prior-temperature", type=float, default=1.3)
    ap.add_argument("--prior-top-k", type=int, default=0)
    ap.add_argument("--codebook-k", type=int, default=64)
    ap.add_argument("--codebook-problems", type=int, default=64)
    # OPT-IN learned action density (scripts/train_flow_prior.py).  Default
    # None -> every existing interface behaves exactly as before.
    ap.add_argument("--flow-prior", default=None,
                    help="path to a fitted ConditionalFlowPrior (flow_rerank / flow_decode)")
    ap.add_argument("--flow-oversample", type=int, default=64,
                    help="flow_rerank: phrases decoded before the density picks --prior-samples of them")
    ap.add_argument("--flow-temperature", type=float, default=1.0)
    ap.add_argument("--flow-no-diversity", action="store_true",
                    help="flow_rerank ablation: rank by density alone, no max-min spread term")
    # OPT-IN target architecture: learned discrete action prior + DETACHED
    # context-conditioned decoder.  Default None -> every existing interface
    # behaves exactly as before.
    ap.add_argument("--code-prior", default=None,
                    help="path to a fitted CodePrior (scripts/train_code_prior.py)")
    ap.add_argument("--action-decoder", default=None,
                    help="path to a trained ContextActionDecoder "
                         "(scripts/train_action_decoder.py)")
    ap.add_argument("--code-prior-temperature", type=float, default=1.0)
    ap.add_argument("--code-prior-max-ctx", type=int, default=768,
                    help="must match train_action_decoder.py --max-ctx")
    ap.add_argument("--code-prior-sample", action="store_true",
                    help="sample codes from p(code | state) instead of top-K")
    ap.add_argument("--generate-outcomes", dest="generate_outcomes",
                    action="store_true", default=None,
                    help="model writes the outcome sentence (and hence the "
                         "answer) instead of the environment; implied by "
                         "--interface autonomous")
    ap.add_argument("--no-answer-emission", action="store_true",
                    help="skip the answer-emission success criterion (2026-08-21): by "
                         "default, at the solving step the model's token head generates "
                         "the final outcome sentence itself (before the env's rendering "
                         "enters the context); success_answer requires that generation "
                         "terminates by the model's own choice ('.'-final token) AND its "
                         "final integer equals the true answer.  Applied identically to "
                         "the random / first-feasible reference rows")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    model, vocab, cfg = load_flat_run(args.ckpt, str(device))
    flow_prior = None
    if args.flow_prior is not None:
        from textjepa.planning.flow_prior import ConditionalFlowPrior
        flow_prior = ConditionalFlowPrior.load(args.flow_prior, map_location=device).to(device)
    code_prior = action_decoder = None
    if args.code_prior is not None or args.action_decoder is not None:
        if not (args.code_prior and args.action_decoder):
            raise SystemExit(
                "--code-prior and --action-decoder must be given together")
        from textjepa.planning.action_decoder import ContextActionDecoder
        from textjepa.planning.code_prior import CodePrior
        code_prior = CodePrior.load(
            args.code_prior, map_location=device).to(device).eval()
        action_decoder = ContextActionDecoder.load(
            args.action_decoder, map_location=device).to(device).eval()
    dc = cfg["data"]
    dataset, caps = build_eval_dataset(
        cfg, vocab, args.episode_start + args.n_episodes, args.split_seed, args.max_op,
        args.max_edge, args.op_lo, args.op_hi, args.nec_lo, args.nec_hi,
    )
    planner = FlatPlanner(
        model, vocab, device, lookahead=args.lookahead, max_expand=args.max_expand, branch=args.branch, aggregate=args.aggregate,
        expansion=args.expansion, movement_weight=args.movement_weight,
        root_agg=args.root_agg,
        beam_diagnostics=not args.no_beam_diagnostics, scorer=args.scorer,
        distance_metric=args.distance_metric, endpoints=args.endpoints,
        true_render_avg=args.true_render_avg,
        goal_set_samples=args.goal_set_samples,
        candidate_interface=args.interface, cap_mult=args.cap_mult,
        prior_samples=args.prior_samples, prior_top_p=args.prior_top_p,
        prior_temperature=args.prior_temperature, prior_top_k=args.prior_top_k,
        codebook_k=args.codebook_k, max_len=int(cfg["model"]["max_len"]),
        flow_prior=flow_prior, flow_oversample=args.flow_oversample,
        flow_temperature=args.flow_temperature,
        flow_diversity=not args.flow_no_diversity,
        answer_emission=not args.no_answer_emission,
        code_prior=code_prior, action_decoder=action_decoder,
        code_prior_temperature=args.code_prior_temperature,
        code_prior_sample=args.code_prior_sample,
        code_prior_max_ctx=args.code_prior_max_ctx,
        generate_outcomes=args.generate_outcomes,
    )
    if args.interface == "codebook_ground":
        train_ds, _ = build_eval_dataset(
            cfg, vocab, args.codebook_problems, dc["train_seed"],
        )
        planner.fit_action_prior([train_ds.problem(i)[0] for i in range(args.codebook_problems)])
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if args.precision == "bf16" and device.type == "cuda" else torch.autocast("cpu", enabled=False))
    with torch.no_grad(), ctx:
        results = evaluate_flat_planning(planner, dataset, args.n_episodes, seed=args.seed,
                                         episode_start=args.episode_start)
    results["protocol"] = {
        "ckpt": args.ckpt, "interface": args.interface, "lookahead": args.lookahead,
        "episode_start": args.episode_start,
        "distance_metric": args.distance_metric,
        "max_expand": args.max_expand, "branch": args.branch,
        "root_agg": args.root_agg,
        "cap_mult": args.cap_mult,
        "flow_prior": args.flow_prior,
        "code_prior": args.code_prior,
        "action_decoder": args.action_decoder,
        "code_prior_temperature": (args.code_prior_temperature
                                   if args.code_prior else None),
        "code_prior_sample": bool(args.code_prior_sample),
        "generate_outcomes": planner.generate_outcomes,
        "flow_oversample": args.flow_oversample if args.flow_prior else None,
        "flow_diversity": (not args.flow_no_diversity) if args.flow_prior else None,
        "aggregate": args.aggregate, "scorer": args.scorer, "endpoints": args.endpoints,
        "true_render_avg": args.true_render_avg,
        "goal_set_samples": args.goal_set_samples,
        "movement_weight": args.movement_weight if args.aggregate == "movement" else None,
        "beam_diagnostics": not args.no_beam_diagnostics,
        "expansion": (
            "LEGACY uniform random tails (oracle-free)" if args.expansion == "random"
            else "energy-guided beam (oracle-free)" if not planner.oracle_diagnostic
            else "beam guided by an ORACLE diagnostic scorer/state-source"),
        "caps": caps,
        "oracle_future_actions": False, "budget": "none (runaway cap only)",
        "answer_emission": not args.no_answer_emission,
        "success_criterion": (
            "success_answer: env solved within cap AND the model itself emitted "
            "the final outcome sentence (greedy decode from history ending in the "
            "solving intent phrase; env rendering not in context), properly "
            "terminated ('.'-final token, not token-cap exhaustion), with final "
            "integer == true answer.  success_env is the old env-side criterion."
            if not args.no_answer_emission else "success_env only (legacy)"),
        "evidence_label": ("CANDIDATE-PRIVILEGED ORACLE DIAGNOSTIC "
            f"(scorer={args.scorer}, endpoints={args.endpoints}) -- NOT a headline row; "
            if planner.oracle_diagnostic else "") + (
            "menu (feasibility oracle at the root only)" if args.interface == "feasible_menu"
            else "menu-free, oracle executor" if args.interface != "autonomous"
            else "menu-free, model-written outcomes, env used for grading only"
        ),
    }
    c = results["compute"]["per_episode"]
    print("compute/episode  " + "  ".join(
        f"{k}={v:.1f}" for k, v in c.items()))
    for name in ("latent_planner", "random_policy", "first_feasible_policy"):
        m = results[name]
        print(f"{name:22s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, float)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
