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
    model = FlatIntentJEPA(vocab_size=len(vocab), pad_id=vocab.pad_id, **mcfg)
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval(), vocab, cfg


def build_eval_dataset(cfg, vocab, size: int, seed: int, max_op=None,
                       max_edge=None, op_lo=None, op_hi=None):
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
    ), {"generator": "faithful", "max_op": mo, "max_edge": me,
        "op_range": list(orange)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--interface", default="feasible_menu", choices=CANDIDATE_INTERFACES)
    ap.add_argument("--lookahead", type=int, default=1)
    ap.add_argument("--max-expand", type=int, default=64)
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--cap-mult", type=float, default=4.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=2, help="problem seed (2 = val)")
    ap.add_argument("--max-op", type=int, default=None)
    ap.add_argument("--max-edge", type=int, default=None)
    ap.add_argument("--op-lo", type=int, default=None)
    ap.add_argument("--op-hi", type=int, default=None)
    ap.add_argument("--prior-samples", type=int, default=16)
    ap.add_argument("--prior-top-p", type=float, default=0.95)
    ap.add_argument("--prior-temperature", type=float, default=1.3)
    ap.add_argument("--prior-top-k", type=int, default=0)
    ap.add_argument("--codebook-k", type=int, default=64)
    ap.add_argument("--codebook-problems", type=int, default=64)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    model, vocab, cfg = load_flat_run(args.ckpt, str(device))
    dc = cfg["data"]
    dataset, caps = build_eval_dataset(
        cfg, vocab, args.n_episodes, args.split_seed, args.max_op,
        args.max_edge, args.op_lo, args.op_hi,
    )
    planner = FlatPlanner(
        model, vocab, device, lookahead=args.lookahead, max_expand=args.max_expand,
        candidate_interface=args.interface, cap_mult=args.cap_mult,
        prior_samples=args.prior_samples, prior_top_p=args.prior_top_p,
        prior_temperature=args.prior_temperature, prior_top_k=args.prior_top_k,
        codebook_k=args.codebook_k, max_len=int(cfg["model"]["max_len"]),
    )
    if args.interface == "codebook_ground":
        train_ds, _ = build_eval_dataset(
            cfg, vocab, args.codebook_problems, dc["train_seed"],
        )
        planner.fit_action_prior([train_ds.problem(i)[0] for i in range(args.codebook_problems)])
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if args.precision == "bf16" and device.type == "cuda" else torch.autocast("cpu", enabled=False))
    with torch.no_grad(), ctx:
        results = evaluate_flat_planning(planner, dataset, args.n_episodes, seed=args.seed)
    results["protocol"] = {
        "ckpt": args.ckpt, "interface": args.interface, "lookahead": args.lookahead,
        "max_expand": args.max_expand, "cap_mult": args.cap_mult,
        "caps": caps,
        "oracle_future_actions": False, "budget": "none (runaway cap only)",
        "evidence_label": (
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
