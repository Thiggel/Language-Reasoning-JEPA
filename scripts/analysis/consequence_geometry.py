"""DELIVERABLE 1 -- consequence geometry of frozen representations.

Question
--------
Does the JEPA's predictor target make representations organise by CONSEQUENCE?
Concretely, on held-out problems:

  (a) PARAPHRASE COLLAPSE -- two intent phrases with identical consequences
      (stylized iGSM: commuted operands of a commutative operator; ProofWriter:
      reordered premises of one rule application) should sit close together,
      closer than a matched control pair with a different consequence.
  (b) NEGATION / POLARITY SEPARATION -- a statement and its negation
      (ProofWriter polarity flip: a ONE-token edit) should sit far apart.
      In stylized iGSM, which has no negation, the nearest available analogue
      is an operator substitution; it is reported as ``operator_flip`` and
      must never be called negation.
  (c) The same metrics on the token LM and the sentence LM at matched
      positions, and -- where a FlatIntentJEPA is given together with the token
      LM it was initialised from -- LM-init-BEFORE-training vs after-training,
      which isolates what JEPA training adds to the SAME weights.

Everything is read-only on frozen checkpoints; all phrases come from the
environments' own renderers (see ``scripts/analysis/pairs.py``).

Evidence labels
---------------
The consequence labels ("these two phrases mean the same thing") are ORACLE
facts about the environment.  They are used to LABEL pairs, never as a model
input, and no planning claim is made here.  The metric is unsupervised: only
distances between frozen vectors are compared.

Usage
-----
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" .venv/bin/python \
        scripts/analysis/consequence_geometry.py \
        --domain stylized --n-problems 300 \
        --model jepa=<ckpt> --model token_lm=<ckpt> --random-init jepa=<ckpt> \
        --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import pairs as P                     # noqa: E402
from analysis.adapters import load_adapter          # noqa: E402
from analysis.common import DISTANCES, separation   # noqa: E402

torch.set_num_threads(min(8, torch.get_num_threads()))


# --------------------------------------------------------------------------- #
def _evaluate_contexts(adapter, contexts: list[P.ContextPairs],
                       spaces: tuple[str, ...] = ("action_ctx", "action_free",
                                                  "predicted_next")) -> dict:
    """distance samples[space][distance][subkind] -> list[float]."""
    samples: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for ctx in contexts:
        phrases = ctx.phrases
        reps: dict[str, torch.Tensor | None] = {}
        if "action_ctx" in spaces:
            reps["action_ctx"] = adapter.encode_actions_ctx(
                ctx.prompt, ctx.steps, phrases
            )
        if "action_free" in spaces:
            reps["action_free"] = adapter.encode_actions_free(phrases)
        if "predicted_next" in spaces:
            code = reps.get("action_free")
            if code is None or code.shape[-1] != adapter.d_action:
                code = reps.get("action_ctx")
            state = adapter.encode_states(ctx.prompt, ctx.steps)[-1]
            pred = None
            if code is not None:
                pred = adapter.predict(
                    state.unsqueeze(0).expand(len(phrases), -1), code
                )
            reps["predicted_next"] = pred
        for i, j, spec in ctx.index_pairs():
            for space, R in reps.items():
                if R is None:
                    continue
                for dname, dfn in DISTANCES.items():
                    d = float(dfn(R[i:i + 1], R[j:j + 1])[0])
                    samples[space][dname][spec.subkind].append(d)
    return samples


def _summarize(samples: dict, contexts: list[P.ContextPairs]) -> dict:
    """Same-consequence vs different-consequence separation, per space."""
    relation = {}
    for ctx in contexts:
        for p in ctx.pairs:
            relation[p.subkind] = p.relation
    same_kinds = [k for k, r in relation.items() if r == "same_consequence"]
    diff_kinds = [k for k, r in relation.items() if r == "different_consequence"]
    out: dict = {}
    for space, per_dist in samples.items():
        out[space] = {}
        for dname, per_kind in per_dist.items():
            block = {
                "per_subkind": {
                    k: {"n": len(v),
                        "mean": float(torch.tensor(v).mean()) if v else float("nan"),
                        "median": float(torch.tensor(v).median()) if v else float("nan")}
                    for k, v in per_kind.items()
                }
            }
            same = torch.tensor(
                [x for k in same_kinds for x in per_kind.get(k, [])]
            )
            for dk in diff_kinds:
                diff = torch.tensor(per_kind.get(dk, []))
                if len(same) and len(diff):
                    block[f"same_vs_{dk}"] = separation(same, diff).to_dict()
            alldiff = torch.tensor(
                [x for k in diff_kinds for x in per_kind.get(k, [])]
            )
            if len(same) and len(alldiff):
                block["same_vs_all_different"] = separation(same, alldiff).to_dict()
            out[space][dname] = block
    return out


def _surface_baseline(contexts: list[P.ContextPairs]) -> dict:
    """Token edit distance as a pure-surface reference.

    If the representation merely tracked surface form, its separation numbers
    would match this row.  Reporting it is what makes the geometry claim
    falsifiable.
    """
    per_kind: dict[str, list[float]] = defaultdict(list)
    relation = {}
    for ctx in contexts:
        for p in ctx.pairs:
            per_kind[p.subkind].append(float(p.edit_distance))
            relation[p.subkind] = p.relation
    same = torch.tensor([x for k, v in per_kind.items()
                         if relation[k] == "same_consequence" for x in v])
    diff = torch.tensor([x for k, v in per_kind.items()
                         if relation[k] == "different_consequence" for x in v])
    out = {"per_subkind": {k: {"n": len(v), "mean": float(torch.tensor(v).mean())}
                           for k, v in per_kind.items()}}
    if len(same) and len(diff):
        out["same_vs_all_different"] = separation(same, diff).to_dict()
    return out


# --------------------------------------------------------------------------- #
def _state_commutation(adapter, cases: list[dict]) -> dict:
    """Order-invariance of states (both iGSM domains).

    same: A-then-B vs B-then-A  (identical resolved set, different text)
    diff: A-then-B vs A-then-C  (different resolved set)
    """
    out: dict = {d: {"same": [], "diff": []} for d in DISTANCES}
    for c in cases:
        prompt, pre = c["prompt"], c["prefix"]
        def last(steps):
            return adapter.encode_states(prompt, pre + steps)[-1:]
        ab, ba, ac = last(c["ab"]), last(c["ba"]), last(c["ac"])
        for dname, dfn in DISTANCES.items():
            out[dname]["same"].append(float(dfn(ab, ba)[0]))
            out[dname]["diff"].append(float(dfn(ab, ac)[0]))
    return {
        d: separation(torch.tensor(v["same"]), torch.tensor(v["diff"])).to_dict()
        for d, v in out.items() if v["same"]
    }


# --------------------------------------------------------------------------- #
def build_domain(domain: str, cfg, vocab, n: int, seed: int, split: str):
    """Returns (contexts, commutation_cases, notes)."""
    notes: dict = {}
    if domain == "stylized":
        from textjepa.utils.checkpoint import build_dataset
        ds = build_dataset(cfg, vocab, split=split, size=n)
        return (P.stylized_contexts(ds, n, seed),
                P.stylized_state_commutation(ds, min(n, 200), seed), notes)
    if domain == "faithful":
        # cfg is an OmegaConf node for DiscourseJEPA/LM runs and a plain dict
        # for FlatIntentJEPA runs; go through the dataset directly so both work.
        from textjepa.data.faithful import FaithfulDataset
        dc = cfg["data"] if isinstance(cfg, dict) else cfg.data
        get = (lambda k, d=None: dc.get(k, d))
        ds = FaithfulDataset(
            vocab, size=n, seed=int(get("val_seed", 2) if split == "val"
                                    else get("test_seed", 3)),
            max_op=int(get("max_op", 15)), max_edge=int(get("max_edge", 20)),
            op_range=tuple(get("op_range", (3, 15))), distractor_prob=0.0,
        )
        notes["action_paraphrase"] = P.FAITHFUL_ACTION_PARAPHRASE_NOTE
        return ([], P.faithful_state_commutation(ds, min(n, 200), seed), notes)
    if domain == "proofwriter":
        from textjepa.data.observed_action import load_observed_action_jsonl
        dc = cfg["data"] if isinstance(cfg, dict) else cfg.data
        path = dc["val_path"] if split == "val" else dc["test_path"]
        eps = load_observed_action_jsonl(path, expected_domain="proofwriter")
        return P.proofwriter_contexts(eps, n, seed), [], notes
    raise ValueError(f"unknown domain {domain!r}")


def parse_model_arg(arg: str) -> tuple[str, str, str]:
    """``spec=path`` or ``spec=path=name``."""
    parts = arg.split("=")
    if len(parts) == 2:
        spec, path = parts
        return spec, path, f"{spec}:{Path(path).parents[1].name}"
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(f"bad --model {arg!r} (expected spec=path[=name])")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True,
                    choices=["stylized", "faithful", "proofwriter"])
    ap.add_argument("--model", action="append", default=[],
                    help="spec=ckpt[=name]; spec in jepa|flat_jepa|token_lm|sent_lm")
    ap.add_argument("--random-init", action="append", default=[],
                    help="same syntax; architecture rebuilt with fresh weights")
    ap.add_argument("--flat-lm-init", default=None,
                    help="flat_jepa ckpt=lm_ckpt: adds an LM-init-before-training arm")
    ap.add_argument("--n-problems", type=int, default=200)
    ap.add_argument("--split", default="val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results: dict = {
        "domain": args.domain, "n_problems": args.n_problems,
        "split": args.split, "seed": args.seed,
        "evidence_label": (
            "ORACLE-LABELLED DIAGNOSTIC: consequence equality is an environment "
            "fact used only to label pairs; it is never a model input. Not a "
            "planning result."
        ),
        "models": {},
    }
    arms: list[tuple[str, str, str, bool, str | None]] = []
    for a in args.model:
        spec, path, name = parse_model_arg(a)
        arms.append((spec, path, name, False, None))
    for a in args.random_init:
        spec, path, name = parse_model_arg(a)
        arms.append((spec, path, f"{name}__random_init", True, None))
    if args.flat_lm_init:
        ck, lm = args.flat_lm_init.split("=", 1)
        arms.append(("flat_jepa", ck, "flat_jepa__lm_init_untrained", False, lm))

    cached_domain = None
    for spec, path, name, rand, lm_init in arms:
        t0 = time.time()
        print(f"[{name}] loading {spec} from {path} ...", flush=True)
        run = load_adapter(spec, path, name=name, random_init=rand,
                           flat_lm_init=lm_init)
        ad = run.adapter
        if cached_domain is None:
            contexts, commutation, notes = build_domain(
                args.domain, run.cfg, run.vocab, args.n_problems,
                args.seed, args.split,
            )
            cached_domain = (contexts, commutation, notes)
            results["notes"] = notes
            results["n_contexts"] = len(contexts)
            results["n_pairs"] = sum(len(c.pairs) for c in contexts)
            results["n_commutation_cases"] = len(commutation)
            results["pair_list"] = [
                {"context": c.problem_id, "prompt_len": len(c.prompt),
                 "n_steps": len(c.steps), **p.to_dict()}
                for c in contexts for p in c.pairs
            ]
            if contexts:
                results["surface_edit_distance_baseline"] = _surface_baseline(contexts)
        contexts, commutation, _ = cached_domain
        block: dict = {"info": ad.info(), "ckpt": path, "spec": spec}
        if contexts:
            samples = _evaluate_contexts(ad, contexts)
            block["pair_geometry"] = _summarize(samples, contexts)
        if commutation:
            block["state_order_invariance"] = _state_commutation(ad, commutation)
        block["seconds"] = round(time.time() - t0, 1)
        results["models"][name] = block
        print(f"[{name}] done in {block['seconds']}s", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
