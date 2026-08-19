"""DELIVERABLE 2 -- linear + MLP probe battery with random-init controls.

What is probed (all on FROZEN representations, held-out problems, stylized
iGSM where every label is available from the symbolic oracle):

  ``resolved``      is variable v already computed at step t?      (binary)
  ``feasible``      is candidate c legal at step t?                (binary)
  ``remaining``     how many necessary steps are left?  **DIAGNOSTIC ONLY**
                    -- per CLAUDE.md a steps-to-go regressor must never become
                    a component of the system; it is measured here purely to
                    characterise the representation.
  ``value``         the computed value of the step that is taken   (regression)
  ``operator``      the operator of the step, read from the STATE
                    DISPLACEMENT s_{t+1} - s_t                     (4-class)

Controls that make the numbers mean something:
  * a RANDOM-INIT encoder of the identical architecture (the "collusion"
    control: a trainable head on random features is often already strong, so
    a probe number without this row is uninterpretable);
  * a shuffled-state control (state taken from an unrelated row);
  * the majority-class / target-variance floors;
  * per-DEPENDENCY-DEPTH curves for every probe, which is what the
    2026-08-12 LM state-readout report compared across families.

Evidence label: every target here is ORACLE / CANDIDATE-PRIVILEGED. None of
these probes is a planning result and none of them is a system component.

Usage
-----
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" .venv/bin/python \
        scripts/analysis/probe_battery.py --n-train 1500 --n-val 600 \
        --model jepa=<ckpt> --model token_lm=<ckpt> --model sent_lm=<ckpt> \
        --random-init jepa=<ckpt> --out probes.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.adapters import load_adapter                       # noqa: E402
from analysis.common import (                                    # noqa: E402
    binary_summary, multiclass_summary, regression_summary, standardize,
    train_binary_probe, train_multiclass_probe, train_regression_probe,
)
from analysis.consequence_geometry import parse_model_arg        # noqa: E402

torch.set_num_threads(min(8, torch.get_num_threads()))

OP_INDEX = {"const": 0, "add": 1, "sub": 2, "mul": 3}


def var_depths(problem) -> list[int]:
    """Dependency depth of every variable (leaves = 0)."""
    depth = [0] * len(problem.vars)
    for v in problem.vars:  # topologically ordered: parents have smaller idx
        depth[v.idx] = 0 if v.is_leaf else 1 + max(depth[p] for p in v.parents)
    return depth


# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract(adapter, dataset, n: int) -> list[dict]:
    """One record per problem: frozen features + oracle structure."""
    from textjepa.data.igsm.render import (
        action_phrase, catalogue_phrases, prompt_sentences, step_sentence,
    )
    import random
    records = []
    for i in range(n):
        problem, _ = dataset.problem(i)
        item = dataset[i]
        trace = list(item["var_idx"])
        prompt = prompt_sentences(problem, random.Random(f"prompt:{i}"))
        steps = [step_sentence(problem, v) for v in trace]
        states = adapter.encode_states(prompt, steps)          # [T+1, D]
        phrases = catalogue_phrases(problem)
        u = adapter.encode_actions_free(phrases)
        if u is None:
            u = adapter.encode_actions_ctx(prompt, [], phrases)
        records.append(dict(
            s=states[:-1].clone(),                 # state BEFORE step t
            s_next=states[1:].clone(),             # state AFTER step t
            u=u.clone(),
            trace=trace,
            parents=[tuple(v.parents) for v in problem.vars],
            depth=var_depths(problem),
            ops=[OP_INDEX[v.op] for v in problem.vars],
            values=[int(x) for x in problem.values],
            ancestors=sorted(problem.query_ancestors),
            n_vars=len(problem.vars),
        ))
    return records


# --------------------------------------------------------------------------- #
def rows_pairwise(records):
    """(problem, step t, variable v) -> resolvedness / feasibility."""
    S, U, R, F, D = [], [], [], [], []
    for r in records:
        done: set[int] = set()
        for t in range(len(r["trace"])):
            s = r["s"][t]
            for v in range(r["n_vars"]):
                S.append(s); U.append(r["u"][v])
                is_done = v in done
                R.append(float(is_done))
                F.append(float((not is_done)
                               and all(p in done for p in r["parents"][v])))
                D.append(r["depth"][v])
            done.add(r["trace"][t])
    return (torch.stack(S), torch.stack(U), torch.tensor(R),
            torch.tensor(F), torch.tensor(D))


def rows_stepwise(records):
    """(problem, step t) -> remaining necessary steps / value / operator."""
    S, DS, U, REM, VAL, OP, D = [], [], [], [], [], [], []
    for r in records:
        done: set[int] = set()
        anc = set(r["ancestors"])
        for t, a in enumerate(r["trace"]):
            S.append(r["s"][t])
            DS.append(r["s_next"][t] - r["s"][t])
            U.append(r["u"][a])
            REM.append(float(len(anc - done)))
            VAL.append(float(r["values"][a]))
            OP.append(r["ops"][a])
            D.append(r["depth"][a])
            done.add(a)
    return (torch.stack(S), torch.stack(DS), torch.stack(U),
            torch.tensor(REM), torch.tensor(VAL),
            torch.tensor(OP, dtype=torch.long), torch.tensor(D))


# --------------------------------------------------------------------------- #
def run_probes(train_rec, val_rec, seed: int, subsample: int = 120_000) -> dict:
    out: dict = {}
    g = torch.Generator().manual_seed(seed)

    def cut(*tensors):
        n = len(tensors[0])
        if n <= subsample:
            return tensors
        idx = torch.randperm(n, generator=g)[:subsample]
        return tuple(t[idx] for t in tensors)

    # ---- pairwise: resolvedness + feasibility ---------------------------- #
    Str, Utr, Rtr, Ftr, Dtr = cut(*rows_pairwise(train_rec))
    Sva, Uva, Rva, Fva, Dva = cut(*rows_pairwise(val_rec))
    d_s = Str.shape[1]
    Xtr, Xva = standardize(torch.cat([Str, Utr], 1), torch.cat([Sva, Uva], 1))
    perm_tr = torch.randperm(len(Xtr), generator=g)
    perm_va = torch.randperm(len(Xva), generator=g)
    Xtr_sh = torch.cat([Xtr[perm_tr, :d_s], Xtr[:, d_s:]], 1)
    Xva_sh = torch.cat([Xva[perm_va, :d_s], Xva[:, d_s:]], 1)
    for task, ytr, yva in (("resolved", Rtr, Rva), ("feasible", Ftr, Fva)):
        block = {"n_train": len(Xtr), "n_val": len(Xva), "d_in": Xtr.shape[1]}
        for kind in ("linear", "mlp"):
            block[kind] = binary_summary(
                train_binary_probe(kind, Xtr, ytr, Xva, yva, seed), yva, Dva
            )
        block["mlp_shuffled_state"] = binary_summary(
            train_binary_probe("mlp", Xtr_sh, ytr, Xva_sh, yva, seed), yva, Dva
        )
        out[task] = block

    # ---- stepwise -------------------------------------------------------- #
    Str2, DStr, Utr2, REMtr, VALtr, OPtr, Dtr2 = rows_stepwise(train_rec)
    Sva2, DSva, Uva2, REMva, VALva, OPva, Dva2 = rows_stepwise(val_rec)

    # remaining steps to goal -- DIAGNOSTIC ONLY (never a component)
    A, B = standardize(Str2, Sva2)
    blk = {"n_train": len(A), "n_val": len(B),
           "LABEL": "DIAGNOSTIC ONLY -- oracle steps-to-go; per CLAUDE.md this "
                    "must never become a system component."}
    for kind in ("linear", "mlp"):
        blk[kind] = regression_summary(
            train_regression_probe(kind, A, REMtr, B, REMva, seed), REMva, Dva2
        )
    out["remaining_steps_DIAGNOSTIC"] = blk

    # computed value of the step taken, from (state, action)
    A, B = standardize(torch.cat([Str2, Utr2], 1), torch.cat([Sva2, Uva2], 1))
    blk = {"n_train": len(A), "n_val": len(B)}
    for kind in ("linear", "mlp"):
        blk[kind] = regression_summary(
            train_regression_probe(kind, A, VALtr, B, VALva, seed), VALva, Dva2
        )
    out["step_value"] = blk

    # operator identity from the STATE DISPLACEMENT alone
    A, B = standardize(DStr, DSva)
    blk = {"n_train": len(A), "n_val": len(B),
           "classes": {k: v for k, v in OP_INDEX.items()}}
    for kind in ("linear", "mlp"):
        blk[kind] = multiclass_summary(
            train_multiclass_probe(kind, A, OPtr, B, OPva, 4, seed), OPva, Dva2
        )
    out["operator_from_displacement"] = blk
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--random-init", action="append", default=[])
    ap.add_argument("--flat-lm-init", default=None)
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = {
        "n_train_problems": args.n_train, "n_val_problems": args.n_val,
        "seed": args.seed, "domain": "stylized_igsm",
        "evidence_label": (
            "ORACLE / CANDIDATE-PRIVILEGED DIAGNOSTIC. No probe here is a "
            "planning result and none may become a system component; the "
            "steps-to-go probe in particular is measurement only."
        ),
        "models": {},
    }
    arms = [(*parse_model_arg(a), False, None) for a in args.model]
    arms += [(s, p, f"{n}__random_init", True, None)
             for s, p, n in map(parse_model_arg, args.random_init)]
    if args.flat_lm_init:
        ck, lm = args.flat_lm_init.split("=", 1)
        arms.append(("flat_jepa", ck, "flat_jepa__lm_init_untrained", False, lm))

    for spec, path, name, rand, lm_init in arms:
        t0 = time.time()
        print(f"[{name}] loading ...", flush=True)
        run = load_adapter(spec, path, name=name, random_init=rand,
                           flat_lm_init=lm_init)
        from textjepa.utils.checkpoint import build_dataset
        tr = build_dataset(run.cfg, run.vocab, split="train", size=args.n_train)
        va = build_dataset(run.cfg, run.vocab, split="val", size=args.n_val)
        print(f"[{name}] extracting features ...", flush=True)
        train_rec = extract(run.adapter, tr, args.n_train)
        val_rec = extract(run.adapter, va, args.n_val)
        print(f"[{name}] probing ...", flush=True)
        block = {"info": run.adapter.info(), "ckpt": path, "spec": spec,
                 "probes": run_probes(train_rec, val_rec, args.seed)}
        block["seconds"] = round(time.time() - t0, 1)
        results["models"][name] = block
        for task, b in block["probes"].items():
            head = b.get("mlp", {})
            key = "auc" if "auc" in head else ("r2" if "r2" in head else "acc")
            print(f"  {task:32s} mlp {key}={head.get(key):.3f}", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
