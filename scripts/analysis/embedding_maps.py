"""DELIVERABLE 3 -- t-SNE / UMAP maps of action and state representations.

Two stages, because the training venv must never import matplotlib:

  ``--export``  (run with ``.venv/bin/python``)  encodes held-out stylized
      iGSM problems with a frozen checkpoint and writes an NPZ of action and
      state vectors plus every colouring label:
        (i) operator type, (ii) dependency depth, (iii) feasible/infeasible,
        (iv) paraphrase-pair membership (commuted-operand partners share an id).
  ``--plot``    (run with ``.venv2/bin/python``)  reads those NPZs, fits t-SNE
      and UMAP, writes PNG + PDF per (checkpoint, space, colouring) and the
      2-D coordinates as NPZ + JSON so every figure is reproducible without
      re-running the embedding.

Usage
-----
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" .venv/bin/python \
        scripts/analysis/embedding_maps.py --export \
        --model jepa=<ckpt> --n-problems 250 --out-dir <dir>/data

    OMP_NUM_THREADS=8 .venv2/bin/python scripts/analysis/embedding_maps.py \
        --plot --in-dir <dir>/data --fig-dir <dir>/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LABEL_FIELDS = ("operator", "depth", "feasible", "paraphrase_group")
OP_NAMES = ["const", "add", "sub", "mul"]


# --------------------------------------------------------------------------- #
# stage 1: export (torch venv)
# --------------------------------------------------------------------------- #
def export(args) -> None:
    import random

    import numpy as np
    import torch

    from analysis.adapters import load_adapter
    from analysis.consequence_geometry import parse_model_arg
    from analysis.pairs import _styl_phrase, _COMMUTATIVE
    from analysis.probe_battery import OP_INDEX, var_depths

    from textjepa.data.igsm.env import SymbolicEnv
    from textjepa.data.igsm.render import (
        catalogue_phrases, prompt_sentences, step_sentence,
    )
    from textjepa.utils.checkpoint import build_dataset

    torch.set_num_threads(min(8, torch.get_num_threads()))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = [(*parse_model_arg(a), False, None) for a in args.model]
    arms += [(s, p, f"{n}__random_init", True, None)
             for s, p, n in map(parse_model_arg, args.random_init)]
    if args.flat_lm_init:
        ck, lm = args.flat_lm_init.split("=", 1)
        arms.append(("flat_jepa", ck, "flat_jepa__lm_init_untrained", False, lm))

    for spec, path, name, rand, lm_init in arms:
        run = load_adapter(spec, path, name=name, random_init=rand,
                           flat_lm_init=lm_init)
        ad = run.adapter
        ds = build_dataset(run.cfg, run.vocab, split="val", size=args.n_problems)
        A, S = [], []
        lab = {k: [] for k in LABEL_FIELDS}
        slab = {"depth": [], "step_index": [], "remaining": []}
        group = 0
        with torch.no_grad():
            for i in range(args.n_problems):
                problem, _ = ds.problem(i)
                item = ds[i]
                trace = list(item["var_idx"])
                prompt = prompt_sentences(problem, random.Random(f"prompt:{i}"))
                depth = var_depths(problem)
                env = SymbolicEnv(problem)
                t = min(len(trace) - 1, i % max(1, len(trace)))
                steps = [step_sentence(problem, trace[k]) for k in range(t)]
                for k in range(t):
                    env.step(trace[k])
                feasible = set(env.feasible_actions())
                # --- actions: the full catalogue + commuted paraphrases ---
                phrases = list(catalogue_phrases(problem))
                meta = [(v.idx, -1) for v in problem.vars]
                for v in problem.vars:
                    if v.is_leaf or v.op not in _COMMUTATIVE:
                        continue
                    a, b = v.parents
                    if a == b:
                        continue
                    group += 1
                    meta[v.idx] = (v.idx, group)
                    phrases.append(_styl_phrase(problem, v.idx, b, a, v.op))
                    meta.append((v.idx, group))
                U = ad.encode_actions_free(phrases)
                if U is None:
                    U = ad.encode_actions_ctx(prompt, steps, phrases)
                A.append(U.numpy())
                for vidx, grp in meta:
                    lab["operator"].append(OP_INDEX[problem.vars[vidx].op])
                    lab["depth"].append(depth[vidx])
                    lab["feasible"].append(int(vidx in feasible))
                    lab["paraphrase_group"].append(grp)
                # --- states along the trajectory ---
                allsteps = [step_sentence(problem, v) for v in trace]
                st = ad.encode_states(prompt, allsteps)
                S.append(st.numpy())
                anc = set(problem.query_ancestors)
                done: set[int] = set()
                for k in range(st.shape[0]):
                    slab["step_index"].append(k)
                    slab["remaining"].append(len(anc - done))
                    slab["depth"].append(
                        depth[trace[k]] if k < len(trace) else -1
                    )
                    if k < len(trace):
                        done.add(trace[k])
        np.savez_compressed(
            out_dir / f"embed_{name}.npz",
            actions=np.concatenate(A, 0).astype("float32"),
            states=np.concatenate(S, 0).astype("float32"),
            **{f"action_{k}": np.array(v, dtype="int32") for k, v in lab.items()},
            **{f"state_{k}": np.array(v, dtype="int32") for k, v in slab.items()},
        )
        (out_dir / f"embed_{name}.meta.json").write_text(json.dumps({
            "name": name, "spec": spec, "ckpt": path,
            "info": ad.info(), "n_problems": args.n_problems,
            "op_names": OP_NAMES,
        }, indent=2) + "\n")
        print(f"exported {out_dir / f'embed_{name}.npz'}", flush=True)


# --------------------------------------------------------------------------- #
# stage 2: plot (.venv2 only)
# --------------------------------------------------------------------------- #
def plot(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    try:
        import umap
        HAVE_UMAP = True
    except Exception:
        HAVE_UMAP = False

    in_dir, fig_dir = Path(args.in_dir), Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    coords_out: dict = {}

    for npz_path in sorted(in_dir.glob("embed_*.npz")):
        name = npz_path.stem[len("embed_"):]
        d = np.load(npz_path)
        rng = np.random.default_rng(args.seed)
        for space in ("actions", "states"):
            X = d[space]
            if len(X) > args.max_points:
                keep = rng.choice(len(X), args.max_points, replace=False)
            else:
                keep = np.arange(len(X))
            Xs = X[keep]
            prefix = "action_" if space == "actions" else "state_"
            labels = {k[len(prefix):]: d[k][keep]
                      for k in d.files if k.startswith(prefix)}
            methods = {"tsne": TSNE(
                n_components=2, init="pca", perplexity=min(30, len(Xs) // 4),
                random_state=args.seed,
            )}
            if HAVE_UMAP:
                methods["umap"] = umap.UMAP(
                    n_components=2, random_state=args.seed, n_neighbors=15,
                    min_dist=0.1,
                )
            for mname, model in methods.items():
                Y = model.fit_transform(Xs.astype("float64"))
                coords_out[f"{name}/{space}/{mname}"] = Y.tolist()
                np.savez_compressed(
                    fig_dir / f"coords_{name}_{space}_{mname}.npz",
                    xy=Y.astype("float32"),
                    **{f"label_{k}": v for k, v in labels.items()},
                )
                for lname, lv in labels.items():
                    fig, ax = plt.subplots(figsize=(5.2, 4.6))
                    if lname == "paraphrase_group":
                        base = lv < 0
                        ax.scatter(Y[base, 0], Y[base, 1], s=4, c="0.82",
                                   linewidths=0, label="not in a pair")
                        pair = ~base
                        ax.scatter(Y[pair, 0], Y[pair, 1], s=8,
                                   c=lv[pair] % 20, cmap="tab20",
                                   linewidths=0)
                        for g in np.unique(lv[pair]):
                            m = lv == g
                            if m.sum() == 2:
                                ax.plot(Y[m, 0], Y[m, 1], lw=0.5, c="0.35",
                                        alpha=0.7)
                        ax.set_title(
                            f"{name} / {space} / {mname}\n"
                            "commuted-operand paraphrase partners joined"
                        )
                    else:
                        sc = ax.scatter(Y[:, 0], Y[:, 1], s=5, c=lv,
                                        cmap="viridis", linewidths=0)
                        fig.colorbar(sc, ax=ax, label=lname)
                        ax.set_title(f"{name} / {space} / {mname} / {lname}")
                    ax.set_xticks([]); ax.set_yticks([])
                    fig.tight_layout()
                    stem = fig_dir / f"{name}_{space}_{mname}_{lname}"
                    fig.savefig(f"{stem}.png", dpi=170)
                    fig.savefig(f"{stem}.pdf")
                    plt.close(fig)
                print(f"plotted {name}/{space}/{mname}", flush=True)
    (fig_dir / "coordinates.json").write_text(
        json.dumps({"n_maps": len(coords_out), "keys": sorted(coords_out)},
                   indent=2) + "\n"
    )
    print(f"wrote figures to {fig_dir}")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--random-init", action="append", default=[])
    ap.add_argument("--flat-lm-init", default=None)
    ap.add_argument("--n-problems", type=int, default=250)
    ap.add_argument("--out-dir", default="embed_data")
    ap.add_argument("--in-dir", default="embed_data")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--max-points", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.export:
        export(args)
    if args.plot:
        plot(args)
    if not (args.export or args.plot):
        ap.error("pass --export (torch venv) and/or --plot (.venv2)")


if __name__ == "__main__":
    main()
