"""Merge sharded plan_flat outputs (produced with --episode-start) into one JSON.

Usage: merge_plan_shards.py --out merged.json shard0.json shard1.json ...
Shards must cover disjoint episode ranges of the same protocol; summaries are
recomputed from the concatenated per-episode records, so the merged file is
equivalent to an unsharded run over the union of episodes.
"""
import argparse, json, sys
sys.path.insert(0, "src")
from textjepa.planning.flat_search import summarize


def _sum_tree(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _sum_tree(a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b)}
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + b
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("shards", nargs="+")
    args = ap.parse_args()
    datas = []
    for f in args.shards:
        with open(f) as fh:
            datas.append((f, json.load(fh)))
    datas.sort(key=lambda t: t[1].get("protocol", {}).get("episode_start", 0))
    keyprot = None
    episodes, rand_, first_, compute = [], [], [], None
    for f, d in datas:
        prot = dict(d.get("protocol", {}))
        start = prot.pop("episode_start", 0)
        if keyprot is None:
            keyprot = prot
        elif prot != keyprot:
            raise SystemExit(f"protocol mismatch in {f}")
        episodes += d["episodes"]
        rand_ += d.get("episodes_random", [])
        first_ += d.get("episodes_first_feasible", [])
        compute = d.get("compute") if compute is None else _sum_tree(compute, d.get("compute", {}))
    out = {
        "compute": compute,
        "latent_planner": summarize(episodes),
        "random_policy": summarize(rand_),
        "first_feasible_policy": summarize(first_),
        "episodes": episodes,
        "episodes_random": rand_,
        "episodes_first_feasible": first_,
        "protocol": {**(keyprot or {}), "episode_start": 0, "merged_shards": len(datas)},
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    n = len(episodes)
    s = sum(e["solved"] for e in episodes)
    print(f"merged {len(datas)} shards -> {args.out}: n={n} solved={s/max(n,1):.3f}")


if __name__ == "__main__":
    main()
