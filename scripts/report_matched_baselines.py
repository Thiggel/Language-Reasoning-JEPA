"""Create the live exact-interface policy comparison table."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def success(run: str, filename: str, key_prefix: str) -> float | None:
    source = RUNS / run / filename
    if not source.exists():
        return None
    payload = json.loads(source.read_text())
    return next(
        (float(metrics["success"]) for key, metrics in payload.items()
         if key.startswith(key_prefix)),
        None,
    )


def seeded_success(
    run: str, filename: str, key_prefix: str
) -> list[float]:
    values = [
        success(run + suffix, filename, key_prefix)
        for suffix in ("", "_s2", "_s3")
    ]
    # Do not report a changing partial-seed mean. Until all three final seeds
    # exist, retain the original seed-1 validation result.
    if all(value is not None for value in values):
        return [float(value) for value in values]
    return [] if values[0] is None else [float(values[0])]


def suffix(split: str) -> str:
    return "" if split == "val" else f"_{split}"


def latent_success(run: str, slack: int, split: str) -> list[float]:
    return seeded_success(
        run, f"plan_slack{slack}_look1{suffix(split)}.json", "latent_planner"
    )


def lm_success(run: str, slack: int, split: str) -> list[float]:
    return seeded_success(
        run, f"plan_slack{slack}_lm_intent{suffix(split)}.json", "lm_intent"
    )


def sentence_success(
    run: str, slack: int, score: str, split: str
) -> list[float]:
    return seeded_success(
        run,
        f"plan_slack{slack}_sentlm_intent_{score}{suffix(split)}.json",
        f"sentlm_intent_{score}",
    )


def baseline_success(
    run: str, slack: int, name: str, split: str
) -> float | None:
    source = RUNS / run / f"plan_slack{slack}_look1{suffix(split)}.json"
    if not source.exists():
        return None
    return json.loads(source.read_text()).get(name, {}).get("success")


def parameters(run: str) -> str:
    log = ROOT / f"runs_{run}.log"
    if log.exists():
        matches = re.findall(
            r"(?:model|LM|SentenceLM) parameters:\s*([0-9.]+)M",
            log.read_text(errors="replace"),
            flags=re.IGNORECASE,
        )
        if matches:
            return f"{float(matches[-1]):.2f}M"
    return "---"


def fmt(value: float | list[float] | None) -> str:
    if value is None or value == []:
        return "---"
    if isinstance(value, list):
        if len(value) == 1:
            return f"{value[0]:.3f}"
        return f"{statistics.mean(value):.3f} ± {statistics.stdev(value):.3f}"
    return f"{value:.3f}"


def exact(values: float | list[float] | None) -> str:
    if not isinstance(values, list) or len(values) != 3:
        return "---"
    return " / ".join(f"{value:.3f}" for value in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("stylized", "official"),
                        default="stylized")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--jepa")
    parser.add_argument("--out")
    args = parser.parse_args()
    official = args.domain == "official"
    jepa = args.jepa or (
        "real_latent_goal_h2_r1" if official else "disc_latent_goal_h2_r1"
    )
    lm_run = "lm_intent_faithful" if official else "lm_intent"
    sentence_run = (
        "sentlm_intent_faithful" if official else "sentlm_intent"
    )
    sentence_latent_run = (
        "sentlm_latent_intent_faithful"
        if official else "sentlm_latent_intent"
    )
    symbolic_run = (
        "real_rank_fresh_noldad"
        if official else "disc_fresh_symbolic_rank_k2"
    )
    # Random and oracle policies depend only on the fixed validation problems,
    # not on the learned checkpoint.  Use the symbolic reference while the
    # selected non-symbolic model is still training so those controls are not
    # needlessly shown as missing.
    baseline_run = (
        jepa
        if (
            RUNS / jepa
            / f"plan_slack0_look1{suffix(args.split)}.json"
        ).exists()
        else symbolic_run
    )
    rows = [
        (
            "random feasible policy", "none", "shared feasible intents", "---",
            baseline_success(baseline_run, 0, "random_policy", args.split),
            baseline_success(baseline_run, 2, "random_policy", args.split),
        ),
        (
            "token autoregressive policy", "intent-token likelihood",
            "shared feasible intents", parameters(lm_run),
            lm_success(lm_run, 0, args.split),
            lm_success(lm_run, 2, args.split),
        ),
        (
            "sentence autoregressive policy", "intent reconstruction likelihood",
            "shared feasible intents", parameters(sentence_run),
            sentence_success(sentence_run, 0, "decoder", args.split),
            sentence_success(sentence_run, 2, "decoder", args.split),
        ),
        (
            "sentence policy + latent prediction", "intent likelihood",
            "shared feasible intents", parameters(sentence_latent_run),
            sentence_success(
                sentence_latent_run, 0, "decoder", args.split
            ),
            sentence_success(
                sentence_latent_run, 2, "decoder", args.split
            ),
        ),
        (
            "sentence policy + latent prediction", "latent distance",
            "shared feasible intents", parameters(sentence_latent_run),
            sentence_success(
                sentence_latent_run, 0, "latent", args.split
            ),
            sentence_success(
                sentence_latent_run, 2, "latent", args.split
            ),
        ),
        (
            "non-symbolic latent dynamics", "learned value energy",
            "shared feasible intents", parameters(jepa),
            latent_success(jepa, 0, args.split),
            latent_success(jepa, 2, args.split),
        ),
        (
            "symbolic-preference latent reference", "learned value energy",
            "shared feasible intents", parameters(symbolic_run),
            latent_success(symbolic_run, 0, args.split),
            latent_success(symbolic_run, 2, args.split),
        ),
        (
            "oracle policy", "exact necessary action", "environment state", "---",
            1.0, 1.0,
        ),
    ]
    lines = [
        f"# Matched intent-policy baselines: {args.domain} iGSM ({args.split})",
        "",
        "Seed 1 is shown until all three final seeds for a row are complete;",
        "completed replications are reported as mean ± sample standard",
        "deviation. Every learned policy ranks the same currently feasible",
        "outcome-free intent phrases. A dash is an active or pending run.",
        "",
        "| method | selection score | candidate information | parameters | strict | slack-2 |",
        "|---|---|---|---:|---:|---:|",
    ]
    lines.extend(
        f"| {method} | {score} | {interface} | {params} | {fmt(strict)} | {fmt(slack)} |"
        for method, score, interface, params, strict, slack in rows
    )
    replicated = [
        (method, score, strict, slack)
        for method, score, _interface, _params, strict, slack in rows
        if isinstance(strict, list) and len(strict) == 3
        and isinstance(slack, list) and len(slack) == 3
    ]
    if replicated:
        lines.extend([
            "",
            "## Exact final-seed values",
            "",
            "Main-table replicated entries are mean ± sample standard",
            "deviation. Values below are seeds 1 / 2 / 3.",
            "",
            "| method | selection score | strict seeds | slack-2 seeds |",
            "|---|---|---:|---:|",
        ])
        lines.extend(
            f"| {method} | {score} | {exact(strict)} | {exact(slack)} |"
            for method, score, strict, slack in replicated
        )
    if args.out:
        destination = ROOT / args.out
    else:
        domain_suffix = "_official" if official else ""
        split_suffix = "_test" if args.split == "test" else ""
        destination = ROOT / (
            f"runs/matched_baselines{domain_suffix}{split_suffix}.md"
        )
    destination.write_text("\n".join(lines) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
