"""No-LM diagnostics for oracle goals and the primitive inverse interface."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import step_sentence
from textjepa.data.lm import LMDataset
from textjepa.models import MultilevelTokenHierarchyJEPA
from textjepa.planning.token_cem import categorical_cem, latent_l1


def load(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(payload["cfg"])
    vocab = build_vocab(cfg.data.modulus)
    model = MultilevelTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **cfg.model
    ).to(device)
    model.load_state_dict(payload["model"]); model.eval()
    dataset = LMDataset(
        vocab, size=10000, seed=cfg.data.val_seed, modulus=cfg.data.modulus,
        n_vars_range=tuple(cfg.data.n_vars_range), leaf_prob=cfg.data.leaf_prob,
        steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    return model, vocab, dataset


def alternative_trace(problem, seed):
    rng, env, trace = random.Random(seed), SymbolicEnv(problem), []
    while not env.solved:
        action = rng.choice(env.feasible_actions())
        trace.append(action); env.step(action)
    return trace


def summarize(values):
    return {
        "mean": float(np.mean(values)),
        "p90": float(np.quantile(values, .9)),
        "n": len(values),
    } if values else {"mean": float("nan"), "p90": float("nan"), "n": 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    model, vocab, dataset = load(args.ckpt, args.device)
    inverse = {h: {k: [] for k in (
        "token_accuracy", "exact_span", "selected_goal_cost",
        "reference_model_cost", "selected_reference_gap", "period_rate",
    )} for h in args.horizons}
    terminal_pairs, terminal_goals = [], []
    with torch.no_grad():
        for index in range(args.examples):
            item = dataset[index]
            problem, _ = dataset.igsm.problem(index)
            prompt = item["tokens"][:item["prompt_len"]]
            reference = item["tokens"][item["prompt_len"]:]
            alternatives = []
            for alt in range(3):
                trace = alternative_trace(problem, f"{index}:{alt}:terminal")
                steps = [
                    token for action in trace
                    for token in vocab.encode(step_sentence(problem, action))
                ]
                full = torch.tensor([prompt + steps], device=args.device)
                alternatives.append(model.teacher(full)[0, -1])
            alt_stack = torch.stack(alternatives)
            terminal_pairs.extend([
                float(latent_l1(alt_stack[a:a + 1], alt_stack[b:b + 1]))
                for a in range(3) for b in range(a + 1, 3)
            ])
            terminal_goals.append(alt_stack[0])

            anchors = sorted(set([0, len(reference) // 2]))
            full_reference = torch.tensor([prompt + reference], device=args.device)
            target_states = model.teacher(full_reference)
            for anchor in anchors:
                for horizon in args.horizons:
                    if anchor + horizon > len(reference):
                        continue
                    prefix_ids = prompt + reference[:anchor]
                    prefix = torch.tensor([prefix_ids], device=args.device)
                    states = model.encoder(prefix)
                    start = states[:, -1]
                    history_states = states[:, item["prompt_len"] - 1:]
                    history_actions = (
                        model.token_action(torch.tensor(
                            [reference[:anchor]], device=args.device
                        )) if anchor else states.new_zeros(1, 0, model.d_action)
                    )
                    target = target_states[:, item["prompt_len"] + anchor + horizon - 1]
                    def rollout(ids):
                        return model.low_predictor.rollout(
                            start.expand(len(ids), -1), model.token_action(ids),
                            state_history=history_states,
                            action_history=history_actions,
                        )
                    result = categorical_cem(
                        rollout, target, horizon, len(vocab),
                        candidates=args.candidates, iterations=args.iterations,
                        elites=max(8, args.candidates // 8),
                        forbidden=(vocab.pad_id, vocab.token_to_id[vocab.UNK]),
                    )
                    truth = torch.tensor(reference[anchor:anchor + horizon], device=args.device)
                    truth_states = rollout(truth[None])[0]
                    ref_cost = float(latent_l1(truth_states[-1:], target))
                    selected = result.actions
                    inverse[horizon]["token_accuracy"].append(float((selected == truth).float().mean()))
                    inverse[horizon]["exact_span"].append(float(torch.equal(selected, truth)))
                    inverse[horizon]["selected_goal_cost"].append(result.cost)
                    inverse[horizon]["reference_model_cost"].append(ref_cost)
                    inverse[horizon]["selected_reference_gap"].append(result.cost - ref_cost)
                    inverse[horizon]["period_rate"].append(float(
                        (selected == vocab.token_to_id["."]).float().mean()
                    ))
    goals = torch.stack(terminal_goals)
    across = []
    for index in range(len(goals) - 1):
        across.append(float(latent_l1(goals[index:index + 1], goals[index + 1:index + 2])))
    result = {
        "terminal_goal_invariance": {
            "same_problem_alternative_valid_traces": summarize(terminal_pairs),
            "different_problems": summarize(across),
            "within_over_between": float(np.mean(terminal_pairs) / max(np.mean(across), 1e-12)),
        },
        "primitive_inverse_cem": {
            str(h): {key: summarize(values) for key, values in metrics.items()}
            for h, metrics in inverse.items()
        },
        "uses_auxiliary_lm": False,
        "args": vars(args),
    }
    destination = Path(args.ckpt).parent / "token_planning_interface_audit.json"
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
