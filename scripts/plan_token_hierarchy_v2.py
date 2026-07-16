"""End-to-end flat and hierarchical generation for the hard text project."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from textjepa.data.igsm.dataset import build_vocab
from textjepa.data.igsm.env import SymbolicEnv
from textjepa.data.igsm.render import step_sentence
from textjepa.data.lm import LMDataset
from textjepa.models import MultilevelTokenHierarchyJEPA
from textjepa.models.lm_baseline import DecoderLM


def ln_l1(x, y):
    return (F.layer_norm(x, x.shape[-1:]) - F.layer_norm(y, y.shape[-1:])).abs().mean(-1)


def load_models(hierarchy_ckpt, proposal_ckpt, device):
    hp = torch.load(hierarchy_ckpt, map_location="cpu", weights_only=False)
    hc = OmegaConf.create(hp["cfg"])
    vocab = build_vocab(hc.data.modulus)
    hierarchy = MultilevelTokenHierarchyJEPA(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **hc.model
    ).to(device)
    hierarchy.load_state_dict(hp["model"])
    hierarchy.eval()
    lp = torch.load(proposal_ckpt, map_location="cpu", weights_only=False)
    lc = OmegaConf.create(lp["cfg"])
    proposal = DecoderLM(
        vocab_size=len(vocab), pad_id=vocab.pad_id, **lc.model
    ).to(device)
    proposal.load_state_dict(lp["model"])
    proposal.eval()
    return hierarchy, proposal, vocab, hc


@torch.no_grad()
def beam_spans(model, context, length, beam_width, branch, pad_id, device):
    beams = [([], 0.0)]
    for _ in range(length):
        expanded = []
        for suffix, score in beams:
            sequence = torch.tensor([context + suffix], device=device)
            logits = model(sequence)[0, -1]
            logits[pad_id] = -torch.inf
            logp = logits.log_softmax(-1)
            values, ids = logp.topk(min(branch, logp.numel() - 1))
            expanded.extend([
                (suffix + [int(token)], score - float(value))
                for value, token in zip(values, ids)
            ])
        beams = sorted(expanded, key=lambda item: item[1])[:beam_width]
    return beams


def macro_codes(model, token_ids):
    token_actions = model.token_action(token_ids)
    source = token_actions
    source_stride = 1
    codes = []
    for span, level in zip(model.level_spans, model.levels):
        ratio = span // source_stride
        count = source.shape[1] // ratio
        source = level.action(
            source[:, :count * ratio].reshape(-1, ratio, source.shape[-1])
        ).reshape(token_ids.shape[0], count, -1)
        codes.append(source)
        source_stride = span
    return token_actions, codes


def observed_history(model, prefix_states, generated, level_index):
    """Return causal state/action history at one temporal scale."""
    span = model.level_spans[level_index]
    executed = len(generated) // span
    state_history = prefix_states[:, len(prefix_states[0]) - len(generated) - 1::span]
    state_history = state_history[:, :executed + 1]
    if not executed:
        dim = model.level_dims[level_index]
        return state_history, prefix_states.new_zeros(1, 0, dim)
    past = torch.tensor([generated[:executed * span]], device=prefix_states.device)
    return state_history, macro_codes(model, past)[1][level_index]


@torch.no_grad()
def cem_subgoal(
    level, start, goal, initial, horizon, candidates, iterations, elites, alpha,
    goal_weight, value_weight, support_weight, prior_weight,
    state_history=None, action_history=None,
):
    """Prior-initialized, support-regularized CEM over macro trajectories."""
    prior_mu, prior_logvar = level.action.prior_params(start)
    mean = prior_mu[:, None].expand(1, horizon, -1).clone()
    std = (0.5 * prior_logvar)[:, None].exp().expand_as(mean).clamp_min(0.05)
    elite_n = min(elites, candidates)
    best_cost = None
    best_states = None
    for _ in range(iterations):
        codes = mean + std * torch.randn(
            candidates, horizon, mean.shape[-1], device=start.device
        )
        states = level.predictor.rollout(
            start.expand(candidates, -1), codes,
            state_history=state_history, action_history=action_history,
        )
        terminal = states[:, -1]
        goal_cost = ln_l1(terminal, goal.expand(candidates, -1))
        value = level.value(terminal, initial.expand(candidates, -1))
        rollout_start = torch.cat([
            start.expand(candidates, -1).unsqueeze(1), states[:, :-1]
        ], 1)
        support = F.softplus(-level.support(rollout_start, codes)).mean(1)
        p_mu, p_logvar = level.action.prior_params(rollout_start)
        prior = 0.5 * (
            p_logvar + (codes - p_mu).square() * (-p_logvar).exp()
        ).mean((1, 2))
        cost = (
            goal_weight * goal_cost + value_weight * value
            + support_weight * support + prior_weight * prior
        )
        elite_ids = cost.topk(elite_n, largest=False).indices
        elite = codes[elite_ids]
        new_mean = elite.mean(0, keepdim=True)
        new_std = elite.std(0, unbiased=False, keepdim=True).clamp_min(0.025)
        mean = alpha * mean + (1.0 - alpha) * new_mean
        std = alpha * std + (1.0 - alpha) * new_std
        index = int(cost.argmin())
        if best_cost is None or cost[index] < best_cost:
            best_cost = cost[index]
            best_states = states[index]
    return best_states[0], float(best_cost)


@torch.no_grad()
def top_down_cem_choice(
    model, prompt, generated, spans, lm_costs, goal, args, prefix_states,
):
    """Optimize macro states, descend through scales, then retrieve text."""
    start = prefix_states[:, -1]
    initial = prefix_states[:, len(prompt) - 1]
    target = goal
    cem_costs = []
    for level_index in reversed(range(len(model.levels))):
        level = model.levels[level_index]
        child_span = 1 if level_index == 0 else model.level_spans[level_index - 1]
        horizon = (
            args.horizon if level_index == len(model.levels) - 1
            else model.level_spans[level_index] // child_span
        )
        history_states, history_codes = observed_history(
            model, prefix_states, generated, level_index
        )
        target, cost = cem_subgoal(
            level, start, target, initial, horizon,
            args.cem_candidates, args.cem_iterations, args.cem_elites,
            args.cem_alpha, args.goal_weight, args.value_weight,
            args.support_weight, args.prior_weight,
            history_states, history_codes,
        )
        cem_costs.append(cost)
    candidate_ids = torch.tensor(spans, device=start.device)
    actions = model.token_action(candidate_ids)
    endpoints = model.low_predictor.rollout(start.expand(len(spans), -1), actions)[:, -1]
    reach = ln_l1(endpoints, target.expand(len(spans), -1))
    cost = args.lm_weight * torch.tensor(lm_costs, device=start.device) / len(spans[0])
    cost = cost + args.reachability_weight * reach
    return int(cost.argmin()), {
        "reachability": reach,
        "cem_cost": torch.full_like(reach, sum(cem_costs) / len(cem_costs)),
    }


@torch.no_grad()
def score_candidates(
    model, prompt, generated, candidates, lm_costs, mode, weights,
    oracle_goal=None,
):
    device = next(model.parameters()).device
    n = len(candidates)
    top_span = model.level_spans[-1]
    horizon = len(candidates[0]) // top_span
    prefix = torch.tensor([prompt + generated], device=device)
    prefix_states = model.encoder(prefix)
    prompt_state = prefix_states[:, len(prompt) - 1]
    current = prefix_states[:, -1]
    candidate_ids = torch.tensor(candidates, device=device)
    token_actions, level_codes = macro_codes(model, candidate_ids)
    goal = oracle_goal if oracle_goal is not None else model.goal_head(prompt_state)

    if mode == "flat":
        endpoint = model.low_predictor.rollout(
            current.expand(n, -1), token_actions
        )[:, -1]
        value = model.low_value(endpoint, prompt_state.expand(n, -1))
        goal_cost = ln_l1(endpoint, goal.expand(n, -1))
        return (
            weights["lm"] * torch.tensor(lm_costs, device=device) / len(candidates[0])
            + weights["value"] * value
            + weights["goal"] * goal_cost
        ), {"value": value, "goal": goal_cost}

    top = model.levels[-1]
    codes = level_codes[-1]
    executed_steps = len(generated) // top_span
    if executed_steps:
        observed = prefix_states[:, len(prompt) - 1::top_span]
        history_states = observed[:, :executed_steps + 1]
        past_ids = torch.tensor(
            [generated[:executed_steps * top_span]], device=device
        )
        _, past_codes = macro_codes(model, past_ids)
        history_codes = past_codes[-1]
    else:
        history_states = prompt_state.unsqueeze(1)
        history_codes = codes[:, :0]
    predictions = top.predictor.rollout(
        current.expand(n, -1),
        codes,
        state_history=history_states.expand(n, -1, -1),
        action_history=history_codes.expand(n, -1, -1),
    )
    endpoint = predictions[:, -1]
    first_pred = predictions[:, 0]
    value = top.value(endpoint, prompt_state.expand(n, -1))
    goal_cost = ln_l1(endpoint, goal.expand(n, -1))
    support = F.softplus(-top.support(
        current.expand(n, -1), codes[:, 0]
    ))
    low_endpoint = model.low_predictor.rollout(
        current.expand(n, -1), token_actions[:, :top_span]
    )[:, -1]
    reachability = ln_l1(first_pred, low_endpoint)
    total = (
        weights["lm"] * torch.tensor(lm_costs, device=device) / len(candidates[0])
        + weights["value"] * value
        + weights["goal"] * goal_cost
        + weights["support"] * support
        + weights["reachability"] * reachability
    )
    return total, {
        "value": value, "goal": goal_cost, "support": support,
        "reachability": reachability,
    }


def replay_validity(problem, vocab, generated):
    env = SymbolicEnv(problem)
    period = vocab.token_to_id["."]
    start = invalid = valid = 0
    for end, token in enumerate(generated):
        if token != period:
            continue
        sentence = generated[start:end + 1]
        start = end + 1
        match = None
        for action in env.feasible_actions():
            if vocab.encode(step_sentence(problem, action)) == sentence:
                match = action
                break
        if match is None:
            invalid += 1
        else:
            valid += 1
            env.step(match)
        if env.solved:
            break
    return env.solved, valid, invalid


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchy-ckpt", required=True)
    parser.add_argument("--proposal-ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--mode", choices=["lm", "flat", "hierarchy", "latent-cem"],
        default="hierarchy",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--beam", type=int, default=16)
    parser.add_argument("--branch", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--max-macros", type=int, default=16)
    parser.add_argument("--oracle-goal", action="store_true")
    parser.add_argument("--lm-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--goal-weight", type=float, default=1.0)
    parser.add_argument("--support-weight", type=float, default=0.1)
    parser.add_argument("--reachability-weight", type=float, default=1.0)
    parser.add_argument("--prior-weight", type=float, default=0.1)
    parser.add_argument("--cem-candidates", type=int, default=1000)
    parser.add_argument("--cem-iterations", type=int, default=20)
    parser.add_argument("--cem-elites", type=int, default=100)
    parser.add_argument("--cem-alpha", type=float, default=0.1)
    parser.add_argument(
        "--output-tag", default="",
        help="optional suffix for distinct exploratory evaluation budgets",
    )
    args = parser.parse_args()
    model, proposal, vocab, cfg = load_models(
        args.hierarchy_ckpt, args.proposal_ckpt, args.device
    )
    dataset = LMDataset(
        vocab, size=max(args.episodes, 100), seed=cfg.data.val_seed,
        modulus=cfg.data.modulus, n_vars_range=tuple(cfg.data.n_vars_range),
        leaf_prob=cfg.data.leaf_prob, steps_range=tuple(cfg.data.steps_range),
        distractor_prob=cfg.data.distractor_prob,
        max_distractors=cfg.data.max_distractors,
    )
    weights = {
        "lm": args.lm_weight, "value": args.value_weight,
        "goal": args.goal_weight, "support": args.support_weight,
        "reachability": args.reachability_weight,
    }
    totals = {
        "success": 0, "valid_sentences": 0, "invalid_sentences": 0,
        "proposal_reference_hits": 0, "reference_selected": 0,
        "reference_available_not_selected": 0, "decisions": 0,
        "chosen_value": 0.0, "chosen_goal": 0.0,
        "chosen_support": 0.0, "chosen_reachability": 0.0,
    }
    top_span = model.level_spans[-1]
    for index in range(args.episodes):
        item = dataset[index]
        problem, _ = dataset.igsm.problem(index)
        prompt = item["tokens"][:item["prompt_len"]]
        reference = item["tokens"][item["prompt_len"]:]
        generated = []
        oracle_goal = None
        if args.oracle_goal:
            full = torch.tensor([item["tokens"]], device=args.device)
            oracle_goal = model.teacher(full)[:, -1]
        for _ in range(args.max_macros):
            proposal_length = (
                model.level_spans[0] if args.mode == "latent-cem"
                else top_span * args.horizon
            )
            candidates = beam_spans(
                proposal, prompt + generated, proposal_length,
                args.beam, args.branch, vocab.pad_id, args.device,
            )
            spans, lm_costs = zip(*candidates)
            reference_next = reference[len(generated):len(generated) + len(spans[0])]
            reference_available = reference_next in spans
            totals["proposal_reference_hits"] += int(reference_available)
            totals["decisions"] += 1
            if args.mode == "lm":
                choice, diagnostics = 0, {}
            elif args.mode == "latent-cem":
                prefix = torch.tensor([prompt + generated], device=args.device)
                prefix_states = model.encoder(prefix)
                prompt_state = prefix_states[:, len(prompt) - 1]
                goal = oracle_goal if oracle_goal is not None else model.goal_head(prompt_state)
                choice, diagnostics = top_down_cem_choice(
                    model, prompt, generated, list(spans), list(lm_costs),
                    goal, args, prefix_states,
                )
            else:
                costs, diagnostics = score_candidates(
                    model, prompt, generated, list(spans), list(lm_costs),
                    args.mode, weights, oracle_goal,
                )
                choice = int(costs.argmin())
            selected_reference = spans[choice] == reference_next
            totals["reference_selected"] += int(selected_reference)
            totals["reference_available_not_selected"] += int(
                reference_available and not selected_reference
            )
            execute = model.level_spans[0] if args.mode == "latent-cem" else top_span
            generated.extend(spans[choice][:execute])
            for name in ("value", "goal", "support", "reachability"):
                if name in diagnostics:
                    totals[f"chosen_{name}"] += float(diagnostics[name][choice])
            solved, _, _ = replay_validity(problem, vocab, generated)
            if solved:
                break
        solved, valid, invalid = replay_validity(problem, vocab, generated)
        totals["success"] += int(solved)
        totals["valid_sentences"] += valid
        totals["invalid_sentences"] += invalid
    decisions = max(totals["decisions"], 1)
    result = {
        "mode": args.mode,
        "success": totals["success"] / args.episodes,
        "valid_sentences_per_episode": totals["valid_sentences"] / args.episodes,
        "invalid_sentences_per_episode": totals["invalid_sentences"] / args.episodes,
        "proposal_reference_recall": totals["proposal_reference_hits"] / decisions,
        "reference_selection_rate": totals["reference_selected"] / decisions,
        "ranking_failure_rate": (
            totals["reference_available_not_selected"]
            / max(totals["proposal_reference_hits"], 1)
        ),
        **{f"mean_{name}": totals[f"chosen_{name}"] / decisions
           for name in ("value", "goal", "support", "reachability")},
        "episodes": args.episodes, "beam": args.beam,
        "horizon": args.horizon, "oracle_goal": args.oracle_goal,
        "max_macros": args.max_macros,
        "cem_candidates": args.cem_candidates if args.mode == "latent-cem" else None,
        "cem_iterations": args.cem_iterations if args.mode == "latent-cem" else None,
    }
    suffix = "_oracle" if args.oracle_goal else ""
    if args.mode == "latent-cem":
        suffix += f"_c{args.cem_candidates}_i{args.cem_iterations}"
    if args.output_tag:
        suffix += f"_{args.output_tag}"
    dest = Path(args.hierarchy_ckpt).parent / (
        f"planning_{args.mode}_b{args.beam}_h{args.horizon}{suffix}.json"
    )
    dest.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
